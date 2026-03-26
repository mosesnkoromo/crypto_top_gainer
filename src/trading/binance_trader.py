"""
src/trading/binance_trader.py — v3
Full automated execution for both SPOT and FUTURES on Binance.

Fixes vs v2:
  - _daily_loss now tracked correctly after each trade
  - prec() variable name 's' no longer shadows loop variable
  - cancel_all_orders uses correct separate paths for GET vs DELETE
  - OCO validation: checks TP1 > entry > SL before placing
  - hmac.new → hmac.new (was already correct, confirmed)
"""
import hmac, hashlib, time, requests
from urllib.parse import urlencode
from dataclasses import dataclass
from typing import Literal
from src.utils.logger import get_logger

log = get_logger(__name__)

_SPOT_BASE = "https://api.binance.com"
_SPOT_TEST = "https://testnet.binance.vision"
_FUT_BASE  = "https://fapi.binance.com"
_FUT_TEST  = "https://testnet.binancefuture.com"


@dataclass
class SymbolInfo:
    price_precision: int   = 6
    qty_precision:   int   = 2
    min_qty:         float = 0.001
    min_notional:    float = 5.0
    tick_size:       float = 0.0001
    step_size:       float = 0.001


@dataclass
class TradeResult:
    success:        bool
    symbol:         str
    side:           str
    qty:            float
    entry_price:    float
    mode:           str   = "spot"
    leverage:       int   = 1
    position_usdt:  float = 0.0
    risk_usdt:      float = 0.0
    entry_order_id: str   = ""
    oco_id:         str   = ""
    tp1_order_id:   str   = ""
    tp2_order_id:   str   = ""
    tp3_order_id:   str   = ""
    sl_order_id:    str   = ""
    error:          str   = ""


class BinanceTrader:

    def __init__(self, api_key: str, api_secret: str,
                 mode: Literal["spot", "futures"] = "spot",
                 live: bool = False,
                 risk_pct: float = 2.0,
                 daily_loss_limit_pct: float = 6.0,
                 max_trades_per_day: int = 10):
        self._key          = api_key
        self._secret       = api_secret
        self._mode         = mode
        self._live         = live
        self._risk         = risk_pct
        self._loss_limit   = daily_loss_limit_pct
        self._max_trades   = max_trades_per_day
        self._daily_loss   = 0.0   # % lost today — updated after each SL hit
        self._daily_trades = 0
        self._sym_cache: dict[str, SymbolInfo] = {}

        self._base = (_FUT_BASE if live else _FUT_TEST) if mode == "futures" \
                     else (_SPOT_BASE if live else _SPOT_TEST)

        log.info("BinanceTrader: %s %s | risk=%.1f%% | loss_limit=%.1f%%",
                 mode.upper(), "LIVE" if live else "TESTNET",
                 risk_pct, daily_loss_limit_pct)

    # ── Public ──────────────────────────────────────────────────

    def execute_signal(self, signal, balance_usdt: float) -> TradeResult:
        sym = signal.symbol
        ok, reason = self._pre_flight(balance_usdt)
        if not ok:
            log.warning("Trade blocked [%s]: %s", sym, reason)
            return TradeResult(False, sym, signal.signal, 0, signal.price,
                               mode=self._mode, error=reason)

        info    = self._get_symbol_info(sym)
        sl_dist = abs(signal.price - signal.sl) / max(signal.price, 1e-10)
        if sl_dist < 0.001:
            return TradeResult(False, sym, signal.signal, 0, signal.price,
                               mode=self._mode, error="SL too close to entry")

        risk_usdt = balance_usdt * (self._risk / 100)
        pos_usdt  = min(risk_usdt / sl_dist, balance_usdt * 0.25)
        if pos_usdt < info.min_notional:
            return TradeResult(False, sym, signal.signal, 0, signal.price,
                               mode=self._mode,
                               error=f"Position ${pos_usdt:.2f} below Binance minimum ${info.min_notional}")

        qty = self._fmt_qty(pos_usdt / signal.price, info)
        if qty <= 0:
            return TradeResult(False, sym, signal.signal, 0, signal.price,
                               mode=self._mode, error="Qty rounds to zero")

        log.info("AUTO → %s %s mode=%s qty=%s entry=%.6g TP1=%.6g TP2=%.6g TP3=%.6g SL=%.6g",
                 signal.signal, sym, self._mode, qty,
                 signal.price, signal.tp1, signal.tp2, signal.tp3, signal.sl)
        try:
            r = self._spot(sym, signal, qty, info, risk_usdt, pos_usdt) \
                if self._mode == "spot" \
                else self._futures(sym, signal, qty, info, risk_usdt, pos_usdt)
            if r.success:
                self._daily_trades += 1
            return r
        except Exception as e:
            log.error("Execution error [%s]: %s", sym, e, exc_info=True)
            return TradeResult(False, sym, signal.signal, qty, signal.price,
                               mode=self._mode, error=str(e))

    def record_loss(self, loss_pct: float):
        """Call this when a SL is hit so daily_loss limit is tracked."""
        self._daily_loss += abs(loss_pct)
        log.info("Daily loss updated: %.1f%% / %.1f%% limit",
                 self._daily_loss, self._loss_limit)
        if self._daily_loss >= self._loss_limit:
            log.warning("Daily loss limit %.1f%% reached — auto-trade will pause",
                        self._loss_limit)

    def cancel_all_orders(self, symbol: str = None) -> dict:
        cancelled, errors = [], []
        try:
            # Separate paths for GET (list) and DELETE (cancel)
            if self._mode == "futures":
                get_path = "/fapi/v1/openOrders"
                del_path = "/fapi/v1/allOpenOrders"
            else:
                get_path = "/api/v3/openOrders"
                del_path = "/api/v3/openOrders"   # same path, DELETE method

            if symbol:
                self._req("DELETE", del_path, {"symbol": symbol})
                cancelled.append(symbol)
            else:
                open_orders = self._req("GET", get_path, {}) or []
                syms = set(o["symbol"] for o in open_orders)
                for s in syms:
                    try:
                        self._req("DELETE", del_path, {"symbol": s})
                        cancelled.append(s)
                    except Exception as e:
                        errors.append(f"{s}: {e}")
            log.warning("EMERGENCY STOP — cancelled orders for: %s", cancelled)
        except Exception as e:
            log.error("Emergency stop error: %s", e)
            errors.append(str(e))
        return {"cancelled": cancelled, "errors": errors}

    def get_balance(self) -> dict:
        """
        Returns dict with wallet_balance, available_balance, unrealised_pnl.
        wallet_balance = total USDT deposited (stable, use for display).
        available_balance = free margin (fluctuates with open positions).
        """
        result = {"wallet_balance": 0.0, "available_balance": 0.0,
                  "unrealised_pnl": 0.0, "error": ""}
        try:
            if self._mode == "futures":
                assets = self._req("GET", "/fapi/v2/balance", {}) or []
                for a in assets:
                    if a.get("asset") == "USDT":
                        result["wallet_balance"]    = float(a.get("balance",           0))
                        result["available_balance"] = float(a.get("availableBalance",  0))
                        result["unrealised_pnl"]    = float(a.get("crossUnPnl",        0))
                        log.info("Futures wallet=$%.2f available=$%.2f pnl=%+.2f",
                                 result["wallet_balance"],
                                 result["available_balance"],
                                 result["unrealised_pnl"])
                        return result
            else:
                acct = self._req("GET", "/api/v3/account", {}) or {}
                for a in acct.get("balances", []):
                    if a["asset"] == "USDT":
                        result["wallet_balance"]    = float(a["free"]) + float(a["locked"])
                        result["available_balance"] = float(a["free"])
                        return result
        except Exception as e:
            log.error("Balance error: %s", e)
            result["error"] = str(e)
        return result

    def get_open_orders(self, symbol: str = None) -> list:
        try:
            path   = "/fapi/v1/openOrders" if self._mode == "futures" else "/api/v3/openOrders"
            params = {"symbol": symbol} if symbol else {}
            return self._req("GET", path, params) or []
        except Exception as e:
            log.error("Open orders error: %s", e)
            return []

    def get_positions(self) -> list:
        if self._mode != "futures":
            return []
        try:
            pos = self._req("GET", "/fapi/v2/positionRisk", {}) or []
            return [p for p in pos if float(p.get("positionAmt", 0)) != 0]
        except Exception as e:
            log.error("Positions error: %s", e)
            return []

    def reset_daily_counters(self):
        self._daily_trades = 0
        self._daily_loss   = 0.0
        log.info("Daily counters reset")

    # ── Spot ────────────────────────────────────────────────────

    def _spot(self, sym, signal, qty, info, risk_usdt, pos_usdt) -> TradeResult:
        if signal.signal != "BUY":
            return TradeResult(False, sym, signal.signal, qty, signal.price, mode="spot",
                               error="SELL signals not supported on spot. Enable futures mode.")

        # Validate levels before placing any order
        if signal.tp1 <= signal.price:
            return TradeResult(False, sym, signal.signal, qty, signal.price, mode="spot",
                               error=f"TP1 ({signal.tp1}) must be above entry ({signal.price})")
        if signal.sl >= signal.price:
            return TradeResult(False, sym, signal.signal, qty, signal.price, mode="spot",
                               error=f"SL ({signal.sl}) must be below entry ({signal.price})")

        # Market BUY
        buy = self._req("POST", "/api/v3/order", {
            "symbol": sym, "side": "BUY", "type": "MARKET", "quantity": qty,
        })
        if not buy or "orderId" not in buy:
            raise Exception(f"Market BUY rejected: {buy}")
        fills      = buy.get("fills", [])
        fill_price = float(fills[0]["price"]) if fills else signal.price
        entry_id   = str(buy["orderId"])
        log.info("Spot BUY: %s qty=%s @ %.6g id=%s", sym, qty, fill_price, entry_id)

        q_oco = self._fmt_qty(qty * 0.40, info)
        q_tp2 = self._fmt_qty(qty * 0.35, info)
        q_tp3 = self._fmt_qty(qty * 0.25, info)
        tp1_p = self._fp(signal.tp1, info)
        tp2_p = self._fp(signal.tp2, info)
        tp3_p = self._fp(signal.tp3, info)
        sl_p  = self._fp(signal.sl,  info)
        sl_lim= self._fp(signal.sl * 0.998, info)

        # OCO: TP1 limit + SL stop-limit
        oco_id = tp1_id = sl_id = ""
        if q_oco >= info.min_qty:
            oco = self._req("POST", "/api/v3/order/oco", {
                "symbol": sym, "side": "SELL", "quantity": q_oco,
                "price": tp1_p,              # take-profit limit
                "stopPrice": sl_p,           # stop trigger
                "stopLimitPrice": sl_lim,    # stop limit fill
                "stopLimitTimeInForce": "GTC",
            })
            if oco:
                oco_id = str(oco.get("orderListId", ""))
                orders = oco.get("orders", [])
                tp1_id = str(orders[0]["orderId"]) if orders else ""
                sl_id  = str(orders[1]["orderId"]) if len(orders) > 1 else ""
                log.info("OCO placed: %s TP1=%s SL=%s oco=%s", sym, tp1_id, sl_id, oco_id)
            else:
                log.error("OCO FAILED for %s — position has NO stop-loss!", sym)

        tp2_id = ""
        if q_tp2 >= info.min_qty:
            r = self._req("POST", "/api/v3/order", {
                "symbol": sym, "side": "SELL", "type": "LIMIT",
                "timeInForce": "GTC", "quantity": q_tp2, "price": tp2_p,
            })
            tp2_id = str((r or {}).get("orderId", ""))
            log.info("TP2: %s qty=%s @ %s id=%s", sym, q_tp2, tp2_p, tp2_id)

        tp3_id = ""
        if q_tp3 >= info.min_qty:
            r = self._req("POST", "/api/v3/order", {
                "symbol": sym, "side": "SELL", "type": "LIMIT",
                "timeInForce": "GTC", "quantity": q_tp3, "price": tp3_p,
            })
            tp3_id = str((r or {}).get("orderId", ""))
            log.info("TP3: %s qty=%s @ %s id=%s", sym, q_tp3, tp3_p, tp3_id)

        return TradeResult(True, sym, "BUY", qty, fill_price, mode="spot",
                           position_usdt=round(pos_usdt, 2), risk_usdt=round(risk_usdt, 2),
                           entry_order_id=entry_id, oco_id=oco_id,
                           tp1_order_id=tp1_id, tp2_order_id=tp2_id,
                           tp3_order_id=tp3_id, sl_order_id=sl_id)

    # ── Futures ─────────────────────────────────────────────────

    def _futures(self, sym, signal, qty, info, risk_usdt, pos_usdt) -> TradeResult:
        is_long    = signal.signal == "BUY"
        side       = "BUY"  if is_long else "SELL"
        ps_side    = "LONG" if is_long else "SHORT"
        close_side = "SELL" if is_long else "BUY"

        # Validate levels
        if is_long and signal.tp1 <= signal.price:
            return TradeResult(False, sym, signal.signal, qty, signal.price, mode="futures",
                               error=f"TP1 ({signal.tp1}) must be above entry for LONG")
        if not is_long and signal.tp1 >= signal.price:
            return TradeResult(False, sym, signal.signal, qty, signal.price, mode="futures",
                               error=f"TP1 ({signal.tp1}) must be below entry for SHORT")

        # Leverage
        lev = 3 if getattr(signal, "btc_score", 50) >= 62 else 2
        self._req("POST", "/fapi/v1/leverage", {"symbol": sym, "leverage": lev})

        # Hedge mode — detect current mode first, only switch if needed
        # Cannot switch while position is open (Binance error -4059)
        try:
            mode_info = self._pub("GET", "/fapi/v1/positionSide/dual", {}) or {}
            already_hedge = mode_info.get("dualSidePosition", False)
            if not already_hedge:
                switch_result = self._req("POST", "/fapi/v1/positionSide/dual",
                                          {"dualSidePosition": "true"})
                if switch_result is None:
                    log.warning("Could not switch to hedge mode — likely have open positions. "
                                "Trading in one-way mode. Close all positions first to enable hedge.")
        except Exception as e:
            log.debug("Hedge mode check: %s", e)

        # Market entry
        entry = self._req("POST", "/fapi/v1/order", {
            "symbol": sym, "side": side,
            "positionSide": ps_side, "type": "MARKET", "quantity": qty,
        })
        if not entry or "orderId" not in entry:
            raise Exception(f"Futures {side} rejected: {entry}")
        entry_id   = str(entry["orderId"])
        fill_price = float(entry.get("avgPrice", signal.price)) or signal.price
        log.info("Futures %s %s: qty=%s @ %.6g lev=%dx id=%s",
                 ps_side, sym, qty, fill_price, lev, entry_id)

        q_tp1 = self._fmt_qty(qty * 0.40, info)
        q_tp2 = self._fmt_qty(qty * 0.35, info)
        q_tp3 = self._fmt_qty(qty * 0.25, info)
        tp1_p = self._fp(signal.tp1, info)
        tp2_p = self._fp(signal.tp2, info)
        tp3_p = self._fp(signal.tp3, info)
        sl_p  = self._fp(signal.sl,  info)

        def _tp(stop_p, q) -> str:
            if q < info.min_qty:
                return ""
            r = self._req("POST", "/fapi/v1/order", {
                "symbol": sym, "side": close_side, "positionSide": ps_side,
                "type": "TAKE_PROFIT_MARKET",
                "stopPrice": stop_p, "quantity": q,
                "timeInForce": "GTC", "workingType": "MARK_PRICE",
                "priceProtect": "TRUE", "reduceOnly": "true",
            })
            oid = str((r or {}).get("orderId", ""))
            log.info("Futures TP @ %s qty=%s id=%s", stop_p, q, oid)
            return oid

        tp1_id = _tp(tp1_p, q_tp1)
        tp2_id = _tp(tp2_p, q_tp2)
        tp3_id = _tp(tp3_p, q_tp3)

        # Stop-market (closes full remaining position)
        sl_r  = self._req("POST", "/fapi/v1/order", {
            "symbol": sym, "side": close_side, "positionSide": ps_side,
            "type": "STOP_MARKET",
            "stopPrice": sl_p, "closePosition": "true",
            "workingType": "MARK_PRICE", "priceProtect": "TRUE",
        })
        sl_id = str((sl_r or {}).get("orderId", ""))
        log.info("Futures SL @ %s id=%s", sl_p, sl_id)

        return TradeResult(True, sym, side, qty, fill_price, mode="futures",
                           leverage=lev,
                           position_usdt=round(pos_usdt * lev, 2),
                           risk_usdt=round(risk_usdt, 2),
                           entry_order_id=entry_id,
                           tp1_order_id=tp1_id, tp2_order_id=tp2_id,
                           tp3_order_id=tp3_id, sl_order_id=sl_id)

    # ── Safety ──────────────────────────────────────────────────

    def _pre_flight(self, balance: float) -> tuple[bool, str]:
        if not self._key or not self._secret:
            return False, "No API keys in .env (BINANCE_API_KEY / BINANCE_API_SECRET)"
        if balance <= 0:
            return False, (f"Balance is $0.00 — possible causes:\n"
                           f"  1. Futures wallet is empty (transfer USDT from spot wallet)\n"
                           f"  2. API key does not have Futures Trading permission\n"
                           f"  3. Using spot keys with futures mode — get futures testnet keys")
        if balance < 5:
            return False, f"Balance ${balance:.2f} is too low (minimum $5)"
        if self._daily_loss >= self._loss_limit:
            return False, f"Daily loss limit {self._loss_limit}% reached — paused until midnight"
        if self._daily_trades >= self._max_trades:
            return False, f"Daily trade limit ({self._max_trades}) reached"
        if not self._live:
            log.debug("TESTNET mode — orders sent to testnet, not live Binance")
        return True, ""

    # ── Symbol info ─────────────────────────────────────────────

    def _get_symbol_info(self, sym: str) -> SymbolInfo:
        if sym in self._sym_cache:
            return self._sym_cache[sym]
        info = SymbolInfo()
        try:
            endpoint = "/fapi/v1/exchangeInfo" if self._mode == "futures" else "/api/v3/exchangeInfo"
            ex = self._pub("GET", endpoint, {} if self._mode == "futures" else {"symbol": sym})
            for item in (ex or {}).get("symbols", []):
                if item.get("symbol") != sym:
                    continue
                filters = {f["filterType"]: f for f in item.get("filters", [])}
                pf   = filters.get("PRICE_FILTER", {})
                lf   = filters.get("LOT_SIZE",     {})
                nf   = filters.get("MIN_NOTIONAL", {}) or filters.get("NOTIONAL", {})
                tick = float(pf.get("tickSize",      "0.0001") or "0.0001")
                step = float(lf.get("stepSize",      "0.001")  or "0.001")
                mq   = float(lf.get("minQty",        "0.001")  or "0.001")
                mn   = float(nf.get("minNotional",   "5")      or nf.get("notional", "5") or "5")

                def _prec(val: float) -> int:   # FIX: renamed from 's' to avoid loop var shadow
                    txt = str(val)
                    return len(txt.rstrip("0").split(".")[1]) if "." in txt else 0

                info = SymbolInfo(_prec(tick), _prec(step), mq, mn, tick, step)
                break
        except Exception as e:
            log.debug("Symbol info %s: %s — using defaults", sym, e)
        self._sym_cache[sym] = info
        return info

    # ── HTTP ────────────────────────────────────────────────────

    def _req(self, method: str, path: str, params: dict) -> dict | None:
        p = {k: v for k, v in params.items() if v is not None}
        p["timestamp"]  = int(time.time() * 1000)
        p["recvWindow"] = 10000   # allow ±10s clock drift — fixes -1021 timestamp error
        query = urlencode(p)
        sig   = hmac.new(self._secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        url   = f"{self._base}{path}?{query}&signature={sig}"
        resp  = requests.request(method, url,
                                 headers={"X-MBX-APIKEY": self._key}, timeout=12)
        if resp.status_code not in (200, 201):
            log.error("Binance %s %s → %d: %s", method, path,
                      resp.status_code, resp.text[:400])
            return None
        return resp.json()

    def _pub(self, method: str, path: str, params: dict) -> dict | None:
        resp = requests.request(method, f"{self._base}{path}", params=params, timeout=12)
        return resp.json() if resp.ok else None

    def _fp(self, price: float, info: SymbolInfo) -> float:
        return round(price, info.price_precision)

    def _fmt_qty(self, qty: float, info: SymbolInfo) -> float:
        return round(qty, info.qty_precision)