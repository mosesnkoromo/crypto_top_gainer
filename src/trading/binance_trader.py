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

    def get_available_balance(self) -> float:
        """Returns available USDT as a plain float — use this for trading decisions."""
        info = self.get_balance()
        return float(info.get("available_balance", 0) or 0)

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

    # ── Futures (ONE-WAY MODE) ──────────────────────────────────
    # Always uses one-way mode (no positionSide parameter).
    # BUY = open long, SELL = open short.
    # If same symbol has open position in opposite direction,
    # the new order will reduce/close it first (Binance default).

    def _futures(self, sym, signal, qty, info, risk_usdt, pos_usdt) -> TradeResult:
        is_long    = signal.signal == "BUY"
        side       = "BUY"  if is_long else "SELL"
        close_side = "SELL" if is_long else "BUY"

        # Validate signal levels
        if is_long and signal.tp1 <= signal.price:
            return TradeResult(False, sym, signal.signal, qty, signal.price, mode="futures",
                               error=f"TP1 {signal.tp1:.6g} must be above entry {signal.price:.6g}")
        if not is_long and signal.tp1 >= signal.price:
            return TradeResult(False, sym, signal.signal, qty, signal.price, mode="futures",
                               error=f"TP1 {signal.tp1:.6g} must be below entry {signal.price:.6g}")

        # Set leverage (conservative: 2x bear, 3x bull)
        lev = 3 if getattr(signal, "btc_score", 50) >= 62 else 2
        self._req("POST", "/fapi/v1/leverage", {"symbol": sym, "leverage": lev})

        # NEVER attempt to switch position mode — use account's current mode as-is
        # One-way mode: no positionSide field in any order

        # Market entry — ONE-WAY MODE (no positionSide)
        entry = self._req("POST", "/fapi/v1/order", {
            "symbol":   sym,
            "side":     side,
            "type":     "MARKET",
            "quantity": qty,
        })
        if not entry or "orderId" not in entry:
            raise Exception(f"Futures {side} rejected: {entry}")
        entry_id   = str(entry["orderId"])
        fill_price = float(entry.get("avgPrice", signal.price)) or signal.price
        log.info("Futures %s %s: qty=%s @ %.6g lev=%dx id=%s",
                 "LONG" if is_long else "SHORT", sym, qty, fill_price, lev, entry_id)

        # Position split:
        #   TP1 → 40% at TP1 level (quick profit lock)
        #   TP2 → 35% at TP2 level (main target)
        #   SL  → closePosition=true (closes ALL remaining when hit)
        #   TSL → 25% trailing stop, activates at TP1 (protects runner)
        # Total explicit qty = 75% (TP1+TP2). SL closes whatever is left.
        # This avoids -2010 "would reduce too much" error.
        q_tp1 = self._fmt_qty(qty * 0.40, info)
        q_tp2 = self._fmt_qty(qty * 0.35, info)
        q_tsl = self._fmt_qty(qty * 0.25, info)
        tp1_p = self._fp(signal.tp1, info)
        tp2_p = self._fp(signal.tp2, info)
        tp3_p = self._fp(signal.tp3, info)
        sl_p  = self._fp(signal.sl,  info)

        def _place_tp(label, stop_p, q) -> str:
            if q < info.min_qty:
                return ""
            if stop_p <= 0:
                log.error("TP %s skipped — price is 0", label)
                return ""
            # Attempt order types in order: algo → limit
            attempts = [
                ("TAKE_PROFIT_MARKET", {
                    "symbol": sym, "side": close_side, "type": "TAKE_PROFIT_MARKET",
                    "stopPrice": stop_p, "quantity": q, "timeInForce": "GTC",
                    "workingType": "MARK_PRICE", "priceProtect": "true", "reduceOnly": "true",
                }),
                ("LIMIT", {
                    "symbol": sym, "side": close_side, "type": "LIMIT",
                    "price": stop_p, "quantity": q,
                    "timeInForce": "GTC", "reduceOnly": "true",
                }),
            ]
            for name, params in attempts:
                r = self._req("POST", "/fapi/v1/order", params)
                if isinstance(r, dict) and "orderId" in r:
                    log.info("Futures %s (%s) @ %.6g qty=%s id=%s ✅", label, name, stop_p, q, r["orderId"])
                    return str(r["orderId"])
                code = r.get("code", 0) if isinstance(r, dict) else 0
                if code in (-4120, -1104) and name != "LIMIT":
                    log.warning("Futures %s: %s rejected (code %d) — trying LIMIT", label, name, code)
                    continue
                log.error("Futures %s FAILED (%s) @ %.6g: code=%d", label, name, stop_p, code)
                return ""
            return ""

        tp1_id = _place_tp("TP1", tp1_p, q_tp1)
        tp2_id = _place_tp("TP2", tp2_p, q_tp2)

        # ── Hard Stop-Loss: closePosition=true, NO quantity, NO reduceOnly ──
        # closePosition closes 100% of remaining position (no qty needed, avoids conflicts)
        sl_id = ""
        if sl_p <= 0:
            log.error("Futures SL skipped — price rounds to 0 (tickSize issue for %s)", sym)
        else:
            # Try STOP_MARKET first, fall back to STOP (limit) if rejected
            sl_limit_p = self._fp(sl_p * 1.005 if not is_long else sl_p * 0.995, info)
            for sl_type, sl_params in [
                ("STOP_MARKET", {
                    "symbol": sym, "side": close_side, "type": "STOP_MARKET",
                    "stopPrice": sl_p, "closePosition": "true",
                    "workingType": "MARK_PRICE", "priceProtect": "true",
                }),
                ("STOP", {
                    "symbol": sym, "side": close_side, "type": "STOP",
                    "stopPrice": sl_p, "price": sl_limit_p,
                    "quantity": self._fmt_qty(qty, info),
                    "timeInForce": "GTC", "reduceOnly": "true",
                }),
            ]:
                sl_r = self._req("POST", "/fapi/v1/order", sl_params)
                if isinstance(sl_r, dict) and "orderId" in sl_r:
                    sl_id = str(sl_r["orderId"])
                    log.info("Futures SL (%s) @ %.6g id=%s ✅", sl_type, sl_p, sl_id)
                    break
                code = sl_r.get("code", 0) if isinstance(sl_r, dict) else 0
                if code in (-4120, -1104) and sl_type != "STOP":
                    log.warning("Futures SL: %s rejected (code %d) — trying STOP", sl_type, code)
                    continue
                log.error("Futures SL FAILED (%s) @ %.6g: code=%d  ⚠️ NO STOP LOSS!", sl_type, sl_p, code)
                break

        # ── Trailing Stop-Loss: 25% of position, activates at TP1 ──
        # Protects the runner portion. No qty conflict since SL uses closePosition.
        trailing_id = ""
        if q_tsl >= info.min_qty:
            trail_r = self._req("POST", "/fapi/v1/order", {
                "symbol":          sym,
                "side":            close_side,
                "type":            "TRAILING_STOP_MARKET",
                "quantity":        q_tsl,
                "callbackRate":    1.5,
                "activationPrice": tp1_p,
                "workingType":     "MARK_PRICE",
                "reduceOnly":      "true",
            })
            if isinstance(trail_r, dict) and "orderId" in trail_r:
                trailing_id = str(trail_r["orderId"])
                log.info("Futures TSL activation=%.6g callback=1.5%% id=%s ✅", tp1_p, trailing_id)
            else:
                log.warning("Futures TSL failed (non-critical): %s", trail_r)

        tp3_id = ""  # TP3 not placed separately — TSL covers runner portion

        log.info(
            "Futures orders placed: entry=%s tp1=%s tp2=%s sl=%s tsl=%s",
            entry_id, tp1_id or "FAIL", tp2_id or "FAIL",
            sl_id or "FAIL⚠️", trailing_id or "skip"
        )

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
            # Return the error body so callers can inspect error code
            try:
                return resp.json()   # {"code": -4120, "msg": "..."}
            except Exception:
                return {"code": resp.status_code, "msg": resp.text[:200]}
        return resp.json()

    def _pub(self, method: str, path: str, params: dict) -> dict | None:
        resp = requests.request(method, f"{self._base}{path}", params=params, timeout=12)
        return resp.json() if resp.ok else None

    def _fp(self, price: float, info: SymbolInfo) -> float:
        if price <= 0:
            return 0.0
        # Use symbol precision, but auto-extend if it rounds to 0
        p = info.price_precision
        result = round(price, p)
        if result == 0 and price > 0:
            # tickSize is too coarse — calculate needed precision from actual price
            import math
            p = max(p, -int(math.floor(math.log10(abs(price)))) + 4)
            result = round(price, p)
        return result

    def _fmt_qty(self, qty: float, info: SymbolInfo) -> float:
        return round(qty, info.qty_precision)


    # ══════════════════════════════════════════════════════════
    # PROTECT UNPROTECTED POSITIONS
    # Call this every cycle to set TP/SL on positions that have none
    # ══════════════════════════════════════════════════════════

    def protect_open_positions(self, signal_lookup: dict) -> list:
        """
        Scan all open futures positions. For any that have no open orders
        (no TP/SL), place TP1 + TP2 + SL + TSL immediately.

        signal_lookup: dict mapping symbol → signal-like object with
                       tp1, tp2, tp3, sl fields. Can come from DB records.
        Returns list of symbols that were protected.
        """
        if self._mode != "futures":
            return []

        protected = []
        try:
            positions = self.get_positions()
            open_orders = self.get_open_orders() or []
            syms_with_orders = set(o.get("symbol","") for o in open_orders)

            log.info("protect_open_positions: %d positions, %d open orders, syms_with_orders=%s",
                     len(positions), len(open_orders), sorted(syms_with_orders))

            for pos in positions:
                sym = pos.get("symbol","")
                amt = float(pos.get("positionAmt", 0))
                if amt == 0:
                    continue

                # Count orders for THIS symbol specifically
                sym_order_count = sum(1 for o in open_orders if o.get("symbol") == sym)
                log.info("  Position %s: amt=%.4f orders_for_sym=%d", sym, amt, sym_order_count)

                # Already has orders — skip
                if sym in syms_with_orders:
                    log.info("  %s already has %d orders — skipping", sym, sym_order_count)
                    continue

                # No orders — needs protection
                is_long    = amt > 0
                close_side = "SELL" if is_long else "BUY"
                qty        = abs(amt)
                entry      = float(pos.get("entryPrice", 0))

                info = self._get_symbol_info(sym)
                sig  = signal_lookup.get(sym)

                if sig and sig.tp1 > 0 and sig.sl > 0:
                    # Use stored signal levels
                    tp1_p = self._fp(sig.tp1, info)
                    tp2_p = self._fp(sig.tp2, info)
                    sl_p  = self._fp(sig.sl,  info)
                    log.info("Using stored signal levels for %s: TP1=%.6g TP2=%.6g SL=%.6g",
                             sym, tp1_p, tp2_p, sl_p)
                else:
                    # No signal data — derive from entry price with default percentages
                    # Long: TP above entry, SL below.  Short: TP below entry, SL above.
                    tp1_pct = 0.03   # 3%
                    tp2_pct = 0.06   # 6%
                    sl_pct  = 0.04   # 4%
                    if is_long:
                        tp1_p = self._fp(entry * (1 + tp1_pct), info)
                        tp2_p = self._fp(entry * (1 + tp2_pct), info)
                        sl_p  = self._fp(entry * (1 - sl_pct),  info)
                    else:
                        tp1_p = self._fp(entry * (1 - tp1_pct), info)
                        tp2_p = self._fp(entry * (1 - tp2_pct), info)
                        sl_p  = self._fp(entry * (1 + sl_pct),  info)
                    log.warning("No signal data for %s — using default TP/SL from entry %.6g: "
                                "TP1=%.6g TP2=%.6g SL=%.6g", sym, entry, tp1_p, tp2_p, sl_p)

                if tp1_p <= 0 or sl_p <= 0:
                    log.error("Skipping %s — could not calculate valid TP/SL prices", sym)
                    continue

                q_tp1 = self._fmt_qty(qty * 0.40, info)
                q_tp2 = self._fmt_qty(qty * 0.35, info)
                q_tsl = self._fmt_qty(qty * 0.25, info)

                log.info("Protecting unprotected position: %s %s qty=%.4f entry=%.6g → TP1=%.6g TP2=%.6g SL=%.6g",
                         "LONG" if is_long else "SHORT", sym, qty, entry, tp1_p, tp2_p, sl_p)

                def _prot_tp(label, stop_p, qty_p):
                    if stop_p <= 0 or qty_p < info.min_qty:
                        return
                    for name, params in [
                        ("TAKE_PROFIT_MARKET", {
                            "symbol": sym, "side": close_side,
                            "type": "TAKE_PROFIT_MARKET",
                            "stopPrice": stop_p, "quantity": qty_p,
                            "timeInForce": "GTC", "workingType": "MARK_PRICE",
                            "priceProtect": "true", "reduceOnly": "true",
                        }),
                        ("LIMIT", {
                            "symbol": sym, "side": close_side, "type": "LIMIT",
                            "price": stop_p, "quantity": qty_p,
                            "timeInForce": "GTC", "reduceOnly": "true",
                        }),
                    ]:
                        r = self._req("POST", "/fapi/v1/order", params)
                        if isinstance(r, dict) and "orderId" in r:
                            log.info("  %s (%s) @ %.6g id=%s ✅", label, name, stop_p, r["orderId"])
                            return
                        code = r.get("code", 0) if isinstance(r, dict) else 0
                        if code in (-4120, -1104) and name != "LIMIT":
                            log.warning("  %s: %s rejected (code %d) — trying LIMIT", label, name, code)
                            continue
                        log.error("  %s FAILED (%s): code=%d", label, name, code)
                        return

                _prot_tp("TP1", tp1_p, q_tp1)
                _prot_tp("TP2", tp2_p, q_tp2)

                # SL with closePosition
                if sl_p > 0:
                    for sl_t, sl_p2 in [
                        ("STOP_MARKET", {"stopPrice": sl_p, "closePosition": "true",
                                         "workingType": "MARK_PRICE", "priceProtect": "true"}),
                        ("STOP", {"stopPrice": sl_p,
                                  "price": self._fp(sl_p * (1.005 if not is_long else 0.995), info),
                                  "quantity": self._fmt_qty(qty, info),
                                  "timeInForce": "GTC", "reduceOnly": "true"}),
                    ]:
                        sl_r = self._req("POST", "/fapi/v1/order", {
                            "symbol": sym, "side": close_side, "type": sl_t, **sl_p2})
                        if isinstance(sl_r, dict) and "orderId" in sl_r:
                            log.info("  SL (%s) @ %.6g id=%s ✅", sl_t, sl_p, sl_r["orderId"])
                            break
                        code = sl_r.get("code", 0) if isinstance(sl_r, dict) else 0
                        if code in (-4120, -1104) and sl_t != "STOP":
                            log.warning("  SL: %s rejected (code %d) — trying STOP", sl_t, code)
                            continue
                        log.error("  SL FAILED (%s): code=%d  ⚠️ STILL UNPROTECTED!", sl_t, code)
                        break
                else:
                    log.error("  SL skipped — price rounds to 0 for %s", sym)

                # Trailing SL (activates at TP1)
                if q_tsl >= info.min_qty and tp1_p > 0:
                    r = self._req("POST", "/fapi/v1/order", {
                        "symbol": sym, "side": close_side,
                        "type": "TRAILING_STOP_MARKET",
                        "quantity": q_tsl, "callbackRate": 1.5,
                        "activationPrice": tp1_p,
                        "workingType": "MARK_PRICE", "reduceOnly": "true",
                    })
                    if isinstance(r, dict) and "orderId" in r:
                        log.info("  TSL set activation=%.6g callback=1.5%% id=%s ✅", tp1_p, r["orderId"])

                protected.append(sym)

        except Exception as e:
            log.error("protect_open_positions error: %s", e, exc_info=True)

        return protected