"""
src/trading/binance_trader.py — v4
Full automated execution for both SPOT and FUTURES on Binance.

Fixes vs v3:
  - _place_tp() and SL orders now use _order() for correct algo routing (fix -4120)
  - _fmt_qty() floors to stepSize, not just round (fix -1111)
  - _fp() falls back to 2*tick if snapped price becomes zero
  - Added balance caching to reduce API calls (mitigate IP ban)
  - Daily counters reset automatically at UTC midnight
  - SL limit price clamped to mark price ±5% (fix -4024)
"""
import hmac
import hashlib
import time
import requests
from urllib.parse import urlencode
from dataclasses import dataclass
from typing import Literal
from datetime import datetime
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
        self._daily_loss   = 0.0
        self._daily_trades = 0
        self._sym_cache: dict[str, SymbolInfo] = {}

        # Balance cache (reduces API calls)
        self._balance_cache = {"value": 0.0, "timestamp": 0, "ttl": 2.0}  # 2 seconds
        self._last_reset_date = None

        self._base = (_FUT_BASE if live else _FUT_TEST) if mode == "futures" \
                     else (_SPOT_BASE if live else _SPOT_TEST)

        log.info("BinanceTrader: %s %s | risk=%.1f%% | loss_limit=%.1f%%",
                 mode.upper(), "LIVE" if live else "TESTNET",
                 risk_pct, daily_loss_limit_pct)

    # ── Daily reset ──────────────────────────────────────────────

    def _check_day_reset(self):
        """Reset daily counters if UTC date has changed."""
        now = datetime.utcnow()
        today = now.date()
        if self._last_reset_date is None:
            self._last_reset_date = today
        if today != self._last_reset_date:
            self.reset_daily_counters()
            self._last_reset_date = today

    def reset_daily_counters(self):
        self._daily_trades = 0
        self._daily_loss   = 0.0
        log.info("Daily counters reset")

    # ── Public ──────────────────────────────────────────────────

    def execute_signal(self, signal, balance_usdt: float) -> TradeResult:
        self._check_day_reset()
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
            if balance_usdt >= info.min_notional * 2:
                pos_usdt = info.min_notional
                log.info("Position bumped to minimum notional $%.0f for %s",
                         info.min_notional, sym)
            else:
                return TradeResult(False, sym, signal.signal, 0, signal.price,
                                   mode=self._mode,
                                   error=f"Balance ${balance_usdt:.2f} too low for minimum "
                                         f"${info.min_notional:.0f} position on {sym}")

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
        self._daily_loss += abs(loss_pct)
        log.info("Daily loss updated: %.1f%% / %.1f%% limit",
                 self._daily_loss, self._loss_limit)
        if self._daily_loss >= self._loss_limit:
            log.warning("Daily loss limit %.1f%% reached — auto-trade will pause",
                        self._loss_limit)

    def cancel_all_orders(self, symbol: str = None) -> dict:
        cancelled, errors = [], []
        try:
            if self._mode == "futures":
                get_path = "/fapi/v1/openOrders"
                del_path = "/fapi/v1/allOpenOrders"
            else:
                get_path = "/api/v3/openOrders"
                del_path = "/api/v3/openOrders"

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
        """Returns available USDT with caching to reduce API calls."""
        now = time.time()
        if now - self._balance_cache["timestamp"] < self._balance_cache["ttl"]:
            return self._balance_cache["value"]
        bal = self._fetch_balance()
        self._balance_cache = {"value": bal, "timestamp": now, "ttl": 2.0}
        return bal

    def _fetch_balance(self) -> float:
        """Actual balance fetch without cache."""
        info = self.get_balance()
        return float(info.get("available_balance", 0) or 0)

    def get_balance(self) -> dict:
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

    def _order(self, params: dict) -> dict:
        """Smart order routing: algo orders to /fapi/v1/algoOrder, others to /fapi/v1/order."""
        if self._mode != "futures":
            return self._req("POST", "/api/v3/order", params) or {}

        otype = params.get("type", "")
        algo_types = {"TAKE_PROFIT_MARKET", "STOP_MARKET", "TRAILING_STOP_MARKET",
                      "TAKE_PROFIT", "STOP"}

        if otype in algo_types:
            algo = {
                "algoType":   "CONDITIONAL",
                "symbol":     params["symbol"],
                "side":       params["side"],
                "type":       otype,
                "timeInForce": params.get("timeInForce", "GTC"),
            }
            stop_p = params.get("stopPrice") or params.get("triggerPrice")
            if stop_p:
                algo["triggerPrice"] = stop_p

            if params.get("price"):
                algo["price"] = params["price"]

            close_pos = str(params.get("closePosition", "false")).lower() == "true"
            if close_pos:
                algo["closePosition"] = "true"
            elif params.get("quantity"):
                algo["quantity"]   = params["quantity"]
                algo["reduceOnly"] = params.get("reduceOnly", "true")

            if params.get("workingType"):
                algo["workingType"] = params["workingType"]
            if params.get("priceProtect"):
                algo["priceProtect"] = params["priceProtect"]
            if params.get("callbackRate"):
                algo["callbackRate"] = params["callbackRate"]
            if params.get("activationPrice") or params.get("activatePrice"):
                algo["activatePrice"] = params.get("activationPrice") or params.get("activatePrice")

            r = self._req("POST", "/fapi/v1/algoOrder", algo) or {}
            if isinstance(r, dict) and "algoId" in r and "orderId" not in r:
                r["orderId"] = r["algoId"]
            return r

        return self._req("POST", "/fapi/v1/order", params) or {}

    def get_open_orders(self, symbol: str = None) -> list:
        try:
            params = {"symbol": symbol} if symbol else {}
            path   = "/fapi/v1/openOrders" if self._mode == "futures" else "/api/v3/openOrders"
            result = self._req("GET", path, params)
            orders = result if isinstance(result, list) else []

            if self._mode == "futures":
                algo = self._req("GET", "/fapi/v1/openAlgoOrders", params)
                if isinstance(algo, dict):
                    algo_list = algo.get("algoOrders", algo.get("orders", []))
                elif isinstance(algo, list):
                    algo_list = algo
                else:
                    algo_list = []
                for o in algo_list:
                    if "algoId" in o and "orderId" not in o:
                        o["orderId"] = o["algoId"]
                    if "triggerPrice" in o and "stopPrice" not in o:
                        o["stopPrice"] = o["triggerPrice"]
                    if "orderType" in o and "type" not in o:
                        o["type"] = o["orderType"]
                orders = orders + algo_list
            return orders
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

    # ── Spot ────────────────────────────────────────────────────

    def _spot(self, sym, signal, qty, info, risk_usdt, pos_usdt) -> TradeResult:
        if signal.signal != "BUY":
            return TradeResult(False, sym, signal.signal, qty, signal.price, mode="spot",
                               error="SELL signals not supported on spot. Enable futures mode.")

        if signal.tp1 <= signal.price:
            return TradeResult(False, sym, signal.signal, qty, signal.price, mode="spot",
                               error=f"TP1 ({signal.tp1}) must be above entry ({signal.price})")
        if signal.sl >= signal.price:
            return TradeResult(False, sym, signal.signal, qty, signal.price, mode="spot",
                               error=f"SL ({signal.sl}) must be below entry ({signal.price})")

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

        oco_id = tp1_id = sl_id = ""
        if q_oco >= info.min_qty:
            oco = self._req("POST", "/api/v3/order/oco", {
                "symbol": sym, "side": "SELL", "quantity": q_oco,
                "price": tp1_p,
                "stopPrice": sl_p,
                "stopLimitPrice": sl_lim,
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
        close_side = "SELL" if is_long else "BUY"

        if is_long and signal.tp1 <= signal.price:
            return TradeResult(False, sym, signal.signal, qty, signal.price, mode="futures",
                               error=f"TP1 {signal.tp1:.6g} must be above entry {signal.price:.6g}")
        if not is_long and signal.tp1 >= signal.price:
            return TradeResult(False, sym, signal.signal, qty, signal.price, mode="futures",
                               error=f"TP1 {signal.tp1:.6g} must be below entry {signal.price:.6g}")

        lev = 3 if getattr(signal, "btc_score", 50) >= 62 else 2
        self._req("POST", "/fapi/v1/leverage", {"symbol": sym, "leverage": lev})

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

        q_tp1 = self._fmt_qty(qty * 0.40, info)
        q_tp2 = self._fmt_qty(qty * 0.35, info)
        q_tp3 = self._fmt_qty(qty * 0.15, info)
        q_tsl = self._fmt_qty(qty * 0.10, info)
        tp1_p = self._fp(signal.tp1, info)
        tp2_p = self._fp(signal.tp2, info)
        tp3_p = self._fp(signal.tp3, info)
        sl_p  = self._fp(signal.sl,  info)

        # ---- TP placement ----
        def _place_tp(label, stop_p, q) -> str:
            if q < info.min_qty or stop_p <= 0:
                return ""
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
                r = self._order(params)  # FIX: use _order for proper routing
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
        tp3_id = _place_tp("TP3", tp3_p, q_tp3)

        # ---- Hard Stop-Loss ----
        sl_id = ""
        if sl_p <= 0:
            log.error("Futures SL skipped — price is 0 for %s", sym)
        else:
            sl_qty_val = self._fmt_qty(qty, info)
            sl_lim_p   = self._fp(sl_p * (1.005 if not is_long else 0.995), info)

            # Fetch mark price to clamp limit price if fallback is needed
            mark_price = 0
            try:
                mp_resp = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                                       params={"symbol": sym}, timeout=3)
                if mp_resp.ok:
                    mark_price = float(mp_resp.json().get("markPrice", 0))
            except Exception:
                pass

            for sl_type, sl_params in [
                ("STOP_qty", {
                    "symbol":      sym,
                    "side":        close_side,
                    "type":        "STOP",
                    "stopPrice":   sl_p,
                    "price":       sl_lim_p,
                    "quantity":    sl_qty_val,
                    "timeInForce": "GTC",
                    "reduceOnly":  "true",
                }),
                ("LIMIT_sl", {
                    "symbol":      sym,
                    "side":        close_side,
                    "type":        "LIMIT",
                    "price":       sl_p,
                    "quantity":    sl_qty_val,
                    "timeInForce": "GTC",
                    "reduceOnly":  "true",
                }),
            ]:
                # Clamp LIMIT price to mark price ±5% to avoid -4024
                if sl_type == "LIMIT_sl" and mark_price > 0:
                    if is_long:
                        min_limit = mark_price * 0.95
                        max_limit = mark_price * 1.05
                    else:
                        min_limit = mark_price * 0.95
                        max_limit = mark_price * 1.05
                    clamped = max(min_limit, min(sl_params["price"], max_limit))
                    sl_params["price"] = clamped

                sl_r = self._order(sl_params)  # FIX: use _order
                if isinstance(sl_r, dict) and "orderId" in sl_r:
                    sl_id = str(sl_r["orderId"])
                    log.info("Futures SL (%s) @ %.6g id=%s ✅", sl_type, sl_p, sl_id)
                    break
                code = sl_r.get("code", 0) if isinstance(sl_r, dict) else 0
                if code in (-4120, -1104, -4045):
                    log.warning("Futures SL: %s rejected (code %d) → next", sl_type, code)
                    continue
                log.error("Futures SL FAILED (%s) @ %.6g: code=%d ⚠️", sl_type, sl_p, code)
                try:
                    from dashboard.views import push_trade_alert
                    push_trade_alert("error", f"⚠️ NO SL on {sym} code={code}")
                except Exception:
                    pass
                break

        # ---- Trailing Stop ----
        trailing_id = ""
        if q_tsl >= info.min_qty and tp1_p > 0:
            cur_mark = 0
            try:
                mp_resp = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                                       params={"symbol": sym}, timeout=3)
                if mp_resp.ok:
                    cur_mark = float(mp_resp.json().get("markPrice", 0))
            except Exception:
                pass

            if cur_mark > 0:
                if is_long:
                    act_p = tp1_p if tp1_p >= cur_mark else self._fp(cur_mark * 1.001, info)
                else:
                    act_p = tp1_p if tp1_p <= cur_mark else self._fp(cur_mark * 0.999, info)
            else:
                act_p = tp1_p

            trail_r = self._order({
                "symbol": sym, "side": close_side,
                "type": "TRAILING_STOP_MARKET",
                "quantity": q_tsl, "callbackRate": 1.5,
                "activationPrice": act_p,
                "workingType": "MARK_PRICE", "reduceOnly": "true",
            })
            if isinstance(trail_r, dict) and "orderId" in trail_r:
                trailing_id = str(trail_r["orderId"])
                log.info("Futures TSL activation=%.6g callback=1.5%% id=%s ✅", act_p, trailing_id)

        log.info("Futures orders placed: entry=%s tp1=%s tp2=%s sl=%s tsl=%s",
                 entry_id, tp1_id or "FAIL", tp2_id or "FAIL",
                 sl_id or "FAIL⚠️", trailing_id or "skip")

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
                info.tick_size = tick
                step = float(lf.get("stepSize",      "0.001")  or "0.001")
                mq   = float(lf.get("minQty",        "0.001")  or "0.001")
                mn_raw = float(nf.get("minNotional", "5") or nf.get("notional","5") or "5")
                mn   = max(mn_raw, 20.0) if self._mode == "futures" else mn_raw

                def _prec(val: float) -> int:
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
        p["recvWindow"] = 10000
        query = urlencode(p)
        sig   = hmac.new(self._secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        url   = f"{self._base}{path}?{query}&signature={sig}"
        resp  = requests.request(method, url,
                                 headers={"X-MBX-APIKEY": self._key}, timeout=12)
        if resp.status_code not in (200, 201):
            log.error("Binance %s %s → %d: %s", method, path,
                      resp.status_code, resp.text[:400])
            try:
                return resp.json()
            except Exception:
                return {"code": resp.status_code, "msg": resp.text[:200]}
        return resp.json()

    def _pub(self, method: str, path: str, params: dict) -> dict | None:
        resp = requests.request(method, f"{self._base}{path}", params=params, timeout=12)
        return resp.json() if resp.ok else None

    def _fp(self, price: float, info: SymbolInfo) -> float:
        """Snap price to nearest valid tick size; never return 0 if price > 0."""
        if price <= 0:
            return 0.0
        tick = info.tick_size
        if tick and tick > 0:
            snapped = round(price / tick) * tick
            if snapped <= 0 and price > 0:
                snapped = tick * 2   # fallback to 2 ticks above zero
            if tick >= 1:
                result = round(snapped, 0)
            else:
                tick_str = f"{tick:.10f}".rstrip("0")
                dec = len(tick_str.split(".")[1]) if "." in tick_str else 0
                result = round(snapped, dec)
            if result == 0 and price > 0:
                result = round(price, 4)
            return result

        import math
        p = info.price_precision
        result = round(price, p)
        if result == 0 and price > 0:
            p = max(p, -int(math.floor(math.log10(abs(price)))) + 4)
            result = round(price, p)
        return result

    def _fmt_qty(self, qty: float, info: SymbolInfo) -> float:
        """Floor quantity to nearest stepSize (not just round)."""
        step = info.step_size
        if step <= 0:
            return round(qty, info.qty_precision)
        floored = (qty // step) * step
        dec = info.qty_precision
        return round(floored, dec)

    # ── Protect Open Positions ──────────────────────────────────

    def protect_open_positions(self, signal_lookup: dict) -> list:
        """Scan all open futures positions; place missing TP/SL/TSL."""
        if self._mode != "futures":
            return []
        self._check_day_reset()
        protected = []
        try:
            positions = self.get_positions()
            open_orders = self.get_open_orders() or []
            syms_with_orders = set(o.get("symbol","") for o in open_orders)

            sym_orders = {}
            for o in open_orders:
                s = o.get("symbol","")
                sym_orders.setdefault(s, []).append(o)

            SL_TYPES  = {"STOP_MARKET","STOP","STOP_LOSS","STOP_LOSS_LIMIT"}
            TSL_TYPES = {"TRAILING_STOP_MARKET"}
            TP_TYPES  = {"TAKE_PROFIT_MARKET","TAKE_PROFIT","LIMIT"}

            total_orders = len(open_orders)
            log.info("protect_open_positions: %d positions, %d open orders",
                     len(positions), total_orders)

            max_ok = max(len(positions) * 5, 10)
            if total_orders > max_ok and positions:
                log.warning("Duplicate orders (%d > %d) — cancelling all for clean reset",
                            total_orders, max_ok)
                for _p in positions:
                    _s = _p.get("symbol","")
                    if _s:
                        self._req("DELETE", "/fapi/v1/allOpenOrders", {"symbol": _s})
                        log.info("  Cancelled all orders for %s", _s)
                open_orders = self.get_open_orders() or []
                sym_orders  = {}
                for o in open_orders:
                    sym_orders.setdefault(o.get("symbol",""), []).append(o)
                log.info("  After cleanup: %d orders", len(open_orders))

            for pos in positions:
                sym = pos.get("symbol","")
                amt = float(pos.get("positionAmt", 0))
                if amt == 0:
                    continue

                orders_for_sym = sym_orders.get(sym, [])
                has_sl  = any(o.get("type") in SL_TYPES  for o in orders_for_sym)
                has_tsl = any(o.get("type") in TSL_TYPES for o in orders_for_sym)
                has_tp  = any(o.get("type") in TP_TYPES  for o in orders_for_sym)

                log.info("  %s: amt=%.4f orders=%d has_tp=%s has_sl=%s has_tsl=%s",
                         sym, amt, len(orders_for_sym),
                         "✅" if has_tp else "❌",
                         "✅" if has_sl else "❌",
                         "✅" if has_tsl else "❌")

                if has_tp and has_sl:
                    if not has_tsl:
                        log.info("  %s: TP+SL ok — adding missing TSL", sym)
                    else:
                        log.info("  %s: fully protected ✅", sym)
                        continue

                is_long    = amt > 0
                close_side = "SELL" if is_long else "BUY"
                qty        = abs(amt)
                entry      = float(pos.get("entryPrice", 0))

                info = self._get_symbol_info(sym)
                sig  = signal_lookup.get(sym)

                if sig and sig.tp1 > 0 and sig.sl > 0:
                    tp1_p = self._fp(sig.tp1, info)
                    tp2_p = self._fp(sig.tp2, info)
                    tp3_p = self._fp(getattr(sig, "tp3", 0) or tp2_p * 1.02, info)
                    sl_p  = self._fp(sig.sl,  info)
                    log.info("Using signal levels for %s: TP1=%.6g TP2=%.6g TP3=%.6g SL=%.6g",
                             sym, tp1_p, tp2_p, tp3_p, sl_p)
                else:
                    tp1_pct = 0.03
                    tp2_pct = 0.06
                    tp3_pct = 0.10
                    sl_pct  = 0.04
                    if is_long:
                        tp1_p = self._fp(entry * (1 + tp1_pct), info)
                        tp2_p = self._fp(entry * (1 + tp2_pct), info)
                        tp3_p = self._fp(entry * (1 + tp3_pct), info)
                        sl_p  = self._fp(entry * (1 - sl_pct),  info)
                    else:
                        tp1_p = self._fp(entry * (1 - tp1_pct), info)
                        tp2_p = self._fp(entry * (1 - tp2_pct), info)
                        tp3_p = self._fp(entry * (1 - tp3_pct), info)
                        sl_p  = self._fp(entry * (1 + sl_pct),  info)
                    log.warning("Default TP/SL for %s from entry=%.6g: TP1=%.6g TP2=%.6g TP3=%.6g SL=%.6g",
                                sym, entry, tp1_p, tp2_p, tp3_p, sl_p)

                if sl_p <= 0:
                    log.error("Skipping %s — cannot calculate SL price", sym)
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
                            "workingType": "MARK_PRICE",
                            "priceProtect": "true", "reduceOnly": "true",
                        }),
                        ("LIMIT", {
                            "symbol": sym, "side": close_side, "type": "LIMIT",
                            "price": stop_p, "quantity": qty_p,
                            "timeInForce": "GTC", "reduceOnly": "true",
                        }),
                    ]:
                        r = self._order(params)
                        if isinstance(r, dict) and "orderId" in r:
                            log.info("  %s (%s) @ %.6g id=%s ✅", label, name, stop_p, r["orderId"])
                            return
                        code = r.get("code", 0) if isinstance(r, dict) else 0
                        if code in (-4120, -1104, -2015) and name != "LIMIT":
                            log.warning("  %s: algo rejected (code %d) → LIMIT", label, code)
                            continue
                        log.error("  %s FAILED (%s): code=%d %s",
                                  label, name, code, r.get("msg",""))
                        return

                if not has_tp:
                    _prot_tp("TP1", tp1_p, q_tp1)
                    _prot_tp("TP2", tp2_p, q_tp2)
                    _prot_tp("TP3", tp3_p, q_tsl)

                if not has_sl and sl_p > 0:
                    _sl_qty  = self._fmt_qty(qty, info)
                    _sl_limp = self._fp(sl_p * (1.005 if not is_long else 0.995), info)

                    # Fetch mark price to clamp limit fallback
                    mark_price = 0
                    try:
                        mp_resp = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                                               params={"symbol": sym}, timeout=3)
                        if mp_resp.ok:
                            mark_price = float(mp_resp.json().get("markPrice", 0))
                    except Exception:
                        pass

                    for sl_t, sl_p2 in [
                        ("STOP_MARKET", {"stopPrice": sl_p, "quantity": _sl_qty,
                                         "workingType": "MARK_PRICE", "priceProtect": "true",
                                         "reduceOnly": "true"}),
                        ("STOP", {"stopPrice": sl_p, "price": _sl_limp,
                                  "quantity": _sl_qty,
                                  "timeInForce": "GTC", "reduceOnly": "true"}),
                    ]:
                        # Clamp limit price for STOP (non-market) to avoid -4024
                        if sl_t == "STOP" and mark_price > 0:
                            if is_long:
                                min_limit = mark_price * 0.95
                                max_limit = mark_price * 1.05
                            else:
                                min_limit = mark_price * 0.95
                                max_limit = mark_price * 1.05
                            clamped = max(min_limit, min(sl_p2.get("price", sl_p), max_limit))
                            sl_p2["price"] = clamped

                        sl_r = self._order({"symbol": sym, "side": close_side, "type": sl_t, **sl_p2})
                        if isinstance(sl_r, dict) and "orderId" in sl_r:
                            log.info("  SL (%s) @ %.6g id=%s ✅", sl_t, sl_p, sl_r["orderId"])
                            break
                        code = sl_r.get("code", 0) if isinstance(sl_r, dict) else 0
                        if code in (-4120, -1104) and sl_t != "STOP":
                            log.warning("  SL: %s rejected (code %d) — trying STOP", sl_t, code)
                            continue
                        log.error("  SL FAILED (%s): code=%d  ⚠️ STILL UNPROTECTED!", sl_t, code)
                        try:
                            from dashboard.views import push_trade_alert
                            push_trade_alert("error", f"⚠️ {sym} still unprotected — SL failed (code {code})")
                        except Exception:
                            pass
                        break
                else:
                    log.error("  SL skipped — price rounds to 0 for %s", sym)

                if not has_tsl and q_tsl >= info.min_qty and tp1_p > 0:
                    cur_mark = 0
                    try:
                        mp_resp = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                                               params={"symbol": sym}, timeout=3)
                        if mp_resp.ok:
                            cur_mark = float(mp_resp.json().get("markPrice", 0))
                    except Exception:
                        pass

                    if cur_mark > 0:
                        if is_long:
                            act_p = tp1_p if tp1_p >= cur_mark else self._fp(cur_mark * 1.001, info)
                        else:
                            act_p = tp1_p if tp1_p <= cur_mark else self._fp(cur_mark * 0.999, info)
                    else:
                        act_p = tp1_p

                    r = self._order({
                        "symbol": sym, "side": close_side,
                        "type": "TRAILING_STOP_MARKET",
                        "quantity": q_tsl, "callbackRate": 1.5,
                        "activationPrice": act_p,
                        "workingType": "MARK_PRICE", "reduceOnly": "true",
                    })
                    if isinstance(r, dict) and "orderId" in r:
                        log.info("  TSL set activation=%.6g callback=1.5%% id=%s ✅", act_p, r["orderId"])

                protected.append(sym)

        except Exception as e:
            log.error("protect_open_positions error: %s", e, exc_info=True)

        return protected