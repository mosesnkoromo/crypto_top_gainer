"""
src/trading/binance_trader.py — v9

Fixes vs v8:
  - protect_spot_positions() OCO false positive: _req() returns the error JSON dict
    {"code": -1013} on HTTP 400 — which is truthy — so `if oco:` passed even when
    the order was rejected, logging "placed ✅" incorrectly. Fix: check for
    `oco.get("orderListId")` to confirm actual placement.

  - protect_open_positions() overfull positions: fully-protected positions
    (has_tp✅ has_sl✅ has_tsl✅) were skipped with `continue` BEFORE the per-symbol
    order-count check ran. So ALGOUSDT=15 orders, TAOUSDT=10, SANDUSDT=10 etc.
    accumulated indefinitely, pushing the account total to 200 open orders and
    triggering Binance -4045 "Reach max stop order limit" on new trades.
    Fix: if fully protected BUT order count > _PER_SYM_MAX_ORDERS, cancel and
    rebuild that symbol before skipping. Otherwise skip as before.
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

_FUTURES_1000X_SYMBOLS = {
    "PEPEUSDT":   ("1000PEPEUSDT",   1000),
    "SHIBUSDT":   ("1000SHIBUSDT",   1000),
    "FLOKIUSDT":  ("1000FLOKIUSDT",  1000),
    "BONKUSDT":   ("1000BONKUSDT",   1000),
    "XECUSDT":    ("1000XECUSDT",    1000),
    "RATSUSDT":   ("1000RATSUSDT",   1000),
    "LUNCUSDT":   ("1000LUNCUSDT",   1000),
    "BTTCUSDT":   ("1000BTTCUSDT",   1000),
    "CHEEMSUSDT": ("1000CHEEMSUSDT", 1000),
    "CATUSDT":    ("1000CATUSDT",    1000),
    "WHYUSDT":    ("1000WHYUSDT",    1000),
}

_PER_SYM_MAX_ORDERS = 7   # TP1+TP2+TP3+SL+TSL=5 ideal, 2 slack


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
                 max_trades_per_day: int = 10,
                 stop_market_only: bool = False):
        self._key              = api_key
        self._secret           = api_secret
        self._mode             = mode
        self._live             = live
        self._risk             = risk_pct
        self._loss_limit       = daily_loss_limit_pct
        self._max_trades       = max_trades_per_day
        self._stop_market_only = stop_market_only
        self._daily_loss       = 0.0
        self._daily_trades     = 0
        self._sym_cache: dict[str, SymbolInfo] = {}
        self._balance_cache    = {"value": 0.0, "timestamp": 0, "ttl": 2.0}
        self._last_reset_date  = None

        self._base = (_FUT_BASE if live else _FUT_TEST) if mode == "futures" \
                     else (_SPOT_BASE if live else _SPOT_TEST)

        log.info("BinanceTrader: %s %s | risk=%.1f%% | loss_limit=%.1f%%%s",
                 mode.upper(), "LIVE" if live else "TESTNET",
                 risk_pct, daily_loss_limit_pct,
                 " | STOP_MARKET-only" if stop_market_only else "")

    # ─────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_num(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            if v == 0:
                return "0"
            s = repr(v)
            if "e" in s or "E" in s:
                s = f"{v:.20f}".rstrip("0").rstrip(".")
            return s if s else "0"
        return v

    @staticmethod
    def _resolve_futures_symbol(symbol: str, signal):
        mapping = _FUTURES_1000X_SYMBOLS.get(symbol)
        if not mapping:
            return symbol, signal
        new_sym, mult = mapping
        from types import SimpleNamespace
        scale_attrs = {"price", "tp1", "tp2", "tp3", "sl"}
        kwargs = {}
        for attr in dir(signal):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(signal, attr)
            except Exception:
                continue
            if callable(val):
                continue
            kwargs[attr] = val * mult if attr in scale_attrs and isinstance(val, (int, float)) else val
        kwargs["symbol"] = new_sym
        return new_sym, SimpleNamespace(**kwargs)

    def _check_day_reset(self):
        now   = datetime.utcnow()
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

    # ─────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────

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
                                   error=f"Balance ${balance_usdt:.2f} too low for min "
                                         f"${info.min_notional:.0f} on {sym}")

        qty = self._fmt_qty(pos_usdt / signal.price, info)

        # FIX: _fmt_qty floors qty, so actual notional may fall just below min_notional.
        # Example: pos_usdt=$5, price=$0.3156 → raw=15.84 → floored=15.8 → notional=$4.987 < $5.
        # Bump qty by one step at a time until notional >= min_notional.
        # Safety cap at 110% of intended pos_usdt to avoid over-sizing.
        max_qty_cap = self._fmt_qty(pos_usdt * 1.10 / signal.price, info)
        while (qty * signal.price < info.min_notional
               and info.step_size > 0
               and qty <= max_qty_cap):
            qty = round(qty + info.step_size, info.qty_precision)
        if qty * signal.price < info.min_notional:
            return TradeResult(False, sym, signal.signal, 0, signal.price,
                               mode=self._mode,
                               error=f"Cannot meet min notional ${info.min_notional:.2f} "
                                     f"for {sym} at current price")

        if qty <= 0:
            return TradeResult(False, sym, signal.signal, 0, signal.price,
                               mode=self._mode, error="Qty rounds to zero")

        log.info("AUTO → %s %s mode=%s qty=%s entry=%s TP1=%s TP2=%s TP3=%s SL=%s",
                 signal.signal, sym, self._mode, qty,
                 self._fmt_num(signal.price), self._fmt_num(signal.tp1),
                 self._fmt_num(signal.tp2), self._fmt_num(signal.tp3),
                 self._fmt_num(signal.sl))
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
                for s in set(o["symbol"] for o in open_orders):
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
        now = time.time()
        if now - self._balance_cache["timestamp"] < self._balance_cache["ttl"]:
            return self._balance_cache["value"]
        bal = self._fetch_balance()
        self._balance_cache = {"value": bal, "timestamp": now, "ttl": 2.0}
        return bal

    def _fetch_balance(self) -> float:
        return float(self.get_balance().get("available_balance", 0) or 0)

    def get_balance(self) -> dict:
        result = {"wallet_balance": 0.0, "available_balance": 0.0,
                  "unrealised_pnl": 0.0, "error": ""}
        try:
            if self._mode == "futures":
                assets = self._req("GET", "/fapi/v2/balance", {}) or []
                for a in assets:
                    if a.get("asset") == "USDT":
                        result["wallet_balance"]    = float(a.get("balance",          0))
                        result["available_balance"] = float(a.get("availableBalance", 0))
                        result["unrealised_pnl"]    = float(a.get("crossUnPnl",       0))
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

    def get_open_positions_symbols(self) -> list[str]:
        try:
            return [p.get("symbol", "") for p in self.get_positions()]
        except Exception:
            return []

    def _order(self, params: dict) -> dict:
        if self._mode != "futures":
            return self._req("POST", "/api/v3/order", params) or {}
        otype      = params.get("type", "")
        algo_types = {"TAKE_PROFIT_MARKET", "STOP_MARKET", "TRAILING_STOP_MARKET",
                      "TAKE_PROFIT", "STOP"}
        if otype in algo_types:
            algo = {
                "algoType": "CONDITIONAL", "symbol": params["symbol"],
                "side": params["side"], "type": otype,
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
            for k in ("workingType", "priceProtect", "callbackRate"):
                if params.get(k):
                    algo[k] = params[k]
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

    # ─────────────────────────────────────────────────────────────
    #  Spot execution
    # ─────────────────────────────────────────────────────────────

    def _spot(self, sym, signal, qty, info, risk_usdt, pos_usdt) -> TradeResult:
        if signal.signal != "BUY":
            return TradeResult(False, sym, signal.signal, qty, signal.price, mode="spot",
                               error="SELL signals not supported on spot.")
        if signal.tp1 <= signal.price:
            return TradeResult(False, sym, signal.signal, qty, signal.price, mode="spot",
                               error="TP1 must be above entry")
        if signal.sl >= signal.price:
            return TradeResult(False, sym, signal.signal, qty, signal.price, mode="spot",
                               error="SL must be below entry")

        buy = self._req("POST", "/api/v3/order", {
            "symbol": sym, "side": "BUY", "type": "MARKET", "quantity": qty,
        })
        if not buy or "orderId" not in buy:
            raise Exception(f"Market BUY rejected: {buy}")
        fills      = buy.get("fills", [])
        fill_price = float(fills[0]["price"]) if fills else signal.price
        entry_id   = str(buy["orderId"])
        log.info("Spot BUY: %s qty=%s @ %s id=%s", sym, qty, self._fmt_num(fill_price), entry_id)

        q_oco  = self._fmt_qty(qty * 0.40, info)
        q_tp2  = self._fmt_qty(qty * 0.35, info)
        q_tp3  = self._fmt_qty(qty * 0.25, info)
        q_sl60 = self._fmt_qty(qty * 0.60, info)
        tp1_p  = self._fp(signal.tp1,        info)
        tp2_p  = self._fp(signal.tp2,        info)
        tp3_p  = self._fp(signal.tp3,        info)
        sl_p   = self._fp(signal.sl,         info)
        sl_lim = self._fp(signal.sl * 0.998, info)
        sl_lim2= self._fp(signal.sl * 0.997, info)

        def _ok(label, sub_qty, price):
            if sub_qty < info.min_qty:
                log.warning("Spot %s skipped for %s: qty %.6f < min_qty %.6f",
                            label, sym, sub_qty, info.min_qty)
                return False
            n = sub_qty * price
            if n < info.min_notional:
                log.warning("Spot %s skipped for %s: notional $%.4f < min $%.2f",
                            label, sym, n, info.min_notional)
                return False
            return True

        oco_id = tp1_id = sl_id = ""
        if _ok("OCO(TP1+SL1)", q_oco, fill_price):
            oco = self._req("POST", "/api/v3/order/oco", {
                "symbol": sym, "side": "SELL", "quantity": q_oco,
                "price": tp1_p, "stopPrice": sl_p,
                "stopLimitPrice": sl_lim, "stopLimitTimeInForce": "GTC",
            })
            if oco:
                oco_id = str(oco.get("orderListId", ""))
                orders = oco.get("orders", [])
                tp1_id = str(orders[0]["orderId"]) if orders else ""
                sl_id  = str(orders[1]["orderId"]) if len(orders) > 1 else ""
                log.info("Spot OCO: %s TP1=%s SL=%s oco=%s ✅", sym, tp1_id, sl_id, oco_id)
            else:
                log.error("OCO FAILED for %s — TP1+SL (40%%) unprotected!", sym)

        tp2_id = ""
        if _ok("TP2", q_tp2, tp2_p):
            r = self._req("POST", "/api/v3/order", {
                "symbol": sym, "side": "SELL", "type": "LIMIT",
                "timeInForce": "GTC", "quantity": q_tp2, "price": tp2_p,
            })
            if r and "orderId" in r:
                tp2_id = str(r["orderId"])
                log.info("Spot TP2: %s qty=%s @ %s id=%s ✅", sym, q_tp2, tp2_p, tp2_id)

        tp3_id = ""
        if _ok("TP3", q_tp3, tp3_p):
            r = self._req("POST", "/api/v3/order", {
                "symbol": sym, "side": "SELL", "type": "LIMIT",
                "timeInForce": "GTC", "quantity": q_tp3, "price": tp3_p,
            })
            if r and "orderId" in r:
                tp3_id = str(r["orderId"])
                log.info("Spot TP3: %s qty=%s @ %s id=%s ✅", sym, q_tp3, tp3_p, tp3_id)

        sl2_id = ""
        if _ok("SL(TP2+TP3 60%)", q_sl60, fill_price) and sl_p > 0:
            r2 = self._req("POST", "/api/v3/order", {
                "symbol": sym, "side": "SELL", "type": "STOP_LOSS_LIMIT",
                "timeInForce": "GTC", "quantity": q_sl60,
                "stopPrice": sl_p, "price": sl_lim2,
            })
            if r2 and "orderId" in r2:
                sl2_id = str(r2["orderId"])
                log.info("Spot SL (60%%): %s qty=%s trigger=%s id=%s ✅",
                         sym, q_sl60, sl_p, sl2_id)
            else:
                code = (r2 or {}).get("code", 0)
                log.warning("Spot extra SL FAILED for %s: code=%d "
                            "— TP2+TP3 (60%%) tranche unprotected", sym, code)

        log.info("Spot orders: entry=%s tp1=%s tp2=%s tp3=%s sl1=%s sl2=%s",
                 entry_id, tp1_id or "skip", tp2_id or "skip",
                 tp3_id or "skip", sl_id or "skip⚠️", sl2_id or "skip⚠️")

        return TradeResult(True, sym, "BUY", qty, fill_price, mode="spot",
                           position_usdt=round(pos_usdt, 2), risk_usdt=round(risk_usdt, 2),
                           entry_order_id=entry_id, oco_id=oco_id,
                           tp1_order_id=tp1_id, tp2_order_id=tp2_id,
                           tp3_order_id=tp3_id, sl_order_id=sl_id)

    # ─────────────────────────────────────────────────────────────
    #  Protect open SPOT positions (new in v8)
    # ─────────────────────────────────────────────────────────────

    def protect_spot_positions(self, signal_lookup: dict) -> list:
        """
        Each cycle:
        1. Read account balances — find assets held (non-USDT, balance > 0).
        2. Cancel ORPHAN SELL orders — open SELL orders for assets whose total
           balance is now zero (position closed/filled, orders left dangling).
        3. For each held asset that matches a pending SignalRecord in signal_lookup:
           - Determine which order types are already open.
           - Place ONLY missing ones (OCO / TP2 / TP3 / extra-SL), skipping any
             whose qty or notional falls below Binance's minimums.

        signal_lookup: dict[symbol: str → sig object with .tp1 .tp2 .tp3 .sl .price]
        """
        if self._mode != "spot":
            return []
        protected = []
        try:
            # ── 1. Fetch account balances ───────────────────────
            acct = self._req("GET", "/api/v3/account", {}) or {}
            balances: dict[str, float] = {}     # asset → total (free + locked)
            free_bal: dict[str, float] = {}     # asset → free
            for a in acct.get("balances", []):
                free  = float(a.get("free",   0))
                locked= float(a.get("locked", 0))
                total = free + locked
                if total > 0:
                    balances[a["asset"]] = total
                if free > 0:
                    free_bal[a["asset"]] = free

            # ── 2. Fetch all open spot orders ───────────────────
            open_orders = self._req("GET", "/api/v3/openOrders", {}) or []
            sym_orders: dict[str, list] = {}
            for o in open_orders:
                sym_orders.setdefault(o.get("symbol", ""), []).append(o)

            # ── 3. Cancel orphan SELL orders ────────────────────
            # An orphan is a SELL order for a symbol where the underlying asset
            # balance is now 0 — meaning the position was closed but the order
            # was never cancelled.
            orphans_cancelled = []
            for sym, orders in list(sym_orders.items()):
                if not sym.endswith("USDT"):
                    continue
                asset = sym[:-4]   # strip "USDT"
                if balances.get(asset, 0) > 0:
                    continue       # position still open, leave orders alone
                for o in orders:
                    if o.get("side") == "SELL":
                        try:
                            self._req("DELETE", "/api/v3/order",
                                      {"symbol": sym, "orderId": o["orderId"]})
                            orphans_cancelled.append(f"{sym}#{o['orderId']}")
                        except Exception as _e:
                            log.debug("Orphan cancel error %s: %s", sym, _e)
            if orphans_cancelled:
                log.info("Spot orphan orders cancelled (%d): %s",
                         len(orphans_cancelled),
                         ", ".join(orphans_cancelled[:12]) +
                         (f" … +{len(orphans_cancelled)-12} more"
                          if len(orphans_cancelled) > 12 else ""))
                # Remove cancelled orders from our local index
                for entry in orphans_cancelled:
                    s = entry.split("#")[0]
                    sym_orders.pop(s, None)

            # ── 4. Protect positions that match signal_lookup ───
            for sym, sig in signal_lookup.items():
                asset = sym[:-4] if sym.endswith("USDT") else None
                if not asset:
                    continue
                total_bal = balances.get(asset, 0)
                if total_bal <= 0:
                    continue

                info = self._get_symbol_info(sym)
                if total_bal < info.min_qty:
                    log.debug("Spot %s: balance %.6f < min_qty — skip", sym, total_bal)
                    continue

                orders_for_sym = sym_orders.get(sym, [])

                # Classify existing orders
                has_oco = any(o.get("orderListId", -1) != -1
                              for o in orders_for_sym)
                has_tp  = any(o.get("side") == "SELL" and o.get("type") == "LIMIT"
                              for o in orders_for_sym)
                has_sl  = any(o.get("side") == "SELL" and
                              o.get("type") in ("STOP_LOSS_LIMIT", "STOP_LOSS")
                              for o in orders_for_sym)

                log.info("  Spot %s: balance=%.4f orders=%d "
                         "has_oco=%s has_tp=%s has_sl=%s",
                         sym, total_bal, len(orders_for_sym),
                         "✅" if has_oco else "❌",
                         "✅" if has_tp  else "❌",
                         "✅" if has_sl  else "❌")

                if has_oco and has_sl:
                    log.info("  Spot %s: protected ✅", sym)
                    continue

                tp1_p  = self._fp(sig.tp1,        info)
                tp2_p  = self._fp(sig.tp2,        info)
                tp3_p  = self._fp(sig.tp3,        info)
                sl_p   = self._fp(sig.sl,         info)
                sl_lim = self._fp(sig.sl * 0.998, info)
                sl_lim2= self._fp(sig.sl * 0.997, info)

                if sl_p <= 0 or tp1_p <= sl_p:
                    log.warning("  Spot %s: bad TP/SL levels — skip", sym)
                    continue

                # Use total balance as position size for sub-order quantities
                qty = total_bal

                # Helper: notional + qty check before placing
                def _ok_sp(label, sub_qty, price):
                    if sub_qty < info.min_qty:
                        log.warning("  Spot protect %s skipped for %s: "
                                    "qty %.6f < min_qty %.6f",
                                    label, sym, sub_qty, info.min_qty)
                        return False
                    n = sub_qty * price
                    if n < info.min_notional:
                        log.warning("  Spot protect %s skipped for %s: "
                                    "notional $%.4f < min $%.2f",
                                    label, sym, n, info.min_notional)
                        return False
                    return True

                q_oco  = self._fmt_qty(qty * 0.40, info)
                q_tp2  = self._fmt_qty(qty * 0.35, info)
                q_tp3  = self._fmt_qty(qty * 0.25, info)
                q_sl60 = self._fmt_qty(qty * 0.60, info)

                did_protect = False

                # Place OCO (TP1 + SL1 for 40%) if missing
                if not has_oco and _ok_sp("OCO(TP1+SL1)", q_oco, tp1_p):
                    oco = self._req("POST", "/api/v3/order/oco", {
                        "symbol": sym, "side": "SELL", "quantity": q_oco,
                        "price": tp1_p, "stopPrice": sl_p,
                        "stopLimitPrice": sl_lim, "stopLimitTimeInForce": "GTC",
                    })
                    # FIX: _req returns the error JSON dict on 400 (truthy!) — must
                    # check orderListId to confirm the OCO was actually created.
                    if isinstance(oco, dict) and oco.get("orderListId"):
                        log.info("  Spot %s: OCO (TP1+SL1) placed ✅", sym)
                        did_protect = True
                    else:
                        code = (oco or {}).get("code", 0) if isinstance(oco, dict) else 0
                        log.error("  Spot %s: OCO FAILED code=%d — TP1+SL1 unprotected!",
                                  sym, code)

                # Place TP2 LIMIT if no TP2 exists yet
                tp2_exists = any(o.get("side") == "SELL" and
                                 o.get("type") == "LIMIT" and
                                 o.get("orderListId", -1) == -1
                                 for o in orders_for_sym)
                if not tp2_exists and _ok_sp("TP2", q_tp2, tp2_p):
                    r = self._req("POST", "/api/v3/order", {
                        "symbol": sym, "side": "SELL", "type": "LIMIT",
                        "timeInForce": "GTC", "quantity": q_tp2, "price": tp2_p,
                    })
                    if r and "orderId" in r:
                        log.info("  Spot %s: TP2 placed ✅", sym)
                        did_protect = True

                # Place TP3 LIMIT if no standalone TP exists
                tp3_exists = sum(1 for o in orders_for_sym
                                 if o.get("side") == "SELL" and
                                 o.get("type") == "LIMIT" and
                                 o.get("orderListId", -1) == -1) >= 2
                if not tp3_exists and _ok_sp("TP3", q_tp3, tp3_p):
                    r = self._req("POST", "/api/v3/order", {
                        "symbol": sym, "side": "SELL", "type": "LIMIT",
                        "timeInForce": "GTC", "quantity": q_tp3, "price": tp3_p,
                    })
                    if r and "orderId" in r:
                        log.info("  Spot %s: TP3 placed ✅", sym)
                        did_protect = True

                # Place extra SL for 60% tranche if missing
                if not has_sl and _ok_sp("SL(60%)", q_sl60, sl_p):
                    r2 = self._req("POST", "/api/v3/order", {
                        "symbol": sym, "side": "SELL", "type": "STOP_LOSS_LIMIT",
                        "timeInForce": "GTC", "quantity": q_sl60,
                        "stopPrice": sl_p, "price": sl_lim2,
                    })
                    if r2 and "orderId" in r2:
                        log.info("  Spot %s: SL (60%%) placed ✅", sym)
                        did_protect = True
                    else:
                        code = (r2 or {}).get("code", 0)
                        log.warning("  Spot %s: SL (60%%) FAILED code=%d ⚠️", sym, code)

                if did_protect:
                    protected.append(sym)
                elif not (has_oco and has_sl):
                    log.warning("  Spot %s: partially protected — "
                                "balance too small for some orders", sym)

        except Exception as e:
            log.error("protect_spot_positions error: %s", e, exc_info=True)

        return protected

    # ─────────────────────────────────────────────────────────────
    #  Futures execution
    # ─────────────────────────────────────────────────────────────

    def _futures(self, sym, signal, qty, info, risk_usdt, pos_usdt) -> TradeResult:
        new_sym, signal = self._resolve_futures_symbol(sym, signal)
        if new_sym != sym:
            mult = _FUTURES_1000X_SYMBOLS[sym][1]
            log.info("Symbol remap: %s → %s (×%d)", sym, new_sym, mult)
            sym  = new_sym
            info = self._get_symbol_info(sym)       # correct step_size FIRST
            qty  = self._fmt_qty(qty / mult, info)  # then format qty
            if qty <= 0:
                return TradeResult(False, sym, signal.signal, 0, signal.price,
                                   mode="futures",
                                   error=f"After 1000x remap, qty rounds to zero on {sym}")

        is_long    = signal.signal == "BUY"
        side       = "BUY"  if is_long else "SELL"
        close_side = "SELL" if is_long else "BUY"

        if is_long and signal.tp1 <= signal.price:
            return TradeResult(False, sym, signal.signal, qty, signal.price, mode="futures",
                               error=f"TP1 {signal.tp1:.6g} must be above entry")
        if not is_long and signal.tp1 >= signal.price:
            return TradeResult(False, sym, signal.signal, qty, signal.price, mode="futures",
                               error=f"TP1 {signal.tp1:.6g} must be below entry")

        lev = 3 if getattr(signal, "btc_score", 50) >= 62 else 2
        self._req("POST", "/fapi/v1/leverage", {"symbol": sym, "leverage": lev})

        entry = self._req("POST", "/fapi/v1/order", {
            "symbol": sym, "side": side, "type": "MARKET", "quantity": qty,
        })
        if not entry or "orderId" not in entry:
            raise Exception(f"Futures {side} rejected: {entry}")
        entry_id   = str(entry["orderId"])
        fill_price = float(entry.get("avgPrice", signal.price)) or signal.price
        log.info("Futures %s %s: qty=%s @ %s lev=%dx id=%s",
                 "LONG" if is_long else "SHORT", sym, qty,
                 self._fmt_num(fill_price), lev, entry_id)

        q_tp1 = self._fmt_qty(qty * 0.40, info)
        q_tp2 = self._fmt_qty(qty * 0.35, info)
        q_tp3 = self._fmt_qty(qty * 0.15, info)
        q_tsl = self._fmt_qty(qty * 0.10, info)
        tp1_p = self._fp(signal.tp1, info)
        tp2_p = self._fp(signal.tp2, info)
        tp3_p = self._fp(signal.tp3, info)
        sl_p  = self._fp(signal.sl,  info)

        def _place_tp(label, stop_p, q) -> str:
            if q < info.min_qty or stop_p <= 0:
                return ""
            attempts = [("TAKE_PROFIT_MARKET", {
                "symbol": sym, "side": close_side, "type": "TAKE_PROFIT_MARKET",
                "stopPrice": stop_p, "quantity": q, "timeInForce": "GTC",
                "workingType": "MARK_PRICE", "priceProtect": "true", "reduceOnly": "true",
            })]
            if not self._stop_market_only:
                attempts.append(("LIMIT", {
                    "symbol": sym, "side": close_side, "type": "LIMIT",
                    "price": stop_p, "quantity": q,
                    "timeInForce": "GTC", "reduceOnly": "true",
                }))
            for name, params in attempts:
                r = self._order(params)
                if isinstance(r, dict) and "orderId" in r:
                    log.info("Futures %s (%s) @ %s qty=%s id=%s ✅",
                             label, name, self._fmt_num(stop_p), q, r["orderId"])
                    return str(r["orderId"])
                code = r.get("code", 0) if isinstance(r, dict) else 0
                if code in (-4120, -1104) and name != "LIMIT":
                    log.warning("Futures %s: rejected (code %d) — trying LIMIT", label, code)
                    continue
                log.error("Futures %s FAILED (%s) @ %s: code=%d",
                          label, name, self._fmt_num(stop_p), code)
                return ""
            return ""

        tp1_id = _place_tp("TP1", tp1_p, q_tp1)
        tp2_id = _place_tp("TP2", tp2_p, q_tp2)
        tp3_id = _place_tp("TP3", tp3_p, q_tp3)

        sl_id = ""
        if sl_p <= 0:
            log.error("Futures SL skipped — price is 0 for %s", sym)
        else:
            sl_qty_val = self._fmt_qty(qty, info)
            sl_lim_p   = self._fp(sl_p * (1.005 if not is_long else 0.995), info)
            mark_price = 0
            try:
                mp = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                                  params={"symbol": sym}, timeout=3)
                if mp.ok:
                    mark_price = float(mp.json().get("markPrice", 0))
            except Exception:
                pass
            if mark_price > 0:
                if is_long and sl_p >= mark_price:
                    new_sl = self._fp(mark_price * 0.985, info)
                    log.warning("SL %s would trigger immediately → adjusting to %s",
                                self._fmt_num(sl_p), self._fmt_num(new_sl))
                    sl_p = new_sl; sl_lim_p = self._fp(sl_p * 0.995, info)
                elif not is_long and sl_p <= mark_price:
                    new_sl = self._fp(mark_price * 1.015, info)
                    log.warning("SL %s would trigger immediately → adjusting to %s",
                                self._fmt_num(sl_p), self._fmt_num(new_sl))
                    sl_p = new_sl; sl_lim_p = self._fp(sl_p * 1.005, info)

            sl_attempts = [("STOP_qty", {
                "symbol": sym, "side": close_side, "type": "STOP",
                "stopPrice": sl_p, "price": sl_lim_p,
                "quantity": sl_qty_val, "timeInForce": "GTC", "reduceOnly": "true",
            })]
            if not self._stop_market_only:
                sl_attempts.append(("LIMIT_sl", {
                    "symbol": sym, "side": close_side, "type": "LIMIT",
                    "price": sl_p, "quantity": sl_qty_val,
                    "timeInForce": "GTC", "reduceOnly": "true",
                }))
            for sl_type, sl_params in sl_attempts:
                if sl_type == "LIMIT_sl" and mark_price > 0:
                    sl_params["price"] = max(mark_price * 0.95,
                                             min(sl_params["price"], mark_price * 1.05))
                sl_r = self._order(sl_params)
                if isinstance(sl_r, dict) and "orderId" in sl_r:
                    sl_id = str(sl_r["orderId"])
                    log.info("Futures SL (%s) @ %s id=%s ✅",
                             sl_type, self._fmt_num(sl_p), sl_id)
                    break
                code = sl_r.get("code", 0) if isinstance(sl_r, dict) else 0
                if code in (-4120, -1104, -4045):
                    log.warning("Futures SL: %s rejected (code %d) → next", sl_type, code)
                    continue
                log.error("Futures SL FAILED (%s) @ %s: code=%d ⚠️",
                          sl_type, self._fmt_num(sl_p), code)
                try:
                    from dashboard.views import push_trade_alert
                    push_trade_alert("error", f"⚠️ NO SL on {sym} code={code}")
                except Exception:
                    pass
                break

        trailing_id = ""
        if q_tsl >= info.min_qty and tp1_p > 0:
            cur_mark = 0
            try:
                mp = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                                  params={"symbol": sym}, timeout=3)
                if mp.ok:
                    cur_mark = float(mp.json().get("markPrice", 0))
            except Exception:
                pass
            if cur_mark > 0 and is_long:
                act_p = tp1_p if tp1_p >= cur_mark else self._fp(cur_mark * 1.001, info)
            elif cur_mark > 0 and not is_long:
                act_p = tp1_p if tp1_p <= cur_mark else self._fp(cur_mark * 0.999, info)
            else:
                act_p = tp1_p
            trail_r = self._order({
                "symbol": sym, "side": close_side, "type": "TRAILING_STOP_MARKET",
                "quantity": q_tsl, "callbackRate": 1.5,
                "activationPrice": act_p, "workingType": "MARK_PRICE", "reduceOnly": "true",
            })
            if isinstance(trail_r, dict) and "orderId" in trail_r:
                trailing_id = str(trail_r["orderId"])
                log.info("Futures TSL activation=%s callback=1.5%% id=%s ✅",
                         self._fmt_num(act_p), trailing_id)

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

    # ─────────────────────────────────────────────────────────────
    #  Safety
    # ─────────────────────────────────────────────────────────────

    def _pre_flight(self, balance: float) -> tuple[bool, str]:
        if not self._key or not self._secret:
            return False, "No API keys in .env"
        if balance <= 0:
            return False, "Balance is $0.00"
        if balance < 5:
            return False, f"Balance ${balance:.2f} too low (minimum $5)"
        if self._daily_loss >= self._loss_limit:
            return False, f"Daily loss limit {self._loss_limit}% reached"
        if self._daily_trades >= self._max_trades:
            return False, f"Daily trade limit ({self._max_trades}) reached"
        return True, ""

    # ─────────────────────────────────────────────────────────────
    #  Symbol info
    # ─────────────────────────────────────────────────────────────

    def _get_symbol_info(self, sym: str) -> SymbolInfo:
        if sym in self._sym_cache:
            return self._sym_cache[sym]
        info = SymbolInfo()
        try:
            endpoint = "/fapi/v1/exchangeInfo" if self._mode == "futures" \
                       else "/api/v3/exchangeInfo"
            ex = self._pub("GET", endpoint,
                           {} if self._mode == "futures" else {"symbol": sym})
            for item in (ex or {}).get("symbols", []):
                if item.get("symbol") != sym:
                    continue
                filters = {f["filterType"]: f for f in item.get("filters", [])}
                pf  = filters.get("PRICE_FILTER", {})
                lf  = filters.get("LOT_SIZE",     {})
                nf  = filters.get("MIN_NOTIONAL", {}) or filters.get("NOTIONAL", {})
                tick = float(pf.get("tickSize",   "0.0001") or "0.0001")
                step = float(lf.get("stepSize",   "0.001")  or "0.001")
                mq   = float(lf.get("minQty",     "0.001")  or "0.001")
                mn_r = float(nf.get("minNotional","5") or nf.get("notional","5") or "5")
                mn   = max(mn_r, 20.0) if self._mode == "futures" else mn_r

                def _prec(val: float) -> int:
                    txt = str(val)
                    return len(txt.rstrip("0").split(".")[1]) if "." in txt else 0

                info = SymbolInfo(_prec(tick), _prec(step), mq, mn, tick, step)
                info.tick_size = tick
                break
        except Exception as e:
            log.debug("Symbol info %s: %s — using defaults", sym, e)
        self._sym_cache[sym] = info
        return info

    # ─────────────────────────────────────────────────────────────
    #  HTTP
    # ─────────────────────────────────────────────────────────────

    def _req(self, method: str, path: str, params: dict) -> dict | None:
        p = {k: (self._fmt_num(v) if isinstance(v, float) else v)
             for k, v in params.items() if v is not None}
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
        if price <= 0:
            return 0.0
        tick = info.tick_size
        if tick and tick > 0:
            snapped = round(price / tick) * tick
            if snapped <= 0 and price > 0:
                snapped = tick * 2
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
        step = info.step_size
        if step <= 0:
            return round(qty, info.qty_precision)
        return round((qty // step) * step, info.qty_precision)

    # ─────────────────────────────────────────────────────────────
    #  Cancel regular + algo orders for ONE futures symbol
    # ─────────────────────────────────────────────────────────────

    def _cancel_all_orders_for_symbol(self, sym: str) -> None:
        self._req("DELETE", "/fapi/v1/allOpenOrders", {"symbol": sym})
        try:
            ao = self._req("GET", "/fapi/v1/openAlgoOrders", {"symbol": sym}) or {}
            ao_list = ao.get("algoOrders", []) if isinstance(ao, dict) else \
                      (ao if isinstance(ao, list) else [])
            for item in ao_list:
                aid = item.get("algoId") or item.get("orderId")
                if aid:
                    self._req("DELETE", "/fapi/v1/algoOrder",
                              {"symbol": sym, "algoId": aid})
        except Exception as _e:
            log.debug("Algo cancel error for %s: %s", sym, _e)

    # ─────────────────────────────────────────────────────────────
    #  protect_open_positions — futures, surgical per-symbol
    # ─────────────────────────────────────────────────────────────

    def protect_open_positions(self, signal_lookup: dict) -> list:
        """Scan open futures positions; surgically add missing TP/SL/TSL."""
        if self._mode != "futures":
            return []
        self._check_day_reset()
        protected = []
        try:
            positions   = self.get_positions()
            open_orders = self.get_open_orders() or []

            sym_orders: dict[str, list] = {}
            for o in open_orders:
                sym_orders.setdefault(o.get("symbol", ""), []).append(o)

            SL_TYPES  = {"STOP_MARKET", "STOP", "STOP_LOSS", "STOP_LOSS_LIMIT"}
            TSL_TYPES = {"TRAILING_STOP_MARKET"}
            TP_TYPES  = {"TAKE_PROFIT_MARKET", "TAKE_PROFIT", "LIMIT"}

            total_orders = sum(len(v) for v in sym_orders.values())
            log.info("protect_open_positions: %d positions, %d open orders",
                     len(positions), total_orders)

            # ── Orphan sweep ─────────────────────────────────────────────
            # Orders for symbols with NO open position are stale remnants
            # from positions that closed (all TPs hit, SL hit, or manually
            # closed). They count toward Binance 's 200-order cap and cause
            # -4045 on new trades. Cancel them BEFORE touching live positions.
            open_syms = {p.get("symbol", "") for p in positions
                         if float(p.get("positionAmt", 0)) != 0}
            orphan_syms = [s for s in sym_orders if s and s not in open_syms]
            if orphan_syms:
                log.info("Orphan orders for %d closed positions — cancelling: %s",
                         len(orphan_syms), orphan_syms)
                for _osym in orphan_syms:
                    self._cancel_all_orders_for_symbol(_osym)
                    sym_orders.pop(_osym, None)
                total_orders = sum(len(v) for v in sym_orders.values())
                log.info("After orphan sweep: %d orders remaining", total_orders)

            for pos in positions:
                sym = pos.get("symbol", "")
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

                if has_tp and has_sl and has_tsl:
                    if len(orders_for_sym) <= _PER_SYM_MAX_ORDERS:
                        log.info("  %s: fully protected ✅", sym)
                        continue
                    else:
                        # Fully protected but too many orders (e.g. ALGOUSDT=15, TAOUSDT=10).
                        # These accumulate because the "fully protected → skip" check ran
                        # before the order-count check could clean them. The excess pushes
                        # the account toward Binance's 200-order cap (-4045).
                        # Cancel and rebuild this symbol only — other positions untouched.
                        log.warning(
                            "  %s: fully protected but %d orders > %d — cleaning excess orders",
                            sym, len(orders_for_sym), _PER_SYM_MAX_ORDERS,
                        )
                        self._cancel_all_orders_for_symbol(sym)
                        orders_for_sym = self.get_open_orders(sym) or []
                        sym_orders[sym] = orders_for_sym
                        has_sl  = any(o.get("type") in SL_TYPES  for o in orders_for_sym)
                        has_tsl = any(o.get("type") in TSL_TYPES for o in orders_for_sym)
                        has_tp  = any(o.get("type") in TP_TYPES  for o in orders_for_sym)
                        log.info("  %s after clean: %d orders has_tp=%s has_sl=%s has_tsl=%s",
                                 sym, len(orders_for_sym),
                                 "✅" if has_tp else "❌",
                                 "✅" if has_sl else "❌",
                                 "✅" if has_tsl else "❌")
                        # Fall through to re-protect what's now missing

                if has_tp and has_sl and not has_tsl:
                    log.info("  %s: TP+SL ok — adding missing TSL only", sym)

                # Surgical clean only for this symbol if too many orders
                if len(orders_for_sym) > _PER_SYM_MAX_ORDERS:
                    log.warning("  %s: %d orders > %d — cancelling this symbol only",
                                sym, len(orders_for_sym), _PER_SYM_MAX_ORDERS)
                    self._cancel_all_orders_for_symbol(sym)
                    orders_for_sym = self.get_open_orders(sym) or []
                    sym_orders[sym] = orders_for_sym
                    has_sl  = any(o.get("type") in SL_TYPES  for o in orders_for_sym)
                    has_tsl = any(o.get("type") in TSL_TYPES for o in orders_for_sym)
                    has_tp  = any(o.get("type") in TP_TYPES  for o in orders_for_sym)
                    log.info("  %s after clean: %d orders has_tp=%s has_sl=%s has_tsl=%s",
                             sym, len(orders_for_sym),
                             "✅" if has_tp else "❌",
                             "✅" if has_sl else "❌",
                             "✅" if has_tsl else "❌")

                is_long    = amt > 0
                close_side = "SELL" if is_long else "BUY"
                qty        = abs(amt)
                entry      = float(pos.get("entryPrice", 0))
                info       = self._get_symbol_info(sym)
                sig        = signal_lookup.get(sym)

                if sig and sig.tp1 > 0 and sig.sl > 0:
                    tp1_p = self._fp(sig.tp1, info)
                    tp2_p = self._fp(sig.tp2, info)
                    tp3_p = self._fp(getattr(sig, "tp3", 0) or tp2_p * 1.02, info)
                    sl_p  = self._fp(sig.sl,  info)
                    log.info("Using signal levels for %s: TP1=%s TP2=%s TP3=%s SL=%s",
                             sym, self._fmt_num(tp1_p), self._fmt_num(tp2_p),
                             self._fmt_num(tp3_p), self._fmt_num(sl_p))
                else:
                    tp1_pct = 0.03; tp2_pct = 0.06; tp3_pct = 0.10; sl_pct = 0.04
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
                    log.warning("Default TP/SL for %s from entry=%s: TP1=%s TP2=%s TP3=%s SL=%s",
                                sym, self._fmt_num(entry),
                                self._fmt_num(tp1_p), self._fmt_num(tp2_p),
                                self._fmt_num(tp3_p), self._fmt_num(sl_p))

                if sl_p <= 0:
                    log.error("Skipping %s — cannot calculate SL price", sym)
                    continue

                q_tp1 = self._fmt_qty(qty * 0.40, info)
                q_tp2 = self._fmt_qty(qty * 0.35, info)
                q_tsl = self._fmt_qty(qty * 0.25, info)

                log.info("Protecting: %s %s qty=%.4f entry=%s → TP1=%s TP2=%s SL=%s",
                         "LONG" if is_long else "SHORT", sym, qty,
                         self._fmt_num(entry), self._fmt_num(tp1_p),
                         self._fmt_num(tp2_p), self._fmt_num(sl_p))

                def _prot_tp(label, stop_p, qty_p):
                    if stop_p <= 0 or qty_p < info.min_qty:
                        return
                    attempts = [("TAKE_PROFIT_MARKET", {
                        "symbol": sym, "side": close_side, "type": "TAKE_PROFIT_MARKET",
                        "stopPrice": stop_p, "quantity": qty_p,
                        "workingType": "MARK_PRICE", "priceProtect": "true", "reduceOnly": "true",
                    })]
                    if not self._stop_market_only:
                        attempts.append(("LIMIT", {
                            "symbol": sym, "side": close_side, "type": "LIMIT",
                            "price": stop_p, "quantity": qty_p,
                            "timeInForce": "GTC", "reduceOnly": "true",
                        }))
                    for name, params in attempts:
                        r = self._order(params)
                        if isinstance(r, dict) and "orderId" in r:
                            log.info("  %s (%s) @ %s id=%s ✅",
                                     label, name, self._fmt_num(stop_p), r["orderId"])
                            return
                        code = r.get("code", 0) if isinstance(r, dict) else 0
                        if code in (-4120, -1104, -2015) and name != "LIMIT":
                            log.warning("  %s: algo rejected (code %d) → LIMIT", label, code)
                            continue
                        log.error("  %s FAILED (%s): code=%d %s",
                                  label, name, code, r.get("msg", ""))
                        return

                if not has_tp:
                    _prot_tp("TP1", tp1_p, q_tp1)
                    _prot_tp("TP2", tp2_p, q_tp2)
                    _prot_tp("TP3", tp3_p, q_tsl)

                if not has_sl:
                    _sl_qty  = self._fmt_qty(qty, info)
                    _sl_limp = self._fp(sl_p * (1.005 if not is_long else 0.995), info)
                    mark_price = 0
                    try:
                        mp = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                                          params={"symbol": sym}, timeout=3)
                        if mp.ok:
                            mark_price = float(mp.json().get("markPrice", 0))
                    except Exception:
                        pass
                    if mark_price > 0:
                        if is_long and sl_p >= mark_price:
                            new_sl = self._fp(mark_price * 0.985, info)
                            log.warning("  %s: SL %s >= mark → %s",
                                        sym, self._fmt_num(sl_p), self._fmt_num(new_sl))
                            sl_p = new_sl; _sl_limp = self._fp(sl_p * 0.995, info)
                        elif not is_long and sl_p <= mark_price:
                            new_sl = self._fp(mark_price * 1.015, info)
                            log.warning("  %s: SL %s <= mark → %s",
                                        sym, self._fmt_num(sl_p), self._fmt_num(new_sl))
                            sl_p = new_sl; _sl_limp = self._fp(sl_p * 1.005, info)
                    if sl_p > 0:
                        sl_att = [("STOP_MARKET", {
                            "stopPrice": sl_p, "quantity": _sl_qty,
                            "workingType": "MARK_PRICE", "priceProtect": "true",
                            "reduceOnly": "true",
                        })]
                        if not self._stop_market_only:
                            sl_att.append(("STOP", {
                                "stopPrice": sl_p, "price": _sl_limp,
                                "quantity": _sl_qty,
                                "timeInForce": "GTC", "reduceOnly": "true",
                            }))
                        for sl_t, sl_p2 in sl_att:
                            if sl_t == "STOP" and mark_price > 0:
                                sl_p2["price"] = max(mark_price * 0.95,
                                                     min(sl_p2.get("price", sl_p),
                                                         mark_price * 1.05))
                            sl_r = self._order({"symbol": sym, "side": close_side,
                                                "type": sl_t, **sl_p2})
                            if isinstance(sl_r, dict) and "orderId" in sl_r:
                                log.info("  SL (%s) @ %s id=%s ✅",
                                         sl_t, self._fmt_num(sl_p), sl_r["orderId"])
                                break
                            code = sl_r.get("code", 0) if isinstance(sl_r, dict) else 0
                            if code in (-4120, -1104) and sl_t != "STOP":
                                log.warning("  SL: %s rejected (code %d) → STOP", sl_t, code)
                                continue
                            log.error("  SL FAILED (%s): code=%d ⚠️ UNPROTECTED!", sl_t, code)
                            try:
                                from dashboard.views import push_trade_alert
                                push_trade_alert("error",
                                                 f"⚠️ {sym} unprotected — SL failed (code {code})")
                            except Exception:
                                pass
                            break
                    else:
                        log.error("  SL skipped — rounds to 0 for %s", sym)
                else:
                    log.debug("  %s: SL already in place", sym)

                if not has_tsl and q_tsl >= info.min_qty and tp1_p > 0:
                    cur_mark = 0
                    try:
                        mp = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                                          params={"symbol": sym}, timeout=3)
                        if mp.ok:
                            cur_mark = float(mp.json().get("markPrice", 0))
                    except Exception:
                        pass
                    if cur_mark > 0 and is_long:
                        act_p = tp1_p if tp1_p >= cur_mark else self._fp(cur_mark * 1.001, info)
                    elif cur_mark > 0 and not is_long:
                        act_p = tp1_p if tp1_p <= cur_mark else self._fp(cur_mark * 0.999, info)
                    else:
                        act_p = tp1_p
                    r = self._order({
                        "symbol": sym, "side": close_side, "type": "TRAILING_STOP_MARKET",
                        "quantity": q_tsl, "callbackRate": 1.5,
                        "activationPrice": act_p,
                        "workingType": "MARK_PRICE", "reduceOnly": "true",
                    })
                    if isinstance(r, dict) and "orderId" in r:
                        log.info("  TSL set activation=%s callback=1.5%% id=%s ✅",
                                 self._fmt_num(act_p), r["orderId"])

                protected.append(sym)

        except Exception as e:
            log.error("protect_open_positions error: %s", e, exc_info=True)

        return protected