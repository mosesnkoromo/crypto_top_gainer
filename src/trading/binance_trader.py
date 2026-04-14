"""
src/trading/binance_trader.py — Institutional Scalp v2
───────────────────────────────────────────────────────
Regime-adaptive execution with state machine position management.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlencode

import pandas as pd
import requests as _requests

from src.utils.logger import get_logger
from src.analysis.trade_state_machine import TradeStateMachine

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
    symbol:          str   = ""


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

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        mode: Literal["spot", "futures"] = "spot",
        live: bool = False,
        risk_pct: float = 1.5,
        daily_loss_limit_pct: float = 6.0,
        max_trades_per_day: int = 999,
    ):
        self._key        = api_key
        self._secret     = api_secret
        self._mode       = mode
        self._live       = live
        self._recently_placed: dict = {}   # {symbol: timestamp} for SL grace period
        self._risk       = risk_pct
        self._loss_limit = daily_loss_limit_pct
        self._max_trades = max_trades_per_day
        self._daily_loss   = 0.0
        self._daily_trades = 0
        self._sym_cache: dict[str, SymbolInfo] = {}
        self._state_machine = TradeStateMachine()

        self._base = (
            (_FUT_BASE if live else _FUT_TEST) if mode == "futures"
            else (_SPOT_BASE if live else _SPOT_TEST)
        )
        log.info("BinanceTrader: %s %s | risk=%.1f%% | loss_limit=%.1f%%",
                 mode.upper(), "LIVE" if live else "TESTNET",
                 risk_pct, daily_loss_limit_pct)


    # ════════════════════════════════════════════════════════════════════
    # FUTURES SYMBOL NORMALIZATION
    # Some pairs on Binance Futures use a 1000x prefix because the
    # token price is too small (e.g. LUNC, BONK, SHIB, FLOKI, PEPE, XEC).
    # Spot ticker: LUNCUSDT → Futures: 1000LUNCUSDT
    # ════════════════════════════════════════════════════════════════════

    # Known static mapping: spot → futures symbol
    _FUTURES_1000_SYMBOLS = {
        "LUNCUSDT",   "BONKUSDT",   "SHIBUSDT",   "FLOKIUSDT",
        "XECUSDT",    "PEPEUSDT",   "SATSUSDT",   "RATSUSDT",
        "CATUSDT",    "BTTCUSDT",   "HOTUSDT",    "WINUSDT",
        "NFTUSDT",    "DODOGIUSDT",
    }
    # Cache built at runtime from exchange info
    _futures_sym_cache: dict = {}

    def _normalize_futures_sym(self, sym: str) -> str:
        """
        Convert spot symbol to correct Binance Futures symbol.
        - LUNCUSDT → 1000LUNCUSDT  (known 1000-prefix pairs)
        - Others remain unchanged
        Also validates against exchange info and caches result.
        """
        # Check cache first
        if sym in self._futures_sym_cache:
            return self._futures_sym_cache[sym]

        # Known static mapping
        if sym in self._FUTURES_1000_SYMBOLS:
            mapped = "1000" + sym
            self._futures_sym_cache[sym] = mapped
            log.debug("Futures symbol mapped: %s → %s", sym, mapped)
            return mapped

        # Dynamic check: query exchange info to verify symbol exists
        # If "SYM" doesn't exist but "1000SYM" does, use 1000 version
        try:
            info = self._req("GET", "/fapi/v1/exchangeInfo", {}) or {}
            valid = {s["symbol"] for s in info.get("symbols", [])}
            if valid:
                if sym not in valid and ("1000" + sym) in valid:
                    mapped = "1000" + sym
                    self._futures_sym_cache[sym] = mapped
                    # Also add to static set for future calls
                    self._FUTURES_1000_SYMBOLS.add(sym)
                    log.info("Futures symbol auto-detected: %s → %s", sym, mapped)
                    return mapped
                # Cache the valid result
                self._futures_sym_cache[sym] = sym
        except Exception:
            pass

        return sym

    def execute_signal(self, signal, balance_usdt: float) -> TradeResult:
        sym = signal.symbol
        # Normalize to correct futures symbol (handles 1000LUNCUSDT etc.)
        futures_sym = self._normalize_futures_sym(sym) if self._mode == "futures" else sym
        ok, reason = self._pre_flight(balance_usdt)
        if not ok:
            log.warning("Trade blocked [%s]: %s", sym, reason)
            return TradeResult(False, sym, signal.signal, 0, signal.price,
                               mode=self._mode, error=reason)

        info    = self._get_symbol_info(sym)
        sl_dist = abs(signal.price - signal.sl) / max(signal.price, 1e-10)
        atr_val = getattr(signal, "atr", 0) or 0
        if atr_val > 0:
            sl_dist = max(sl_dist, (atr_val * 1.2) / max(signal.price, 1e-10))
        if sl_dist < 0.001:
            return TradeResult(False, sym, signal.signal, 0, signal.price,
                               mode=self._mode, error="SL too close to entry")

        # ML-based dynamic risk sizing
        _ml_p  = float(getattr(signal, "ml_prob",     0) or 0)
        _sn_p  = float(getattr(signal, "sniper_conf", 1) or 1)
        # Count open positions for capital scaling
        try:
            _open_count = len(self.get_positions()) if self._mode == "futures" else 0
        except Exception:
            _open_count = 0
        if   _open_count >= 6: _cap = 0.40   # 6 positions — reduce risk per trade
        elif _open_count == 5: _cap = 0.45
        elif _open_count == 4: _cap = 0.50
        elif _open_count == 3: _cap = 0.60
        elif _open_count == 2: _cap = 0.70
        elif _open_count == 1: _cap = 0.85
        else:                  _cap = 1.00
        # ML confidence scaling
        if _ml_p >= 0.75 or _sn_p >= 0.90: _conf = 1.2
        elif _ml_p >= 0.55 or _sn_p >= 0.65: _conf = 1.0
        else: _conf = 0.7
        dyn_risk = max(min(self._risk * _cap * _conf, 2.5), 0.5)
        log.debug("Risk: ML=%.0f%% sniper=%.0f%% open=%d cap=%.0f%% → %.1f%%",
                  _ml_p*100, _sn_p*100, _open_count, _cap*100, dyn_risk)
        # REDUCED_RISK signals (no candle history) use 0.5× position size
        # These are high-momentum tokens like 币安人生USDT — real edge but unknown history
        _is_reduced = getattr(signal, "tag", "") == "REDUCED_RISK" or                       "REDUCED_RISK" in getattr(signal, "action", "")
        if _is_reduced:
            dyn_risk = dyn_risk * 0.5
            log.info("  ⚡ REDUCED RISK mode — using %.1f%% risk (0.5× normal)", dyn_risk)

        risk_usdt = balance_usdt * (dyn_risk / 100)
        pos_usdt  = min(risk_usdt / sl_dist, balance_usdt * 0.25)

        if pos_usdt < info.min_notional:
            if balance_usdt >= info.min_notional * 2:
                pos_usdt = info.min_notional
                log.info("Position bumped to minimum $%.0f for %s", info.min_notional, sym)
            else:
                return TradeResult(False, sym, signal.signal, 0, signal.price,
                                   mode=self._mode,
                                   error=f"Balance ${balance_usdt:.2f} too low for ${info.min_notional:.0f} min on {sym}")

        qty = self._fmt_qty(pos_usdt / signal.price, info)
        if qty <= 0:
            return TradeResult(False, sym, signal.signal, 0, signal.price,
                               mode=self._mode, error="Qty rounds to zero")

        log.info("AUTO → %s %s mode=%s qty=%s entry=%.6g TP1=%.6g TP2=%.6g TP3=%.6g SL=%.6g",
                 signal.signal, sym, self._mode, qty,
                 signal.price, signal.tp1, signal.tp2, signal.tp3, signal.sl)

        try:
            r = (self._spot(sym, signal, qty, info, risk_usdt, pos_usdt)
                 if self._mode == "spot"
                 else self._futures(futures_sym, signal, qty, info, risk_usdt, pos_usdt))
            if r.success:
                self._daily_trades += 1
            return r
        except Exception as e:
            log.error("Execution error [%s]: %s", sym, e, exc_info=True)
            return TradeResult(False, sym, signal.signal, qty, signal.price,
                               mode=self._mode, error=str(e))

    # ─── protect_open_positions ────────────────────────────────────────


    def cancel_orphan_orders(self) -> list:
        """Cancel all open orders for symbols with no open position."""
        cancelled = []
        try:
            if self._mode != 'futures': return []
            live = {p['symbol'] for p in self.get_positions()
                    if abs(float(p.get('positionAmt', 0))) > 0}
            for o in self.get_open_orders():
                sym = o.get('symbol', '')
                if sym and sym not in live:
                    self._req('DELETE', '/fapi/v1/allOpenOrders', {'symbol': sym})
                    log.info('🧹 Orphan orders cancelled for %s (no position)', sym)
                    cancelled.append(sym)
        except Exception as _e:
            log.debug('Orphan cleanup: %s', _e)
        return list(set(cancelled))

    def protect_open_positions(self, signal_lookup: dict) -> list:
        if self._mode != "futures":
            return []
        protected = []
        try:
            positions   = self.get_positions()
            open_orders = self.get_open_orders() or []
            total_orders = len(open_orders)

            # Duplicate cleanup — each position can have up to 8 orders legitimately
            max_ok = max(len(positions) * 8, 50)   # 6 positions × ~8 orders each = 48
            if total_orders > max_ok and positions:
                log.warning("Duplicate orders (%d > %d) — cancelling all to reset",
                            total_orders, max_ok)
                for _pos in positions:
                    _s = _pos.get("symbol", "")
                    if _s:
                        self._req("DELETE", "/fapi/v1/allOpenOrders", {"symbol": _s})
                        try:
                            algos = self._req("GET", "/fapi/v1/openAlgoOrders", {"symbol": _s}) or {}
                            for ao in (algos.get("algoOrders", []) if isinstance(algos, dict) else []):
                                aid = ao.get("algoId") or ao.get("orderId")
                                if aid:
                                    self._req("DELETE", "/fapi/v1/algoOrder",
                                              {"symbol": _s, "algoId": aid})
                        except Exception:
                            pass
                open_orders = self.get_open_orders() or []
                log.info("After cleanup: %d orders", len(open_orders))

            # Build per-symbol order map
            sym_orders: dict[str, list] = {}
            for o in open_orders:
                sym_orders.setdefault(o.get("symbol", ""), []).append(o)

            # Fetch algo orders per-symbol (more reliable than global fetch)
            # Also do global fetch as backup
            try:
                # Per-symbol fetch for each open position (avoids global propagation lag)
                for _pos_sym in {p.get("symbol","") for p in positions if p.get("symbol")}:
                    try:
                        _sym_algo = self._req("GET", "/fapi/v1/openAlgoOrders",
                                              {"symbol": _pos_sym}) or {}
                        _sym_algo_list = (_sym_algo.get("algoOrders", [])
                                         if isinstance(_sym_algo, dict) else [])
                        for ao in _sym_algo_list:
                            _type = ao.get("orderType") or ao.get("type", "")
                            sym_orders.setdefault(_pos_sym, []).append({
                                "type":    _type,
                                "orderId": ao.get("algoId", ""),
                                "symbol":  _pos_sym,
                                "_is_algo": True,
                            })
                        if _sym_algo_list:
                            log.debug("  %s: %d algo orders merged", _pos_sym, len(_sym_algo_list))
                    except Exception:
                        pass
            except Exception as _ae:
                log.warning("Algo orders fetch FAILED (SL may be missed): %s", _ae)

            SL_TYPES  = {"STOP_MARKET", "STOP", "STOP_LOSS", "STOP_LOSS_LIMIT", "STOP_MARKET_TRAILING", "CONDITIONAL"}
            TSL_TYPES = {"TRAILING_STOP_MARKET"}
            TP_TYPES  = {"TAKE_PROFIT_MARKET", "TAKE_PROFIT", "LIMIT"}

            log.info("protect_open_positions: %d positions, %d open orders",
                     len(positions), len(open_orders))

            # ── Cleanup orphaned orders (orders with no matching position) ──
            # Includes global algo orders (TSL/SL) that outlive closed positions
            active_syms = {p.get("symbol","") for p in positions
                          if abs(float(p.get("positionAmt", 0))) >= 0.0001}
            try:
                _global_algo = self._req("GET", "/fapi/v1/openAlgoOrders", {}) or {}
                _global_algo_list = (_global_algo.get("algoOrders", [])
                                     if isinstance(_global_algo, dict) else [])
                for _gao in _global_algo_list:
                    _gsym = _gao.get("symbol", "")
                    if _gsym and _gsym not in active_syms:
                        _gaid = _gao.get("algoId") or _gao.get("orderId")
                        if _gaid:
                            self._req("DELETE", "/fapi/v1/algoOrder",
                                     {"symbol": _gsym, "algoId": _gaid})
                            log.info("  %s: cancelled orphaned algo order id=%s ✅",
                                     _gsym, _gaid)
            except Exception as _gae:
                log.debug("Global algo order cleanup: %s", _gae)

            for _osym, _oorders in sym_orders.items():
                if _osym and _oorders and _osym not in active_syms:
                    log.info("  %s: orphaned orders (no position) — cancelling %d orders",
                             _osym, len(_oorders))
                    try:
                        self._req("DELETE", "/fapi/v1/allOpenOrders", {"symbol": _osym})
                        _a = self._req("GET", "/fapi/v1/openAlgoOrders", {"symbol": _osym}) or {}
                        for _ao in (_a.get("algoOrders",[]) if isinstance(_a,dict) else []):
                            _aid = _ao.get("algoId") or _ao.get("orderId")
                            if _aid:
                                self._req("DELETE", "/fapi/v1/algoOrder",
                                          {"symbol": _osym, "algoId": _aid})
                        log.info("  %s: orphaned orders cancelled ✅", _osym)
                    except Exception as _oe:
                        log.warning("  %s: orphan cleanup failed: %s", _osym, _oe)

            for pos in positions:
                sym = pos.get("symbol", "")
                amt = float(pos.get("positionAmt", 0))
                if abs(amt) < 0.0001:
                    continue

                orders_sym = sym_orders.get(sym, [])
                has_sl  = any(o.get("type") in SL_TYPES  for o in orders_sym)
                has_tsl = any(o.get("type") in TSL_TYPES for o in orders_sym)
                has_tp  = any(o.get("type") in TP_TYPES  for o in orders_sym)

                log.info("  %s: amt=%.4f orders=%d has_tp=%s has_sl=%s has_tsl=%s",
                         sym, amt, len(orders_sym),
                         "✅" if has_tp  else "❌",
                         "✅" if has_sl  else "❌",
                         "✅" if has_tsl else "❌")

                if has_tp and has_sl and has_tsl:
                    log.info("  %s: fully protected ✅", sym)
                    continue

                is_long    = amt > 0
                close_side = "SELL" if is_long else "BUY"
                qty        = abs(amt)
                entry      = float(pos.get("entryPrice", 0))
                mark       = float(pos.get("markPrice",  entry))
                unreal     = float(pos.get("unRealizedProfit", 0))
                notional   = qty * entry if entry > 0 else 1.0
                pnl_pct    = (unreal / notional * 100) if notional > 0 else 0.0
                info       = self._get_symbol_info(sym)

                sig = signal_lookup.get(sym)
                regime = getattr(sig, "regime", "Trending") if sig else "Trending"
                state  = getattr(sig, "state",  "ACTIVE_INTRADAY") if sig else "ACTIVE_INTRADAY"

                # MACD slope for momentum decay check
                macd_slope = 0.0
                try:
                    klines = self._pub("GET", "/fapi/v1/klines",
                                       {"symbol": sym, "interval": "5m", "limit": 40})
                    if klines:
                        df_ = pd.DataFrame(klines,
                            columns=["open_time","open","high","low","close","volume",
                                     "close_time","qt","tr","bb","bq","ig"])
                        df_ = df_[["close"]].astype(float)
                        macd_slope = self._macd_histogram_slope(df_)
                except Exception: pass

                # State machine decision
                new_state, action = self._state_machine.update_state(
                    state, regime, pnl_pct,
                    getattr(sig, "atr", 0) if sig else 0,
                    macd_slope, mark,
                )

                if action == "EXIT":
                    # Safety: never state-exit a trade that opened < 5 min ago
                    # This prevents immediately killing freshly placed trades
                    try:
                        from dashboard.models import ScalpPosition as _SPE
                        import datetime as _dte
                        _sp_e = _SPE.objects.filter(symbol=sym, closed=False).first()
                        if _sp_e and _sp_e.opened_at:
                            _age_s = (_dte.datetime.now(_dte.timezone.utc) -
                                      _sp_e.opened_at.astimezone(_dte.timezone.utc)).total_seconds()
                            if _age_s < 300:  # < 5 minutes old
                                log.info("  %s STATE-EXIT deferred (%.0fs old < 5min)", sym, _age_s)
                                action = "HOLD"   # skip exit, continue to protection
                    except Exception:
                        pass

                if action == "EXIT":
                    self._req("DELETE", "/fapi/v1/allOpenOrders", {"symbol": sym})
                    self._req("POST", "/fapi/v1/order", {
                        "symbol": sym, "side": close_side,
                        "type": "MARKET", "quantity": qty, "reduceOnly": "true",
                    })
                    log.info("  %s STATE-EXIT | State=%s→%s | PnL=%.2f%%",
                             sym, state, new_state, pnl_pct)
                    continue

                # ── Move SL to breakeven as soon as TP1 is hit ──────────────
                # Detect TP1 hit by: fewer TPs than we started with, OR price
                # is meaningfully in profit (≥0.5% above entry = TP1 territory)
                tp_remaining = [o for o in orders_sym if o.get("type") in TP_TYPES]
                _tp_count = len(tp_remaining)
                _tp1_hit = (
                    _tp_count <= 1                    # TP1+TP2 both filled
                    or (_tp_count <= 2 and pnl_pct >= 0.3)  # TP1 filled from 3-TP setup
                    or pnl_pct >= 0.5                 # definitely in TP1 territory
                )
                # Check if SL is already at or above entry (breakeven already set)
                _existing_sl_p = next(
                    (float(o.get("stopPrice") or o.get("triggerPrice") or 0)
                     for o in orders_sym if o.get("type") in SL_TYPES), 0.0)
                be_already_set = (
                    _existing_sl_p > 0 and entry > 0 and
                    ((_existing_sl_p >= entry * 0.999) if is_long
                     else (_existing_sl_p <= entry * 1.001))
                )
                if _tp1_hit and not be_already_set and entry > 0 and not has_tsl:
                    # Cancel existing SL orders (replace with breakeven stop)
                    for _o in list(orders_sym):
                        if _o.get("type") in SL_TYPES:
                            _oid = _o.get("orderId") or _o.get("algoId")
                            if _oid:
                                try:
                                    self._req("DELETE", "/fapi/v1/algoOrder",
                                              {"symbol": sym, "algoId": _oid})
                                except Exception:
                                    try:
                                        self._req("DELETE", "/fapi/v1/order",
                                                  {"symbol": sym, "orderId": _oid})
                                    except Exception:
                                        pass
                    # Place new SL exactly at entry (breakeven)
                    be_sl_p = self._fp(entry, info)
                    be_qty  = self._fmt_qty(qty, info)
                    if be_sl_p > 0 and be_qty >= info.min_qty:
                        _be_ap = self._algo_params({
                            "symbol": sym, "side": close_side,
                            "type": "STOP_MARKET", "stopPrice": be_sl_p,
                            "quantity": be_qty,
                            "workingType": "MARK_PRICE", "reduceOnly": "true",
                        })
                        _be_r = self._req("POST", "/fapi/v1/algoOrder", _be_ap) or {}
                        if _be_r.get("algoId") or _be_r.get("orderId"):
                            log.info("  %s: TP1 hit → SL moved to breakeven @ %.6g "
                                     "(pnl=%.2f%%) ✅", sym, be_sl_p, pnl_pct)
                        else:
                            log.warning("  %s: breakeven SL placement failed: %s",
                                        sym, _be_r)

                # ── TRAILING TP: move SL to lock profit when trade running well ──
                # When PnL > 0.4%: cancel existing SL + replace with tighter one
                # This prevents giving back gains when position is in profit
                if pnl_pct >= 0.4 and has_sl and entry > 0 and mark > 0:
                    # New SL = lock in 0.2% minimum profit (was at entry or below)
                    trail_lock_pct = 0.002   # lock 0.2% minimum
                    new_sl_p = self._fp(
                        entry * (1 + trail_lock_pct) if is_long
                        else entry * (1 - trail_lock_pct), info
                    )
                    # Only update if new SL is better than existing
                    existing_sl = next(
                        (float(o.get("stopPrice") or 0)
                         for o in orders_sym if o.get("type") in SL_TYPES), 0)
                    should_update = (
                        (is_long  and new_sl_p > existing_sl > 0) or
                        (not is_long and 0 < new_sl_p < existing_sl)
                    )
                    if should_update:
                        try:
                            # Cancel old SL orders
                            for o in orders_sym:
                                if o.get("type") in SL_TYPES:
                                    _oid = o.get("orderId") or o.get("algoId")
                                    if _oid:
                                        self._req("DELETE", "/fapi/v1/algoOrder",
                                                  {"symbol": sym, "algoId": _oid})
                            # Place tighter SL
                            _nsl_ap = self._algo_params({
                                "symbol": sym, "side": close_side,
                                "type": "STOP_MARKET", "stopPrice": new_sl_p,
                                "quantity": self._fmt_qty(qty, info),
                                "workingType": "MARK_PRICE", "reduceOnly": "true",
                            })
                            _nsl_r = self._req("POST", "/fapi/v1/algoOrder", _nsl_ap) or {}
                            if _nsl_r.get("algoId") or _nsl_r.get("orderId"):
                                log.info("  %s: trailing SL moved to %.6g (pnl=%.2f%%) ✅",
                                         sym, new_sl_p, pnl_pct)
                        except Exception as _tsl_e:
                            log.debug("Trailing SL update: %s", _tsl_e)

                # Micro trailing at +0.25% profit (Improvement 3.2)
                # Locks small wins before TP1, prevents reversals eating profit
                if pnl_pct >= 0.25 and not has_tsl and entry > 0 and mark > 0:
                    micro_qty = self._fmt_qty(qty * 0.4, info)
                    if micro_qty >= info.min_qty:
                        micro_act = self._fp(mark * (1.001 if is_long else 0.999), info)
                        m_ap = self._algo_params({
                            "symbol": sym, "side": close_side,
                            "type": "TRAILING_STOP_MARKET",
                            "quantity": micro_qty, "callbackRate": 0.25,
                            "activatePrice": micro_act,
                            "workingType": "MARK_PRICE", "reduceOnly": "true",
                        })
                        mr = self._req("POST", "/fapi/v1/algoOrder", m_ap) or {}
                        if mr.get("algoId") or mr.get("orderId"):
                            log.info("  %s: micro TSL 0.25%% @ pnl=%.2f%% ✅", sym, pnl_pct)

                # Regime-adaptive trailing stop
                if not has_tsl:
                    atr_v = getattr(sig, "atr", entry * 0.015) if sig else entry * 0.015
                    atr_mult = (1.4 if regime == "Strong_Trend_Impulse" else
                                2.1 if regime == "Trending" else
                                1.2)   # tighter in choppy
                    trail_dist = (atr_v or entry * 0.015) * atr_mult
                    act_p      = self._fp(
                        mark - trail_dist if is_long else mark + trail_dist, info)
                    tsl_qty = self._fmt_qty(qty * 0.6, info)
                    if tsl_qty >= info.min_qty and act_p > 0:
                        tsl_r = self._order({
                            "symbol": sym, "side": close_side,
                            "type": "TRAILING_STOP_MARKET",
                            "quantity": tsl_qty, "callbackRate": 0.5,
                            "activationPrice": act_p,
                            "workingType": "MARK_PRICE", "reduceOnly": "true",
                        })
                        if isinstance(tsl_r, dict) and "orderId" in tsl_r:
                            log.info("  %s: TSL placed (%s × %.1f) ✅", sym, regime, atr_mult)

                # Place SL if missing
                # Extra duplicate guard: check if any existing order IS a stop near SL price
                if not has_sl:
                    # Check raw orders for any stop-like order regardless of type name
                    _stop_keywords = ("STOP", "stop", "CONDITIONAL", "conditional")
                    _has_any_stop  = any(
                        any(kw in str(o.get("type","")) for kw in _stop_keywords)
                        for o in orders_sym
                    )
                    if _has_any_stop:
                        log.debug("  %s: stop-like order found in orders — treating as SL present", sym)
                        has_sl = True

                # Grace period: if position was placed in last 120s, SL was just placed
                # Binance algo orders have 3-10s propagation delay → false has_sl=❌
                if not has_sl:
                    import time as _time_gp
                    _placed_at = self._recently_placed.get(sym, 0)
                    if _time_gp.time() - _placed_at < 120:
                        log.info("  %s: SL grace period (placed %.0fs ago) — skipping duplicate SL",
                                 sym, _time_gp.time() - _placed_at)
                        has_sl = True   # assume SL is there, just not propagated yet

                if not has_sl:
                    # Calculate SL from entry if no signal available
                    if sig:
                        sl_p = self._fp(getattr(sig, "sl", 0), info)
                    else:
                        # Default SL: 0.8% from entry
                        sl_p = self._fp(entry * (0.992 if is_long else 1.008), info)

                    if sl_p > 0:
                        sl_q = self._fmt_qty(qty, info)
                        if sl_q >= info.min_qty:
                            # Use algoOrder directly — avoids -4120
                            sl_ap = self._algo_params({
                                "symbol": sym, "side": close_side,
                                "type": "STOP_MARKET", "stopPrice": sl_p,
                                "quantity": sl_q, "workingType": "MARK_PRICE",
                                "reduceOnly": "true",
                            })
                            sl_r = self._req("POST", "/fapi/v1/algoOrder", sl_ap) or {}
                            sl_ok = sl_r.get("algoId") or sl_r.get("orderId") if isinstance(sl_r, dict) else None
                            if sl_ok:
                                log.info("  %s: SL placed @ %.6g id=%s ✅", sym, sl_p, sl_ok)
                            else:
                                # Fallback: STOP_MARKET on regular endpoint
                                sl_r2 = self._req("POST", "/fapi/v1/order", {
                                    "symbol": sym, "side": close_side,
                                    "type": "STOP_MARKET", "stopPrice": sl_p,
                                    "quantity": sl_q, "workingType": "MARK_PRICE",
                                    "reduceOnly": "true",
                                })
                                if isinstance(sl_r2, dict) and "orderId" in sl_r2:
                                    log.info("  %s: SL (fallback) @ %.6g ✅", sym, sl_p)
                                else:
                                    # Final fallback: STOP_LIMIT (supported even without algoOrder)
                                    try:
                                        _lmt_px = round(sl_p * (0.997 if close_side == "SELL" else 1.003), 8)
                                        sl_r3 = self._req("POST", "/fapi/v1/order", {
                                            "symbol": sym, "side": close_side,
                                            "type": "STOP", "quantity": str(round(sl_q, 8)),
                                            "price": str(_lmt_px), "stopPrice": str(sl_p),
                                            "timeInForce": "GTC", "reduceOnly": "true",
                                        })
                                        if isinstance(sl_r3, dict) and "orderId" in sl_r3:
                                            log.info("  %s: SL (STOP_LIMIT last resort) @ %.6g ✅", sym, sl_p)
                                        else:
                                            log.error("  %s: SL FAILED all 3 methods — position runs without SL ⚠️", sym)
                                    except Exception as _sl3_e:
                                        log.error("  %s: SL FAILED all methods: %s ⚠️", sym, _sl3_e)

                # Place TP orders if missing
                if not has_tp and sig:
                    tp_targets = [
                        (getattr(sig, "tp1", 0), qty*0.50, "TP1"),   # 50% at TP1
                        (getattr(sig, "tp2", 0), qty*0.30, "TP2"),   # 30% at TP2
                        (getattr(sig, "tp3", 0), qty*0.20, "TP3"),   # 20% runner
                    ]
                    for tp_price, tp_qty_raw, tp_lbl in tp_targets:
                        tp_p = self._fp(tp_price, info)
                        tp_q = self._fmt_qty(tp_qty_raw, info)
                        if tp_p <= 0 or tp_q < info.min_qty: continue
                        r_ = self._order({
                            "symbol": sym, "side": close_side,
                            "type": "TAKE_PROFIT_MARKET",
                            "stopPrice": tp_p, "quantity": tp_q,
                            "workingType": "MARK_PRICE", "reduceOnly": "true",
                        })
                        if isinstance(r_, dict) and "orderId" in r_:
                            log.info("  %s: %s placed @ %.6g ✅", sym, tp_lbl, tp_p)

                protected.append(sym)
                log.info("  %s: protected ✅ | State=%s→%s | PnL=%.2f%%",
                         sym, state, new_state, pnl_pct)

        except Exception as e:
            log.error("protect_open_positions error: %s", e, exc_info=True)

        # After protect loop: cleanup orphan orders (positions that closed since last cycle)
        try:
            orphans = self.cancel_orphan_orders()
            if orphans:
                log.info("🧹 Orphan orders cleaned up for: %s", orphans)
        except Exception as _oe:
            log.debug("Orphan cleanup in protect: %s", _oe)

        return protected

    # ─── MACD slope helper ──────────────────────────────────────────────

    def _macd_histogram_slope(self, df: pd.DataFrame, period: int = 5) -> float:
        try:
            ema12 = df["close"].ewm(span=12, adjust=False).mean()
            ema26 = df["close"].ewm(span=26, adjust=False).mean()
            macd_ = ema12 - ema26
            sig_  = macd_.ewm(span=9, adjust=False).mean()
            hist  = macd_ - sig_
            if len(hist) < period+1: return 0.0
            return float((hist.iloc[-1] - hist.iloc[-period]) / period)
        except Exception:
            return 0.0

    # ─── SPOT ───────────────────────────────────────────────────────────

    def _spot(self, sym, signal, qty, info, risk_usdt, pos_usdt) -> TradeResult:
        if signal.signal != "BUY":
            return TradeResult(False, sym, signal.signal, qty, signal.price,
                               mode="spot", error="SELL not supported on spot")
        if signal.tp1 <= signal.price:
            return TradeResult(False, sym, signal.signal, qty, signal.price,
                               mode="spot", error="TP1 must be above entry")

        buy = self._req("POST", "/api/v3/order",
                        {"symbol": sym, "side": "BUY", "type": "MARKET", "quantity": qty})
        if not buy or "orderId" not in buy:
            raise Exception(f"Market BUY rejected: {buy}")

        fills      = buy.get("fills", [])
        fill_price = float(fills[0]["price"]) if fills else signal.price
        entry_id   = str(buy["orderId"])

        q_oco = self._fmt_qty(qty*0.40, info)
        oco_id = tp1_id = sl_id = ""
        if q_oco >= info.min_qty:
            oco = self._req("POST", "/api/v3/order/oco", {
                "symbol": sym, "side": "SELL", "quantity": q_oco,
                "price": self._fp(signal.tp1, info),
                "stopPrice": self._fp(signal.sl, info),
                "stopLimitPrice": self._fp(signal.sl*0.998, info),
                "stopLimitTimeInForce": "GTC",
            })
            if oco:
                oco_id = str(oco.get("orderListId", ""))
                orders = oco.get("orders", [])
                tp1_id = str(orders[0]["orderId"]) if orders else ""
                sl_id  = str(orders[1]["orderId"]) if len(orders)>1 else ""

        q_tp2=self._fmt_qty(qty*0.35,info); q_tp3=self._fmt_qty(qty*0.25,info)
        tp2_id=tp3_id=""
        if q_tp2>=info.min_qty:
            r=self._req("POST","/api/v3/order",{"symbol":sym,"side":"SELL","type":"LIMIT",
                "timeInForce":"GTC","quantity":q_tp2,"price":self._fp(signal.tp2,info)})
            tp2_id=str((r or {}).get("orderId",""))
        if q_tp3>=info.min_qty:
            r=self._req("POST","/api/v3/order",{"symbol":sym,"side":"SELL","type":"LIMIT",
                "timeInForce":"GTC","quantity":q_tp3,"price":self._fp(signal.tp3,info)})
            tp3_id=str((r or {}).get("orderId",""))

        log.info("Spot BUY %s qty=%.4f @ %.6g OCO=%s SL=%s TP2=%s TP3=%s",
                 sym, qty, fill_price, oco_id, sl_id, tp2_id, tp3_id)
        return TradeResult(True, sym, "BUY", qty, fill_price, mode="spot",
                           position_usdt=round(pos_usdt,2), risk_usdt=round(risk_usdt,2),
                           entry_order_id=entry_id, oco_id=oco_id,
                           tp1_order_id=tp1_id, tp2_order_id=tp2_id,
                           tp3_order_id=tp3_id, sl_order_id=sl_id)

    # ─── FUTURES ────────────────────────────────────────────────────────

    def _futures(self, sym, signal, qty, info, risk_usdt, pos_usdt) -> TradeResult:
        # sym is already normalized by execute_signal, but normalize again for safety
        sym = self._normalize_futures_sym(sym)
        is_long    = signal.signal == "BUY"
        side       = "BUY" if is_long else "SELL"
        close_side = "SELL" if is_long else "BUY"

        btc_score = getattr(signal, "btc_score", 50)
        regime    = getattr(signal, "regime", "Trending")
        lev = 3 if btc_score >= 62 or regime == "Strong_Trend_Impulse" else 2
        self._req("POST", "/fapi/v1/leverage", {"symbol": sym, "leverage": lev})

        # 70% market entry (immediate) + 30% limit at slight pullback
        # Improvement 5.3: improves average entry price, reduces drawdown
        qty_market = self._fmt_qty(qty * 0.70, info)
        qty_limit  = self._fmt_qty(qty * 0.30, info)
        if qty_limit < info.min_qty:   # too small for limit — go all market
            qty_market = qty; qty_limit = 0

        entry_resp = self._req("POST", "/fapi/v1/order",
                               {"symbol": sym, "side": side, "type": "MARKET",
                                "quantity": qty_market})
        if not entry_resp or "orderId" not in entry_resp:
            raise Exception(f"Futures {side} rejected: {entry_resp}")

        entry_id   = str(entry_resp["orderId"])
        fill_price = float(entry_resp.get("avgPrice", signal.price)) or signal.price

        # Place 30% limit at 0.1% better price
        # Also check notional — Binance requires min $20 per order (LTCUSDT etc.)
        _limit_notional = qty_limit * fill_price if fill_price > 0 else 0
        _min_notional = max(info.min_notional, 20.0)  # Binance futures min $20
        if qty_limit >= info.min_qty and _limit_notional >= _min_notional:
            limit_p = self._fp(fill_price * (0.999 if is_long else 1.001), info)
            lim_r = self._req("POST", "/fapi/v1/order", {
                "symbol": sym, "side": side, "type": "LIMIT",
                "price": limit_p, "quantity": qty_limit,
                "timeInForce": "GTC",
            })
            if isinstance(lim_r, dict) and "orderId" in lim_r:
                log.info("Futures limit entry (30%%) @ %.6g id=%s ✅", limit_p, lim_r["orderId"])
            else:
                log.debug("Limit entry failed — using full market qty")
        elif qty_limit >= info.min_qty:
            # Notional too small for limit order — add to market qty instead
            log.info("Futures limit entry skipped (notional $%.2f < $%.0f min) — all market",
                     _limit_notional, _min_notional)
            qty_limit = 0  # mark as no staged entry

        # TP distribution: 50/30/20 — secure 50% profit early (Improvement 3.3)
        q_tp1=self._fmt_qty(qty*0.50,info)
        q_tp2=self._fmt_qty(qty*0.30,info)
        q_tp3=self._fmt_qty(qty*0.20,info)

        def _place_tp(lbl, stop_p, q):
            """LIMIT order — universal, no -4120 issues."""
            if q < info.min_qty or stop_p <= 0: return ""
            r = self._req("POST", "/fapi/v1/order", {
                "symbol": sym, "side": close_side, "type": "LIMIT",
                "price": stop_p, "quantity": q,
                "timeInForce": "GTC", "reduceOnly": "true",
            })
            if isinstance(r, dict) and "orderId" in r:
                log.info("Futures %s (LIMIT) @ %.6g qty=%s id=%s ✅", lbl, stop_p, q, r["orderId"])
                return str(r["orderId"])
            log.warning("Futures %s failed: %s", lbl, r)
            return ""

        tp1_id = _place_tp("TP1", self._fp(signal.tp1, info), q_tp1)
        tp2_id = _place_tp("TP2", self._fp(signal.tp2, info), q_tp2)
        # TP3: skip when using staged 70/30 entry — 30% limit not yet filled
        # Placing reduceOnly for full qty when only 70% filled = -2022 error
        # TP3 is placed later by protect loop after limit fills
        tp3_id = ""
        if qty_limit < info.min_qty:   # no staged entry = safe to place TP3
            if q_tp3 >= info.min_qty:
                tp3_id = _place_tp("TP3", self._fp(signal.tp3, info), q_tp3)
        else:
            log.debug("TP3 deferred (staged entry pending — protect loop will add later)")

        sl_id = ""
        # Enforce SL hard cap: never more than 1.2% from entry (prevents -5% losses)
        raw_sl_p = self._fp(signal.sl, info)
        max_sl   = fill_price * (0.988 if is_long else 1.012)  # 1.2% max
        sl_p     = max(raw_sl_p, max_sl) if is_long else min(raw_sl_p, max_sl)
        sl_p     = self._fp(sl_p, info)
        if raw_sl_p != sl_p:
            log.info("SL capped: %.5g → %.5g (1.2%% max from %.5g)", raw_sl_p, sl_p, fill_price)
        if sl_p > 0:
            sl_qty = self._fmt_qty(qty, info)
            if sl_qty >= info.min_qty:
                # Try algoOrder first (avoids -4120) then fallback to STOP_MARKET
                sl_ap = self._algo_params({
                    "symbol": sym, "side": close_side,
                    "type": "STOP_MARKET",
                    "stopPrice": sl_p,
                    "quantity": sl_qty,
                    "workingType": "MARK_PRICE",
                    "reduceOnly": "true",
                })
                sl_r = self._req("POST", "/fapi/v1/algoOrder", sl_ap) or {}
                sl_id_key = sl_r.get("algoId") or sl_r.get("orderId") if isinstance(sl_r, dict) else None
                if sl_id_key:
                    sl_id = str(sl_id_key)
                    log.info("Futures SL (algoOrder STOP_MARKET) @ %.6g id=%s ✅", sl_p, sl_id)
                else:
                    # Fallback: standard STOP_MARKET
                    sl_r2 = self._req("POST", "/fapi/v1/order", {
                        "symbol": sym, "side": close_side,
                        "type": "STOP_MARKET", "stopPrice": sl_p,
                        "quantity": sl_qty,
                        "workingType": "MARK_PRICE", "reduceOnly": "true",
                    })
                    if isinstance(sl_r2, dict) and "orderId" in sl_r2:
                        sl_id = str(sl_r2["orderId"])
                        log.info("Futures SL (STOP_MARKET fallback) @ %.6g id=%s ✅", sl_p, sl_id)
                    else:
                        code = sl_r2.get("code", 0) if isinstance(sl_r2, dict) else 0
                        log.warning("Futures SL failed (code=%d) — position runs without SL ⚠️", code)

        # ── Trailing Stop Loss — activates at TP1, trails with 1.0% callback ──
        # Covers 60% of remaining position after TP1 fills
        # Ensures we lock in profit and never give back more than 1%
        tsl_id = ""
        act_p  = self._fp(signal.tp1, info)
        tsl_qty = self._fmt_qty(qty * 0.6, info)
        if act_p > 0 and tsl_qty >= info.min_qty:
            tsl_ap = self._algo_params({
                "symbol":          sym,
                "side":            close_side,
                "type":            "TRAILING_STOP_MARKET",
                "quantity":        tsl_qty,
                "activatePrice":   act_p,     # activates when price hits TP1
                "callbackRate":    1.0,        # trail 1.0% — tight enough to lock profit
                "workingType":     "MARK_PRICE",
                "reduceOnly":      "true",
            })
            tsl_r  = self._req("POST", "/fapi/v1/algoOrder", tsl_ap) or {}
            tsl_key = tsl_r.get("algoId") or tsl_r.get("orderId") if isinstance(tsl_r, dict) else None
            if tsl_key:
                tsl_id = str(tsl_key)
                log.info("Futures TSL @ activation=%.6g callback=1.0%% id=%s ✅", act_p, tsl_id)
            else:
                code = tsl_r.get("code",0) if isinstance(tsl_r,dict) else 0
                log.debug("TSL not placed (code=%d) — trade still protected by TPs+SL", code)

        log.info("Futures %s %s: qty=%.4f @ %.6g lev=%dx id=%s",
                 side, sym, qty, fill_price, lev, entry_id)
        log.info("Futures orders placed: entry=%s tp1=%s tp2=%s tp3=%s sl=%s tsl=%s",
                 entry_id, tp1_id, tp2_id, tp3_id, sl_id, tsl_id)
        # Mark symbol as recently placed — protect loop skips duplicate SL for 120s
        import time as _time_rp
        self._recently_placed[sym] = _time_rp.time()
        return TradeResult(True, sym, side, qty, fill_price, mode="futures", leverage=lev,
                           position_usdt=round(pos_usdt*lev,2), risk_usdt=round(risk_usdt,2),
                           entry_order_id=entry_id, tp1_order_id=tp1_id,
                           tp2_order_id=tp2_id, tp3_order_id=tp3_id, sl_order_id=sl_id)

    # ─── HELPERS ────────────────────────────────────────────────────────

    def _pre_flight(self, balance: float) -> tuple[bool, str]:
        if not self._key or not self._secret: return False, "No API keys"
        if balance <= 0:    return False, "Balance is $0.00"
        if balance < 5:     return False, f"Balance ${balance:.2f} too low"
        if self._daily_loss >= self._loss_limit: return False, "Daily loss limit"
        if self._daily_trades >= self._max_trades: return False, "Daily trade limit"
        return True, ""

    def _algo_params(self, params: dict) -> dict:
        """
        Convert standard /fapi/v1/order params to /fapi/v1/algoOrder format.
        Key differences per Binance API docs:
          - algoType = "CONDITIONAL" is MANDATORY
          - stopPrice  → triggerPrice
          - activationPrice stays as-is (for TRAILING_STOP_MARKET)
          - callbackRate stays as-is
        """
        p = dict(params)
        p["algoType"] = "CONDITIONAL"   # MANDATORY — always CONDITIONAL
        # Rename stopPrice → triggerPrice for algo endpoint
        if "stopPrice" in p and "triggerPrice" not in p:
            p["triggerPrice"] = p.pop("stopPrice")
        # workingType defaults to MARK_PRICE for better accuracy
        if "workingType" not in p:
            p["workingType"] = "MARK_PRICE"
        return p

    def _order(self, params: dict) -> dict:
        """
        Smart order router for futures.

        Routing logic (per Binance API docs):
          TRAILING_STOP_MARKET → /fapi/v1/algoOrder (MANDATORY, algoType=CONDITIONAL)
          TAKE_PROFIT_MARKET   → /fapi/v1/algoOrder (MANDATORY, algoType=CONDITIONAL)
          STOP / STOP_MARKET   → try /fapi/v1/order first
                                  -4120 → fallback to /fapi/v1/algoOrder
                                  still fails → STOP_MARKET on /fapi/v1/order
          SPOT                 → /api/v3/order

        Error handling:
          -1102 = missing algoType (fixed by _algo_params)
          -4120 = wrong endpoint  → retry on algoOrder
          -4045 = not supported   → skip (position protected by TPs)
          -2022 = reduceOnly bad  → retry without reduceOnly
        """
        if self._mode != "futures":
            return self._req("POST", "/api/v3/order", params) or {}

        otype = params.get("type", "")

        # ── Must-use-algo types ────────────────────────────────────────
        if otype in ("TRAILING_STOP_MARKET", "TAKE_PROFIT_MARKET"):
            ap = self._algo_params(params)
            r  = self._req("POST", "/fapi/v1/algoOrder", ap) or {}
            code = r.get("code", 0) if isinstance(r, dict) else 0
            if code == -4045:
                log.debug("%s not supported on this account — skipping", otype)
                return {}
            if code != 0 and "algoId" not in r:
                log.warning("  algoOrder %s failed (code=%d) — position protected by other TPs", otype, code)
                return {}
            return r

        # ── STOP / STOP_MARKET / TAKE_PROFIT ──────────────────────────
        r = self._req("POST", "/fapi/v1/order", params) or {}
        code = r.get("code", 0) if isinstance(r, dict) else 0

        if code == -4120:
            # Binance says use algo endpoint for this type
            log.info("  -4120 on %s → retrying via /fapi/v1/algoOrder", otype)
            ap  = self._algo_params(params)
            r2  = self._req("POST", "/fapi/v1/algoOrder", ap) or {}
            c2  = r2.get("code", 0) if isinstance(r2, dict) else 0
            if "algoId" in r2 or "orderId" in r2:
                log.info("  %s placed via algoOrder ✅", otype)
                return r2
            # Last resort: STOP_MARKET on normal endpoint (no price, just stop)
            if otype in ("STOP", "STOP_LIMIT"):
                p3 = {k: v for k, v in params.items()
                      if k not in ("price", "timeInForce")}
                p3["type"] = "STOP_MARKET"
                r3 = self._req("POST", "/fapi/v1/order", p3) or {}
                if isinstance(r3, dict) and "orderId" in r3:
                    log.info("  SL placed as STOP_MARKET (last resort) ✅")
                    return r3
            log.warning("  %s failed all endpoints — position runs without SL", otype)
            return {}

        if code == -4045:
            log.debug("%s not supported — skipping", otype)
            return {}

        if code == -2022:
            log.info("  -2022 reduceOnly rejected → retrying without flag")
            p2 = {k: v for k, v in params.items() if k != "reduceOnly"}
            r2 = self._req("POST", "/fapi/v1/order", p2) or {}
            if isinstance(r2, dict) and "orderId" in r2:
                return r2

        return r

    def _server_ts(self) -> int:
        try:
            path = "/fapi/v1/time" if self._mode == "futures" else "/api/v3/time"
            r = _requests.get(f"{self._base}{path}", timeout=8)
            if r.ok: return int(r.json().get("serverTime", 0))
        except Exception: pass
        return int(time.time() * 1000)

    def _req(self, method: str, path: str, params: dict) -> dict | None:
        p = {k: v for k, v in params.items() if v is not None}
        p["recvWindow"] = 20000
        for attempt in range(2):
            p["timestamp"] = int(time.time() * 1000) if attempt == 0 else self._server_ts()
            query = urlencode(p)
            sig   = hmac.new(self._secret.encode(), query.encode(), hashlib.sha256).hexdigest()
            url   = f"{self._base}{path}?{query}&signature={sig}"
            try:
                resp = _requests.request(method, url,
                                         headers={"X-MBX-APIKEY": self._key}, timeout=20)
            except _requests.exceptions.Timeout:
                log.warning("Binance %s %s timeout (attempt %d/2)", method, path, attempt+1)
                if attempt == 0: time.sleep(2); continue
                return {"code": -1, "msg": "timeout"}
            except _requests.exceptions.ConnectionError as e:
                log.warning("Binance connection error %s: %s", path, e)
                return {"code": -1, "msg": str(e)[:100]}
            if resp.status_code not in (200, 201):
                log.error("Binance %s %s → %d: %s",
                          method, path, resp.status_code, resp.text[:400])
                try:   body = resp.json()
                except Exception: return {"code": resp.status_code, "msg": resp.text[:200]}
                if body.get("code") == -1021 and attempt == 0:
                    log.info("  -1021 clock drift → retrying with server time")
                    time.sleep(1); continue
                return body
            return resp.json()
        return {"code": -1, "msg": "max retries exceeded"}

    def _pub(self, method: str, path: str, params: dict) -> dict | None:
        try:
            resp = _requests.request(method, f"{self._base}{path}",
                                     params=params, timeout=20)
            return resp.json() if resp.ok else None
        except Exception:
            return None

    def _get_symbol_info(self, sym: str) -> SymbolInfo:
        if sym in self._sym_cache: return self._sym_cache[sym]
        info = SymbolInfo(symbol=sym)
        try:
            endpoint = "/fapi/v1/exchangeInfo" if self._mode == "futures" else "/api/v3/exchangeInfo"
            ex = self._pub("GET", endpoint, {} if self._mode == "futures" else {"symbol": sym})
            for item in (ex or {}).get("symbols", []):
                if item.get("symbol") != sym:
                    continue
                qty_prec   = int(item.get("quantityPrecision", item.get("baseAssetPrecision", 2)))
                price_prec = int(item.get("pricePrecision",    item.get("quotePrecision", 6)))
                min_qty    = 0.001
                min_notional = 5.0
                tick_size    = 0.0001
                step_size    = 0.001

                for f in item.get("filters", []):
                    ft = f.get("filterType", "")
                    if ft == "LOT_SIZE":
                        # step_size tells us precision: 1.0 → 0dp, 0.1 → 1dp, 0.01 → 2dp
                        step = float(f.get("stepSize", "0.001"))
                        step_size = step
                        if step > 0:
                            import math
                            qty_prec = max(0, int(round(-math.log10(step))))
                        min_qty = float(f.get("minQty", "0.001"))
                    elif ft == "PRICE_FILTER":
                        tick = float(f.get("tickSize", "0.0001"))
                        tick_size = tick
                        if tick > 0:
                            import math
                            price_prec = max(0, int(round(-math.log10(tick))))
                    elif ft in ("MIN_NOTIONAL", "NOTIONAL"):
                        min_notional = float(f.get("notional",
                                               f.get("minNotional", "5.0")))

                info = SymbolInfo(
                    symbol          = sym,
                    price_precision = price_prec,
                    qty_precision   = qty_prec,
                    min_qty         = min_qty,
                    min_notional    = min_notional,
                    tick_size       = tick_size,
                    step_size       = step_size,
                )
                log.debug("SymbolInfo %s: qty_prec=%d price_prec=%d step=%.6g tick=%.6g",
                          sym, qty_prec, price_prec, step_size, tick_size)
                break
        except Exception as e:
            log.debug("SymbolInfo error %s: %s", sym, e)
        self._sym_cache[sym] = info
        return info

    def _fp(self, price: float, info: SymbolInfo) -> float:
        """Round price to valid tick size for this symbol."""
        if price <= 0: return 0.0
        tick = getattr(info, "tick_size", 0.0)
        prec = getattr(info, "price_precision", 6)
        if tick > 0:
            import math
            price = round(round(price / tick) * tick, 10)
        return round(price, prec)

    def _fmt_qty(self, qty: float, info: SymbolInfo) -> float:
        """Round qty to valid step size for this symbol."""
        step = getattr(info, "step_size", 0.0)
        prec = getattr(info, "qty_precision", 2)
        if step > 0:
            import math
            qty = math.floor(qty / step) * step   # floor to valid step
        return round(qty, prec)

    def get_balance(self) -> dict:
        result = {"wallet_balance": 0.0, "available_balance": 0.0,
                  "unrealised_pnl": 0.0, "error": ""}
        try:
            if self._mode == "futures":
                for a in (self._req("GET", "/fapi/v2/balance", {}) or []):
                    if a.get("asset") == "USDT":
                        result["wallet_balance"]    = float(a.get("balance", 0))
                        result["available_balance"] = float(a.get("availableBalance", 0))
                        result["unrealised_pnl"]    = float(a.get("crossUnPnl", 0))
                        log.info("Futures wallet=$%.2f available=$%.2f pnl=%+.2f",
                                 result["wallet_balance"], result["available_balance"],
                                 result["unrealised_pnl"])
                        return result
            else:
                for a in (self._req("GET", "/api/v3/account", {}) or {}).get("balances", []):
                    if a["asset"] == "USDT":
                        result["wallet_balance"]    = float(a["free"]) + float(a["locked"])
                        result["available_balance"] = float(a["free"])
                        return result
        except Exception as e:
            result["error"] = str(e)
            log.error("Balance error: %s", e)
        return result

    def get_available_balance(self) -> float:
        return float(self.get_balance().get("available_balance", 0) or 0)

    def get_open_orders(self, symbol: str = None) -> list:
        try:
            params = {"symbol": symbol} if symbol else {}
            path   = "/fapi/v1/openOrders" if self._mode=="futures" else "/api/v3/openOrders"
            result = self._req("GET", path, params)
            orders = result if isinstance(result, list) else []
            if self._mode == "futures":
                algo_resp = self._req("GET", "/fapi/v1/openAlgoOrders", params)
                algo_list = algo_resp.get("algoOrders",[]) if isinstance(algo_resp,dict) else []
                for o in algo_list:
                    if "algoId" in o: o["orderId"] = o["algoId"]
                orders += algo_list
            return orders
        except Exception as e:
            log.error("Open orders error: %s", e); return []

    def get_positions(self) -> list:
        if self._mode != "futures": return []
        try:
            resp = self._req("GET", "/fapi/v2/positionRisk", {})
            if not isinstance(resp, list): return []
            return [p for p in resp if isinstance(p,dict) and float(p.get("positionAmt",0))!=0]
        except Exception as e:
            log.error("Positions error: %s", e); return []

    def cancel_all_orders(self, symbol: str = None) -> dict:
        cancelled, errors = [], []
        try:
            syms = [symbol] if symbol else list({o["symbol"] for o in self.get_open_orders() if "symbol" in o})
            for s in syms:
                try:
                    path = "/fapi/v1/allOpenOrders" if self._mode=="futures" else "/api/v3/openOrders"
                    self._req("DELETE", path, {"symbol": s})
                    cancelled.append(s)
                except Exception as e:
                    errors.append(f"{s}: {e}")
        except Exception as e:
            errors.append(str(e))
        return {"cancelled": cancelled, "errors": errors}

    def reset_daily_counters(self) -> None:
        self._daily_trades = 0; self._daily_loss = 0.0
        log.info("Daily counters reset")