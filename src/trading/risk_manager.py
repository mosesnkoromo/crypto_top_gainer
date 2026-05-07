"""
src/trading/risk_manager.py
────────────────────────────
Post-entry risk enforcement layer — sits BETWEEN BinanceTrader and the open
position book. Where risk_filter.py prevents bad trades from being opened,
risk_manager.py prevents good trades from going bad.

Three enforcement layers (run every scan cycle, after protect_open_positions):

  L1  Hard loss cap     Force-close any position whose unrealised PnL falls
                        below -{ABSOLUTE_LOSS_CAP_PCT}% of wallet equity.
                        Catches: missing SL, SL placed too far, slippage,
                        slow bleed past the configured stop.

  L2  Time stop         Force-close LOSING positions held longer than
                        {MAX_HOLD_HOURS}h. Winners are left to run.
                        Catches: slow bleeders that sit between TP1-BE and
                        SL for days (the GIGGLE/BANANAS31/WIF pattern).

  L3  Daily DD halt     Block NEW trades for the rest of the UTC day if
                        drawdown from session-start equity exceeds
                        {DAILY_DRAWDOWN_HALT_PCT}%. Existing positions
                        remain managed. Resets at next UTC midnight.

Plus diagnostics-only:
  • Notional audit — warn on positions exceeding {NOTIONAL_CAP_WARN_PCT}% of
    equity (not enforced — would fight ongoing trades).

Failsafe philosophy: this layer NEVER raises. Failures return a structured
RiskAction so the scanner can log+continue. The trader is borrowed (not
owned) so this layer has no auth concerns of its own.

Tunable thresholds at top of file. Re-evaluate every 30 days against the
latest Binance Position History export.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# Tunable thresholds
# ─────────────────────────────────────────────────────────────────

# L1 — Per-trade absolute loss ceiling (% of wallet equity).
# Set conservatively: even at 2% configured risk per trade, no single
# trade should ever exceed 1% realised loss.  This is the seatbelt.
ABSOLUTE_LOSS_CAP_PCT = 1.0

# L2 — Maximum hold time for LOSING positions only.
# 72h covers the natural swing horizon (TP3 typically hits within 48h);
# anything still negative after 3 days is almost certainly a slow bleeder.
# Winning positions are NEVER force-closed by time.
MAX_HOLD_HOURS = 72

# L3 — Daily drawdown halt (% from session-start equity).
# Once breached, no new trades until the next UTC midnight reset.
DAILY_DRAWDOWN_HALT_PCT = 5.0

# Diagnostic only — log warning, no enforcement
NOTIONAL_CAP_WARN_PCT = 30.0


# ─────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────

@dataclass
class RiskAction:
    layer:        str          # "L1_loss_cap" | "L2_time_stop"
    symbol:       str
    reason:       str
    success:      bool
    pnl_usdt:     float = 0.0
    hold_hours:   float = 0.0


@dataclass
class CycleResult:
    actions:           list           # list[RiskAction]
    halted:            bool = False   # L3 daily DD halt active
    equity:            float = 0.0
    session_start:     float = 0.0
    drawdown_pct:      float = 0.0


# ─────────────────────────────────────────────────────────────────
# RiskManager
# ─────────────────────────────────────────────────────────────────

class RiskManager:
    """
    Stateful across cycles only for the L3 daily-drawdown halt.
    L1 and L2 are stateless and idempotent — safe to call every cycle.

    Borrows a futures-mode BinanceTrader; never instantiates one.
    Cache one instance per Scanner; re-bind `._trader` each cycle when
    the scanner rebuilds its trader (see scanner.py integration patch).
    """

    def __init__(self,
                 trader,
                 loss_cap_pct:      float = ABSOLUTE_LOSS_CAP_PCT,
                 max_hold_hours:    float = MAX_HOLD_HOURS,
                 daily_dd_halt_pct: float = DAILY_DRAWDOWN_HALT_PCT):
        self._trader      = trader
        self._loss_cap    = loss_cap_pct
        self._max_hold    = max_hold_hours
        self._dd_halt_pct = daily_dd_halt_pct

        # Session state (resets at UTC midnight)
        self._session_start_equity: Optional[float] = None
        self._session_date:         Optional[str]   = None
        self._dd_halted:            bool            = False

    # ── Master entry point ──────────────────────────────────────

    def enforce_all(self) -> CycleResult:
        """
        Run all enforcement layers. Idempotent and failsafe — call from
        Scanner.run_cycle() AFTER protect_open_positions and BEFORE the
        next signal scan.
        """
        result = CycleResult(actions=[])

        # Only manage leveraged positions. Spot OCO orders are managed
        # natively by Binance and don't need this layer.
        if getattr(self._trader, "_mode", "") != "futures":
            log.debug("RiskManager: skipping (trader mode=%s, not 'futures')",
                      getattr(self._trader, "_mode", "?"))
            return result

        try:
            bal = self._trader.get_balance() or {}
            equity = float(bal.get("wallet_balance", 0) or 0)
            result.equity = equity

            if equity <= 0:
                log.warning("RiskManager: equity=$0, skipping enforcement")
                return result

            # Reset session state if UTC date rolled over
            self._maybe_reset_session(equity)
            result.session_start = self._session_start_equity or 0

            # L3 — drawdown halt (sets the flag; doesn't close positions)
            self._check_daily_drawdown(equity, result)

            # L1 + L2 per-position checks
            positions = self._trader.get_positions() or []
            for pos in positions:
                action = self._evaluate_position(pos, equity)
                if action is not None:
                    result.actions.append(action)

            # Diagnostic-only: notional audit
            self._audit_notionals(positions, equity)

        except Exception as e:
            log.error("RiskManager.enforce_all error: %s", e, exc_info=True)

        return result

    # ── L1 + L2: per-position evaluation ────────────────────────

    def _evaluate_position(self, pos: dict, equity: float) -> Optional[RiskAction]:
        sym = pos.get("symbol", "")
        try:
            amt = float(pos.get("positionAmt", 0))
        except (TypeError, ValueError):
            return None
        if amt == 0 or not sym:
            return None

        try:
            upnl      = float(pos.get("unRealizedProfit", 0))
            update_ms = int(pos.get("updateTime", 0))
        except (TypeError, ValueError):
            return None

        # ── L1 — Hard loss cap ─────────────────────────────────
        cap_usd = equity * (self._loss_cap / 100.0)
        if upnl < -cap_usd:
            ok = self._force_close_market(sym, amt)
            log.warning("⛔ L1 LOSS-CAP %s uPnL=$%+.2f exceeds -$%.2f cap "
                        "(%.1f%% of equity) → %s",
                        sym, upnl, cap_usd, self._loss_cap,
                        "closed" if ok else "FAILED")
            return RiskAction(
                layer="L1_loss_cap", symbol=sym,
                reason=f"uPnL ${upnl:+.2f} breached ${cap_usd:.2f} cap",
                success=ok, pnl_usdt=upnl,
            )

        # ── L2 — Time stop (LOSING positions only) ─────────────
        if update_ms > 0 and upnl < 0:
            age_hours = (time.time() - update_ms / 1000.0) / 3600.0
            if age_hours > self._max_hold:
                ok = self._force_close_market(sym, amt)
                log.warning("⏰ L2 TIME-STOP %s held %.1fh > %.0fh "
                            "(uPnL=$%+.2f) → %s",
                            sym, age_hours, self._max_hold, upnl,
                            "closed" if ok else "FAILED")
                return RiskAction(
                    layer="L2_time_stop", symbol=sym,
                    reason=f"losing position held {age_hours:.1f}h "
                           f"> {self._max_hold}h",
                    success=ok, pnl_usdt=upnl, hold_hours=age_hours,
                )

        return None

    # ── L3: Daily drawdown halt ─────────────────────────────────

    def _maybe_reset_session(self, current_equity: float) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._session_date != today:
            old_date = self._session_date
            self._session_date         = today
            self._session_start_equity = current_equity
            self._dd_halted            = False
            if old_date is not None:
                log.info("RiskManager session reset: %s → %s @ equity=$%.2f",
                         old_date, today, current_equity)
            else:
                log.info("RiskManager session start: %s @ equity=$%.2f",
                         today, current_equity)

    def _check_daily_drawdown(self, equity: float, result: CycleResult) -> None:
        if not self._session_start_equity or self._session_start_equity <= 0:
            return
        dd_usd = self._session_start_equity - equity
        dd_pct = dd_usd / self._session_start_equity * 100.0
        result.drawdown_pct = dd_pct

        if dd_pct >= self._dd_halt_pct and not self._dd_halted:
            self._dd_halted = True
            log.warning("🛑 L3 DAILY-DD: -$%.2f (-%.2f%%) ≥ %.1f%% threshold "
                        "— blocking NEW trades until next UTC midnight "
                        "(existing positions still managed)",
                        dd_usd, dd_pct, self._dd_halt_pct)

        result.halted = self._dd_halted

    def is_trading_halted(self) -> bool:
        """Public — Scanner._run_auto_trade checks this before opening trades."""
        return self._dd_halted

    # ── Diagnostic: notional audit ──────────────────────────────

    def _audit_notionals(self, positions: list, equity: float) -> None:
        warn_thr = equity * (NOTIONAL_CAP_WARN_PCT / 100.0)
        for pos in positions:
            try:
                amt = abs(float(pos.get("positionAmt", 0)))
                ent = float(pos.get("entryPrice", 0))
            except (TypeError, ValueError):
                continue
            if amt == 0:
                continue
            notional = amt * ent
            if notional > warn_thr:
                pct = notional / equity * 100.0 if equity > 0 else 0
                log.warning("📐 NOTIONAL-AUDIT %s: $%.2f = %.1f%% of equity "
                            "(warn>%.0f%%) — investigate sizing",
                            pos.get("symbol", "?"), notional, pct,
                            NOTIONAL_CAP_WARN_PCT)

    # ── Force-close helper ──────────────────────────────────────

    def _force_close_market(self, symbol: str, position_amt: float) -> bool:
        """
        Cancel any pending TP/SL/TSL orders, then send a reduceOnly market
        order in the opposite direction of the position. Returns True if
        the close was confirmed (or if the position was already closed).

        Uses the trader's existing _req() so HMAC signing and base URL
        match the rest of the BinanceTrader code path.
        """
        try:
            # 1. Cancel pending orders so they don't fire after we close
            try:
                self._trader.cancel_all_orders(symbol)
            except Exception as e:
                log.debug("RiskManager: cancel_all_orders(%s) failed: %s",
                          symbol, e)

            # 2. Send the closing market order
            close_side = "SELL" if position_amt > 0 else "BUY"
            qty        = abs(position_amt)

            r = self._trader._req("POST", "/fapi/v1/order", {
                "symbol":     symbol,
                "side":       close_side,
                "type":       "MARKET",
                "quantity":   qty,
                "reduceOnly": "true",
            })

            if isinstance(r, dict) and "orderId" in r:
                log.info("RiskManager force-close: %s %s qty=%s id=%s",
                         close_side, symbol, qty, r["orderId"])
                return True

            # -2022 = ReduceOnly Order Failed (position already 0) — treat as OK
            code = r.get("code", 0) if isinstance(r, dict) else 0
            if code == -2022:
                log.info("RiskManager force-close: %s already closed", symbol)
                return True

            log.error("RiskManager force-close FAILED %s: %s", symbol, r)
            return False
        except Exception as e:
            log.error("RiskManager force-close exception %s: %s", symbol, e)
            return False