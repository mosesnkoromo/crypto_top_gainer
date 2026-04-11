"""
src/analysis/trade_state_machine.py
─────────────────────────────────────
Controls trade lifecycle based on regime, momentum decay and P&L.
Duration emerges from market conditions — not fixed time targets.
"""
from __future__ import annotations
from typing import Tuple, Literal

StateType = Literal[
    "SCANNING", "ENTRY_PENDING",
    "ACTIVE_SNIPER",      # 30s–8min  — choppy/range scalp
    "ACTIVE_INTRADAY",    # 5–60min   — trending intraday
    "ACTIVE_STRUCTURAL",  # 1h+       — strong impulse trend
    "EXIT", "CLOSED"
]


class TradeStateMachine:
    """Controls trade lifecycle — duration emerges from regime, momentum and confluence."""

    def get_initial_state(self, regime: str) -> StateType:
        if regime == "Strong_Trend_Impulse":
            return "ACTIVE_STRUCTURAL"
        elif regime == "Trending":
            return "ACTIVE_INTRADAY"
        else:
            return "ACTIVE_SNIPER"

    def update_state(
        self,
        current_state: StateType,
        regime: str,
        pnl_pct: float,
        atr_val: float,
        macd_slope: float,
        current_price: float,
    ) -> Tuple[StateType, str]:
        """
        Returns (new_state, action).
        Actions: HOLD | TRAIL_TIGHT | PARTIAL_CLOSE | EXIT
        """
        if current_state in ("EXIT", "CLOSED"):
            return "CLOSED", "EXIT"

        # ── Momentum decay exit ───────────────────────────────────
        # If profitable but MACD histogram reversing sharply → exit
        if pnl_pct > 0.3 and macd_slope < -0.0008:
            return "EXIT", "EXIT"
        # If in loss and MACD also against us → cut fast
        if pnl_pct < 0 and macd_slope > 0.0008:
            return "EXIT", "EXIT"

        # ── Choppy regime: exit quickly if not moving ─────────────
        if regime == "Choppy_Range" and pnl_pct < 0.3:
            return "EXIT", "EXIT"

        # ── SNIPER state (5m scalp) ───────────────────────────────
        if current_state == "ACTIVE_SNIPER":
            if pnl_pct >= 1.5 or regime in ("Trending", "Strong_Trend_Impulse"):
                return "ACTIVE_INTRADAY", "TRAIL_TIGHT"
            if pnl_pct < 0.1:
                return "EXIT", "EXIT"
            return "ACTIVE_SNIPER", "HOLD"

        # ── INTRADAY state ────────────────────────────────────────
        if current_state == "ACTIVE_INTRADAY":
            if regime == "Strong_Trend_Impulse" and pnl_pct > 2.5:
                return "ACTIVE_STRUCTURAL", "TRAIL_TIGHT"
            if pnl_pct > 1.5:
                return "ACTIVE_INTRADAY", "PARTIAL_CLOSE"
            return "ACTIVE_INTRADAY", "HOLD"

        # ── STRUCTURAL state ──────────────────────────────────────
        if current_state == "ACTIVE_STRUCTURAL":
            if pnl_pct > 4.0:
                return "ACTIVE_STRUCTURAL", "PARTIAL_CLOSE"
            return "ACTIVE_STRUCTURAL", "HOLD"

        # ── Strong profit protection ──────────────────────────────
        if pnl_pct > 3.0:
            return current_state, "PARTIAL_CLOSE"

        return current_state, "HOLD"