"""
src/analysis/signal_simulator.py — Pre-Trade Backtest Simulation Gate
═══════════════════════════════════════════════════════════════════════
Simulates a trade on the last 120 candles before ANY real money is spent.
Works exactly like a real trade: entry at candle close, TP/SL checked
candle by candle on High/Low, max 3 candles hold (~15 min).

Decision flow:
    Signal detected → simulate on 120 recent candles
        → Win rate < 38%?      → BLOCK
        → Expectancy negative? → BLOCK
        → Momentum against?    → BLOCK
        → TP unreachable?      → BLOCK
        → All pass             → APPROVE + show report

Log output example:
    ┌──────────────────────────────────────────────
    │ 🔬 SIM: ZROUSDT BUY @ 2.004
    │ Entry: 2.004 | TP1: 2.016 | SL: 1.984
    │ Simulated 18 trades on last 120 candles (5m)
    │ ─────────────────────────────────────────────
    │ Trade  1: entered 1.992 → TP1 hit (+0.60%)  ✅
    │ Trade  2: entered 1.998 → SL hit  (-1.00%)  ❌
    │ Trade  3: entered 2.001 → TP1 hit (+0.60%)  ✅
    │ ...
    │ ─────────────────────────────────────────────
    │ Win Rate:   67% (12/18)
    │ Avg Win:   +0.58%  Avg Loss: -0.98%
    │ Expectancy: +0.07% per trade
    │ Virtual P&L: +$34.20 on $1000 virtual wallet
    │ Momentum:  +0.72 (price moving WITH us)
    │ Decision:  ✅ APPROVED — execute real trade
    └──────────────────────────────────────────────
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────
MIN_WIN_RATE         = 0.50   # block if decisive WR < 50% (min 3 decisive required) (target 70% real WR)
MIN_EXPECTANCY       = 0.08   # require at least +0.08% expectancy
MIN_TRADES           = 5      # need at least 5 sim trades
MAX_CONSEC_LOSSES    = 4      # block if last 4 sim trades all SL
MAX_ATR_TP_RATIO     = 1.8    # block if TP1 > 1.8× ATR from entry
VIRTUAL_CAPITAL      = 1000.0 # virtual wallet size
VIRTUAL_RISK         = 0.02   # 2% risk per virtual trade
MAX_HOLD_CANDLES     = 3      # 3 × 5m = 15 min max hold


@dataclass
class VirtualTrade:
    num:          int
    entry_idx:    int
    entry_price:  float
    tp1:          float
    tp2:          float
    sl:           float
    outcome:      str     # TP1 | TP2 | SL | TIMEOUT
    exit_price:   float
    hold_candles: int
    pnl_pct:      float   # e.g. 0.006 = 0.6%


@dataclass
class SimResult:
    symbol:         str
    direction:      str
    entry_price:    float
    tp1:            float
    sl:             float
    n_trades:       int
    wins:           int
    losses:         int
    timeouts:       int
    win_rate:       float
    avg_win:        float
    avg_loss:       float
    expectancy:     float
    virtual_pnl:    float
    consec_losses:  int
    momentum:       float    # -1 to +1
    atr_tp_ratio:   float
    approved:       bool
    reason:         str
    block_reasons:  list
    trades:         list[VirtualTrade] = field(default_factory=list)

    def print_report(self) -> str:
        """Generate a full human-readable simulation report."""
        icon   = "✅ APPROVED" if self.approved else "🚫 BLOCKED"
        border = "─" * 54
        lines  = [
            f"┌{border}",
            f"│ 🔬 PRE-TRADE SIM: {self.symbol} {self.direction} @ {self.entry_price:.5g}",
            f"│ Entry: {self.entry_price:.5g} │ TP1: {self.tp1:.5g} │ SL: {self.sl:.5g}",
            f"│ Simulated {self.n_trades} virtual trades on last 120 candles (5m)",
            f"│{border}",
        ]
        # Show individual trade results (max 10 for readability)
        shown = self.trades[:10]
        for t in shown:
            icon_t = "✅" if t.outcome in ("TP1","TP2") else ("⏱" if t.outcome=="TIMEOUT" else "❌")
            pnl_str = f"{t.pnl_pct*100:+.2f}%"
            lines.append(
                f"│ Trade {t.num:2d}: @ {t.entry_price:.5g} → {t.outcome:7s} "
                f"({pnl_str:>7})  {icon_t}"
            )
        if len(self.trades) > 10:
            lines.append(f"│ ... and {len(self.trades)-10} more trades")
        lines += [
            f"│{border}",
            f"│ Win Rate:    {self.win_rate:.0%} decisive ({self.wins}W/{self.losses}L) | {self.timeouts} timeouts (neutral)",
            f"│ Avg Win:    {self.avg_win*100:+.2f}%   Avg Loss: {self.avg_loss*100:+.2f}%",
            f"│ Expectancy: {self.expectancy*100:+.3f}% per trade",
            f"│ Virtual P&L: ${self.virtual_pnl:+.2f} on ${VIRTUAL_CAPITAL:.0f} virtual wallet",
            f"│ Momentum:   {self.momentum:+.2f}  ({'↑ WITH us' if self.momentum >= 0 else '↓ AGAINST us'})",
            f"│ ATR/TP ratio: {self.atr_tp_ratio:.2f}×  ({'reachable' if self.atr_tp_ratio <= 1.5 else 'far'})",
        ]
        if self.block_reasons:
            lines.append(f"│ Block reasons: {', '.join(self.block_reasons)}")
        lines += [
            f"│ Decision:   {icon}",
            f"└{border}",
        ]
        return "\n".join(lines)


class SignalSimulator:

    def simulate(self, signal, df_5m: pd.DataFrame) -> SimResult:
        sym       = getattr(signal, "symbol",   "???")
        direction = getattr(signal, "signal",   "BUY")
        entry_p   = float(signal.price)
        tp1       = float(signal.tp1)
        tp2       = float(getattr(signal, "tp2", tp1 * 1.01) or tp1)
        sl        = float(signal.sl)
        atr       = float(getattr(signal, "atr", entry_p * 0.008) or entry_p * 0.008)
        is_buy    = direction == "BUY"

        # ── Validate inputs ───────────────────────────────────────────
        if entry_p <= 0 or tp1 <= 0 or sl <= 0:
            return self._skip(sym, direction, entry_p, tp1, sl, "invalid prices")
        if df_5m is None or len(df_5m) < 20:
            return self._skip(sym, direction, entry_p, tp1, sl, "insufficient candle data")

        # ── Use last 120 candles ──────────────────────────────────────
        df      = df_5m.tail(120).reset_index(drop=True)
        highs   = df["high"].astype(float).values
        lows    = df["low"].astype(float).values
        closes  = df["close"].astype(float).values
        n       = len(df)

        # Scale TP/SL distances from current price to each sim entry
        sl_pct  = abs(entry_p - sl)   / entry_p
        tp1_pct = abs(tp1 - entry_p)  / entry_p
        tp2_pct = abs(tp2 - entry_p)  / entry_p

        # ── Run virtual trades every 6 candles (30 min spacing) ───────
        trades:       list[VirtualTrade] = []
        virtual_cash  = VIRTUAL_CAPITAL
        trade_num     = 0

        for i in range(0, n - MAX_HOLD_CANDLES - 1, 6):
            sim_entry = closes[i]
            if sim_entry <= 0:
                continue

            # Scale TP/SL to this entry
            if is_buy:
                sim_tp1 = sim_entry * (1 + tp1_pct)
                sim_tp2 = sim_entry * (1 + tp2_pct)
                sim_sl  = sim_entry * (1 - sl_pct)
            else:
                sim_tp1 = sim_entry * (1 - tp1_pct)
                sim_tp2 = sim_entry * (1 - tp2_pct)
                sim_sl  = sim_entry * (1 + sl_pct)

            # Walk candles forward
            outcome      = "TIMEOUT"
            exit_price   = closes[min(i + MAX_HOLD_CANDLES, n - 1)]
            hold_candles = MAX_HOLD_CANDLES

            for j in range(i + 1, min(i + MAX_HOLD_CANDLES + 1, n)):
                h, l = highs[j], lows[j]
                if is_buy:
                    # Check SL first (conservative — if both hit same candle, SL wins)
                    if l <= sim_sl:
                        outcome = "SL";  exit_price = sim_sl;  hold_candles = j - i; break
                    if h >= sim_tp2:
                        outcome = "TP2"; exit_price = sim_tp2; hold_candles = j - i; break
                    if h >= sim_tp1:
                        outcome = "TP1"; exit_price = sim_tp1; hold_candles = j - i; break
                else:
                    if h >= sim_sl:
                        outcome = "SL";  exit_price = sim_sl;  hold_candles = j - i; break
                    if l <= sim_tp2:
                        outcome = "TP2"; exit_price = sim_tp2; hold_candles = j - i; break
                    if l <= sim_tp1:
                        outcome = "TP1"; exit_price = sim_tp1; hold_candles = j - i; break

            pnl = ((exit_price - sim_entry) / sim_entry if is_buy
                   else (sim_entry - exit_price) / sim_entry)

            # Update virtual wallet
            pos_usdt = (virtual_cash * VIRTUAL_RISK) / max(sl_pct, 0.001)
            virtual_cash += pos_usdt * pnl

            trade_num += 1
            trades.append(VirtualTrade(
                num=trade_num, entry_idx=i, entry_price=sim_entry,
                tp1=sim_tp1, tp2=sim_tp2, sl=sim_sl,
                outcome=outcome, exit_price=exit_price,
                hold_candles=hold_candles, pnl_pct=pnl,
            ))

        if len(trades) < MIN_TRADES:
            return self._skip(sym, direction, entry_p, tp1, sl,
                              f"only {len(trades)} sim trades — too few candles")

        # ── Statistics ────────────────────────────────────────────────
        wins     = [t for t in trades if t.outcome in ("TP1","TP2")]
        losses   = [t for t in trades if t.outcome == "SL"]
        timeouts = [t for t in trades if t.outcome == "TIMEOUT"]
        total    = len(trades)

        # Win rate uses DECISIVE trades only (TP or SL) — not timeouts
        # Timeout = trade expired without hitting either level = neutral
        decisive  = len(wins) + len(losses)
        wr        = len(wins) / decisive if decisive >= 3 else 0.5
        # Full win rate (includes timeouts in denominator) for display
        wr_full   = len(wins) / total if total > 0 else 0.5

        avg_win  = float(np.mean([t.pnl_pct for t in wins]))   if wins   else 0.0
        avg_loss = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0
        # Expectancy based on decisive trades
        expect   = (wr * avg_win) + ((1 - wr) * avg_loss) if decisive >= 3 else 0.0
        vpnl     = virtual_cash - VIRTUAL_CAPITAL

        # Consecutive losses at end
        consec = 0
        for t in reversed(trades[-6:]):
            if t.outcome == "SL": consec += 1
            else: break

        # Momentum: last 5 candle direction vs our trade direction
        last5      = closes[-5:]
        raw_mom    = (last5[-1] - last5[0]) / max(last5[0], 1e-10) * 100
        momentum   = float(np.clip(raw_mom * (1 if is_buy else -1) * 5, -1.0, 1.0))

        # ATR vs TP1 feasibility
        atr_ratio  = float(atr / max(abs(tp1 - entry_p), 1e-10))

        # ── Decision ──────────────────────────────────────────────────
        blocks = []
        # Only block on WR if there are actual SL hits (not just timeouts)
        if decisive >= 3 and wr < MIN_WIN_RATE:
            blocks.append(f"WR={wr:.0%} ({len(wins)}/{decisive} decisive) < {MIN_WIN_RATE:.0%}")
        if decisive >= 3 and expect < MIN_EXPECTANCY:
            blocks.append(f"E={expect*100:+.2f}% negative")
        if consec   >= MAX_CONSEC_LOSSES:
            blocks.append(f"last {consec} sim trades = SL streak")
        if momentum < -0.4:
            blocks.append(f"momentum={momentum:+.2f} against us")
        if atr_ratio > MAX_ATR_TP_RATIO:
            blocks.append(f"TP1 = {atr_ratio:.1f}× ATR (too far)")

        approved = len(blocks) == 0
        reason   = " | ".join(blocks) if blocks else "all checks passed"

        result = SimResult(
            symbol=sym, direction=direction, entry_price=entry_p,
            tp1=tp1, sl=sl, n_trades=total,
            wins=len(wins), losses=len(losses), timeouts=len(timeouts),
            win_rate=wr, avg_win=avg_win, avg_loss=avg_loss,
            expectancy=expect, virtual_pnl=vpnl,
            consec_losses=consec, momentum=momentum,
            atr_tp_ratio=atr_ratio, approved=approved,
            reason=reason, block_reasons=blocks, trades=trades,
        )

        # Always print full report to logs
        for line in result.print_report().split("\n"):
            if self._approved_or_blocked_line(line, approved):
                log.warning(line) if not approved else log.info(line)
            else:
                log.info(line)

        return result

    def _approved_or_blocked_line(self, line: str, approved: bool) -> bool:
        return "Decision:" in line or "Block reasons" in line

    def _skip(self, sym, direction, entry, tp1, sl, reason) -> SimResult:
        log.debug("SIM %s: skipped (%s) — auto-approved", sym, reason)
        return SimResult(
            symbol=sym, direction=direction, entry_price=entry,
            tp1=tp1, sl=sl, n_trades=0, wins=0, losses=0, timeouts=0,
            win_rate=0.5, avg_win=0.0, avg_loss=0.0, expectancy=0.0,
            virtual_pnl=0.0, consec_losses=0, momentum=0.0, atr_tp_ratio=1.0,
            approved=True, reason=reason, block_reasons=[],
        )


_simulator: Optional[SignalSimulator] = None

def get_simulator() -> SignalSimulator:
    global _simulator
    if _simulator is None:
        _simulator = SignalSimulator()
    return _simulator