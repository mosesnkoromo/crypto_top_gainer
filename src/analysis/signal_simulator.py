"""
src/analysis/signal_simulator.py — Institutional Pre-Trade Simulation Gate v4.2
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List
from src.utils.logger import get_logger

log = get_logger(__name__)

MIN_WIN_RATE      = 0.40
MIN_EXPECTANCY    = 0.0001
MIN_TRADES        = 6
MAX_CONSEC_LOSSES = 4
MAX_DRAWDOWN      = 0.30
SPREAD            = 0.0005
SLIPPAGE          = 0.0003
VIRTUAL_CAPITAL   = 500.0
VIRTUAL_RISK_PCT  = 0.02



@dataclass
class VirtualTrade:
    num:          int
    entry_price:  float
    exit_price:   float
    outcome:      str
    pnl_pct:      float
    hold_candles: int
    tp1_hit:      bool = False
    tp2_hit:      bool = False


@dataclass
class SimResult:
    symbol:        str
    direction:     str
    entry_price:   float
    tp1:           float
    sl:            float
    n_trades:      int
    wins:          int
    losses:        int
    timeouts:      int
    win_rate:      float
    avg_win:       float
    avg_loss:      float
    expectancy:    float
    virtual_pnl:   float
    max_drawdown:  float
    consec_losses: int
    momentum:      float
    approved:      bool
    reason:        str
    block_reasons: list
    trades:        list = field(default_factory=list)

    def print_report(self, label: str = "PRE-TRADE SIM") -> str:
        icon   = "✅ APPROVED" if self.approved else "🚫 BLOCKED"
        border = "─" * 54
        lines  = [
            f"┌{border}",
            f"│ 🔬 {label}: {self.symbol} {self.direction} @ {self.entry_price:.5g}",
            f"│ Entry: {self.entry_price:.5g} │ TP1: {self.tp1:.5g} │ SL: {self.sl:.5g}",
            f"│ Simulated {self.n_trades} realistic entries on last 150 candles (5m)",
            f"│{border}",
        ]
        for t in self.trades[:12]:
            icon_t  = "✅" if t.pnl_pct > 0 else "❌"
            outcome = t.outcome[:8]
            lines.append(
                f"│ Trade {t.num:2d}: @ {t.entry_price:.5g}"
                f" → {outcome:8s} ({t.pnl_pct*100:+.2f}%)  {icon_t}"
            )
        if len(self.trades) > 12:
            lines.append(f"│ ... and {len(self.trades)-12} more trades")
        lines += [
            f"│{border}",
            f"│ Win Rate:    {self.win_rate:.0%} ({self.wins}W / {self.losses}L) | {self.timeouts} timeouts",
            f"│ Avg Win:    {self.avg_win*100:+.2f}%   Avg Loss: {self.avg_loss*100:+.2f}%",
            f"│ Expectancy: {self.expectancy*100:+.3f}% per trade",
            f"│ Virtual P&L: ${self.virtual_pnl:+.2f} on ${VIRTUAL_CAPITAL:.0f} | Max DD: {self.max_drawdown*100:.1f}%",
            f"│ Momentum:   {self.momentum:+.2f}",
        ]
        if self.block_reasons:
            lines.append(f"│ Block reasons: {', '.join(self.block_reasons)}")
        lines += [f"│ Decision:   {icon}", f"└{border}"]
        return "\n".join(lines)


class SignalSimulator:

    def simulate(self, signal, df_5m: pd.DataFrame, label: str = "PRE-TRADE SIM") -> SimResult:
        sym       = getattr(signal, "symbol",   "UNKNOWN")
        direction = getattr(signal, "signal",   "BUY")
        entry_p   = float(getattr(signal, "price", 0))
        tp1       = float(getattr(signal, "tp1",   0))
        tp2_raw   = getattr(signal, "tp2", None)
        tp2       = float(tp2_raw) if tp2_raw else tp1 * 1.5
        sl        = float(getattr(signal, "sl",    0))
        is_buy    = direction == "BUY"

        if entry_p <= 0 or tp1 <= 0 or sl <= 0:
            return self._skip(sym, direction, entry_p, tp1, sl, "invalid prices")
        if df_5m is None or len(df_5m) < 60:
            return self._skip_reduced(sym, direction, entry_p, tp1, sl, "insufficient candle data — reduced risk")

        df = df_5m.tail(150).reset_index(drop=True)

        sl_pct  = abs(entry_p - sl)  / entry_p
        tp1_pct = abs(tp1 - entry_p) / entry_p
        tp2_pct = abs(tp2 - entry_p) / entry_p
        if sl_pct < 0.001:
            log.info("  ⏭️  SIM %s: SL too tight (%.4f%%) — skipping sim, auto-approved",
                     sym, sl_pct * 100)
            return self._skip(sym, direction, entry_p, tp1, sl, "SL too tight")

        trades: List[VirtualTrade] = []
        virtual_cash = VIRTUAL_CAPITAL
        trade_num    = 0

        for i in range(40, len(df) - 8):
            if not self._valid_entry_zone(df, i):
                continue
            trade = self._simulate_trade(df, i, is_buy, sl_pct, tp1_pct, tp2_pct)
            if trade is None:
                continue
            trade_num  += 1
            trade.num   = trade_num
            trades.append(trade)
            pos_size     = (virtual_cash * VIRTUAL_RISK_PCT) / max(sl_pct, 0.001)
            virtual_cash += pos_size * trade.pnl_pct

        if len(trades) < MIN_TRADES:
            return self._skip_reduced(sym, direction, entry_p, tp1, sl,
                                     f"only {len(trades)} valid entries — reduced risk")

        wins     = [t for t in trades if t.pnl_pct >  0.0]
        losses   = [t for t in trades if t.pnl_pct <= 0.0 and t.outcome == "SL"]
        timeouts = [t for t in trades if t.outcome in ("TIMEOUT", "RUNNER")]
        decisive = len(wins) + len(losses)

        wr         = len(wins) / max(decisive, 1)
        avg_win    = float(np.mean([t.pnl_pct for t in wins]))   if wins   else 0.0
        avg_loss   = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0
        expectancy = float(np.mean([t.pnl_pct for t in trades]))

        equity = [1.0]; peak = 1.0; max_dd = 0.0
        for t in trades:
            equity.append(equity[-1] * (1 + t.pnl_pct))
            peak   = max(peak, equity[-1])
            dd     = (peak - equity[-1]) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        vpnl   = virtual_cash - VIRTUAL_CAPITAL
        consec = 0
        for t in reversed(trades[-8:]):
            if t.outcome == "SL": consec += 1
            else: break

        last_c   = df["close"].iloc[-20:].astype(float).values
        ema_fast = float(pd.Series(last_c).ewm(span=5,  adjust=False).mean().iloc[-1])
        ema_slow = float(pd.Series(last_c).ewm(span=20, adjust=False).mean().iloc[-1])
        raw_mom  = (ema_fast - ema_slow) / max(ema_slow, 1e-10)
        momentum = float(np.clip(raw_mom * (1 if is_buy else -1) * 10, -1.0, 1.0))

        timeout_wins   = [t for t in trades if t.outcome in ("TIMEOUT","RUNNER") and t.pnl_pct > 0]
        timeout_losses = [t for t in trades if t.outcome in ("TIMEOUT","RUNNER") and t.pnl_pct <= 0]
        timeout_ratio  = len(timeout_wins) + len(timeout_losses)
        if timeout_ratio > 0:
            avg_timeout_pnl = float(np.mean(
                [t.pnl_pct for t in trades if t.outcome in ("TIMEOUT","RUNNER")]
            ))
        else:
            avg_timeout_pnl = 0.0

        recent_highs = df["high"].iloc[-10:].astype(float).values
        recent_lows  = df["low"].iloc[-10:].astype(float).values
        recent_range_pct = (max(recent_highs) - min(recent_lows)) / max(min(recent_lows), 1e-10)

        mc_score = self._monte_carlo(trades)

        quality = 0
        if wr > 0.55:       quality += 1
        if expectancy > 0:  quality += 1
        if momentum > 0:    quality += 1
        if consec < 2:      quality += 1

        _in_cap_mode = len(df_5m) <= 55
        _min_e = -0.0002 if _in_cap_mode else MIN_EXPECTANCY

        _high_edge = (wr >= 0.70 and expectancy > 0.005)
        _is_outlier = (expectancy > 0.015 and wr < 0.50)
        if _high_edge and not _is_outlier:
            approved = True
            reason   = f"HIGH_EDGE_OVERRIDE (WR={wr:.0%} E={expectancy*100:+.2f}%)"
            result = SimResult(
                symbol=sym, direction=direction, entry_price=entry_p,
                tp1=tp1, sl=sl, n_trades=len(trades),
                wins=len(wins), losses=len(losses), timeouts=len(timeouts),
                win_rate=wr, avg_win=avg_win, avg_loss=avg_loss,
                expectancy=expectancy, virtual_pnl=vpnl,
                max_drawdown=max_dd, consec_losses=consec,
                momentum=momentum, approved=True, reason=reason,
                block_reasons=[], trades=trades,
            )
            for line in result.print_report(label=label).split("\n"):
                if "Decision:" in line: log.info(line)
                else: log.info(line)
            return result

        blocks = []
        if _is_outlier:
            blocks.append(f"outlier edge (E={expectancy*100:+.1f}% but WR={wr:.0%})")
        if decisive >= 4 and wr < MIN_WIN_RATE:
            blocks.append(f"WR={wr:.0%} ({len(wins)}/{decisive} decisive) < {MIN_WIN_RATE:.0%}")
        if expectancy < _min_e:
            blocks.append(f"E={expectancy*100:+.3f}% below min {_min_e*100:.2f}%")
        _dd_limit = 0.25 if _in_cap_mode else MAX_DRAWDOWN
        if max_dd > _dd_limit:
            blocks.append(f"max_dd={max_dd*100:.1f}% > {_dd_limit*100:.0f}% limit")
        # if consec >= MAX_CONSEC_LOSSES and not _in_cap_mode:
        #     blocks.append(f"last {consec} sim trades = SL streak")
        if momentum < -0.20:
            blocks.append(f"momentum={momentum:+.2f} against us")
        if recent_range_pct < 0.003:
            blocks.append("dead market — range < 0.3%")
        if timeout_ratio > len(trades) * 0.4 and avg_timeout_pnl < 0:
            blocks.append("timeouts mostly losing")
        if quality < 2:
            blocks.append(f"low quality setup (score={quality}/4)")
        if mc_score < 0.05:
            blocks.append(f"unstable strategy (mc={mc_score:.2f})")

        approved = len(blocks) == 0
        reason   = " | ".join(blocks) if blocks else "all checks passed"

        result = SimResult(
            symbol=sym, direction=direction, entry_price=entry_p,
            tp1=tp1, sl=sl, n_trades=len(trades),
            wins=len(wins), losses=len(losses), timeouts=len(timeouts),
            win_rate=wr, avg_win=avg_win, avg_loss=avg_loss,
            expectancy=expectancy, virtual_pnl=vpnl,
            max_drawdown=max_dd, consec_losses=consec,
            momentum=momentum, approved=approved,
            reason=reason, block_reasons=blocks, trades=trades,
        )

        for line in result.print_report(label=label).split("\n"):
            if "Decision:" in line or "Block reasons" in line:
                log.warning(line) if not approved else log.info(line)
            else:
                log.info(line)

        return result

    def _valid_entry_zone(self, df: pd.DataFrame, i: int) -> bool:
        if i < 40:
            return False
        try:
            closes   = df["close"].iloc[:i+1].astype(float)
            ema9     = float(closes.ewm(span=9,  adjust=False).mean().iloc[-1])
            ema21    = float(closes.ewm(span=21, adjust=False).mean().iloc[-1])
            if abs(ema9 - ema21) < ema21 * 0.0005:
                return False
            vols     = df["volume"].iloc[:i+1].astype(float)
            vol_ma   = float(vols.rolling(20).mean().iloc[-1])
            if vol_ma <= 0:
                return True
            vol_ratio = float(vols.iloc[-1]) / vol_ma
            atr_s    = (df["high"].iloc[:i+1].astype(float) -
                        df["low"].iloc[:i+1].astype(float))
            atr_now  = float(atr_s.rolling(14).mean().iloc[-1])
            atr_mean = float(atr_s.rolling(50).mean().iloc[-1])
            return vol_ratio > 1.1 and atr_now > atr_mean * 0.75
        except Exception:
            return False

    def _adaptive_hold(self, df: pd.DataFrame, i: int) -> int:
        try:
            atr_s    = (df["high"].iloc[:i+1].astype(float) -
                        df["low"].iloc[:i+1].astype(float))
            atr_now  = float(atr_s.rolling(14).mean().iloc[-1])
            atr_mean = float(atr_s.rolling(50).mean().iloc[-1])
            vol_factor = atr_now / max(atr_mean, 1e-10)
            closes   = df["close"].iloc[:i+1].astype(float)
            ema_fast = float(closes.ewm(span=5,  adjust=False).mean().iloc[-1])
            ema_slow = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
            momentum = abs(ema_fast - ema_slow) / max(ema_slow, 1e-10)
            hold = 4
            if vol_factor > 1.2: hold += 2
            if momentum  > 0.002: hold += 1
            if vol_factor < 0.8:  hold -= 1
        except Exception:
            hold = 4
        return int(np.clip(hold, 2, 8))

    def _simulate_trade(self, df, i, is_buy, sl_pct, tp1_pct, tp2_pct):
        sim_entry = float(df["close"].iloc[i])
        sim_entry *= (1 + SPREAD + SLIPPAGE) if is_buy else (1 - SPREAD - SLIPPAGE)

        if is_buy:
            sim_tp1 = sim_entry * (1 + tp1_pct)
            sim_tp2 = sim_entry * (1 + tp2_pct)
            sim_sl  = sim_entry * (1 - sl_pct)
        else:
            sim_tp1 = sim_entry * (1 - tp1_pct)
            sim_tp2 = sim_entry * (1 - tp2_pct)
            sim_sl  = sim_entry * (1 + sl_pct)

        hold_candles = self._adaptive_hold(df, i)
        n            = len(df)

        try:
            atr_s = (df["high"].iloc[:i+1].astype(float) -
                     df["low"].iloc[:i+1].astype(float))
            atr   = float(atr_s.rolling(14).mean().iloc[-1])
        except Exception:
            atr = sim_entry * 0.008

        pos_tp1 = 0.50; pos_tp2 = 0.30; pos_runner = 0.20
        pnl_total = 0.0; tp1_hit = False; tp2_hit = False
        current_sl = sim_sl

        for j in range(i + 1, min(i + hold_candles + 1, n)):
            high  = float(df["high"].iloc[j])
            low   = float(df["low"].iloc[j])

            # When SL is hit
            if is_buy and low <= current_sl:
                remaining = 1.0 - (pos_tp1 if tp1_hit else 0.0) - (pos_tp2 if tp2_hit else 0.0)
                pnl_total += remaining * ((current_sl - sim_entry) / sim_entry)
                outcome = "TP" if (tp1_hit or tp2_hit) else "SL"
                return VirtualTrade(0, sim_entry, current_sl, outcome, pnl_total, j - i, tp1_hit, tp2_hit)
            if not is_buy and high >= current_sl:
                remaining  = 1.0 - (pos_tp1 if tp1_hit else 0.0) - (pos_tp2 if tp2_hit else 0.0)
                pnl_total += remaining * ((sim_entry - current_sl) / sim_entry)
                return VirtualTrade(0, sim_entry, current_sl, "SL", pnl_total, j - i, tp1_hit, tp2_hit)

            if not tp1_hit:
                if (is_buy and high >= sim_tp1) or (not is_buy and low <= sim_tp1):
                    pnl = ((sim_tp1 - sim_entry) / sim_entry) if is_buy else ((sim_entry - sim_tp1) / sim_entry)
                    pnl_total += pos_tp1 * pnl
                    tp1_hit    = True
                    current_sl = sim_entry

            if tp1_hit and not tp2_hit:
                if (is_buy and high >= sim_tp2) or (not is_buy and low <= sim_tp2):
                    pnl = ((sim_tp2 - sim_entry) / sim_entry) if is_buy else ((sim_entry - sim_tp2) / sim_entry)
                    pnl_total += pos_tp2 * pnl
                    tp2_hit    = True

            if tp1_hit:
                close = float(df["close"].iloc[j])
                if is_buy:
                    current_sl = max(current_sl, close - atr)
                else:
                    current_sl = min(current_sl, close + atr)

        exit_idx   = min(i + hold_candles, n - 1)
        exit_price = float(df["close"].iloc[exit_idx])
        pnl_runner = ((exit_price - sim_entry) / sim_entry) if is_buy else ((sim_entry - exit_price) / sim_entry)
        pnl_total += pos_runner * pnl_runner

        outcome = "RUNNER" if (tp1_hit or tp2_hit) else "TIMEOUT"
        return VirtualTrade(0, sim_entry, exit_price, outcome, pnl_total, hold_candles, tp1_hit, tp2_hit)

    def _monte_carlo(self, trades, runs: int = 50) -> float:
        if len(trades) < 15:
            return 0.5
        pnls = np.array([t.pnl_pct for t in trades])
        profitable_runs = 0
        rng = np.random.default_rng(seed=42)
        for _ in range(runs):
            shuffled   = rng.permutation(pnls)
            equity     = 1.0
            for p in shuffled:
                equity *= (1 + p)
            if equity > 1.0:
                profitable_runs += 1
        return profitable_runs / runs

    def _skip(self, sym, direction, entry, tp1, sl, reason) -> SimResult:
        log.info("  ⏭️  SIM %s: skipped (%s) — auto-approved", sym, reason)
        return SimResult(
            symbol=sym, direction=direction, entry_price=entry,
            tp1=tp1, sl=sl, n_trades=0, wins=0, losses=0, timeouts=0,
            win_rate=0.5, avg_win=0.0, avg_loss=0.0, expectancy=0.0,
            virtual_pnl=0.0, max_drawdown=0.0, consec_losses=0,
            momentum=0.0, approved=True, reason=reason,
            block_reasons=[], trades=[],
        )

    def _block(self, sym, direction, entry, tp1, sl, reason) -> SimResult:
        log.warning("  🚫 SIM %s: blocked (%s)", sym, reason)
        return SimResult(
            symbol=sym, direction=direction, entry_price=entry,
            tp1=tp1, sl=sl, n_trades=0, wins=0, losses=0, timeouts=0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0, expectancy=0.0,
            virtual_pnl=0.0, max_drawdown=0.0, consec_losses=0,
            momentum=0.0, approved=False, reason=reason,
            block_reasons=[reason], trades=[],
        )

    def _skip_reduced(self, sym, direction, entry, tp1, sl, reason) -> SimResult:
        log.info("  ⚡ SIM %s: no history — approved at REDUCED RISK (0.5×) | %s", sym, reason)
        return SimResult(
            symbol=sym, direction=direction, entry_price=entry,
            tp1=tp1, sl=sl, n_trades=0, wins=0, losses=0, timeouts=0,
            win_rate=0.5, avg_win=0.0, avg_loss=0.0, expectancy=0.0,
            virtual_pnl=0.0, max_drawdown=0.0, consec_losses=0,
            momentum=0.5,
            approved=True, reason="REDUCED_RISK: " + reason,
            block_reasons=[], trades=[],
        )


_simulator: Optional[SignalSimulator] = None

def get_simulator() -> SignalSimulator:
    global _simulator
    if _simulator is None:
        _simulator = SignalSimulator()
    return _simulator