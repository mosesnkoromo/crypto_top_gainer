"""
src/analysis/backtester.py
───────────────────────────
Historical strategy backtester.

Fetches real Binance OHLCV data, runs the signal engine logic on each
candle as if trading live, and reports full performance statistics.

Process:
  1. Fetch 6 months of daily + 4H + 1H candles per symbol
  2. Walk forward candle by candle (no look-ahead bias)
  3. Apply signal engine rules at each candle
  4. Simulate trade outcomes: TP1/TP2/TP3/SL hit detection
  5. Track capital growth, win rates, drawdown
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Literal

from src.analysis.indicators import (
    atr, bollinger_bands, ema_value, ema,
    obv_trend, rsi, stochastic_rsi, volume_ratio, williams_r,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

_BINANCE_ENDPOINTS = [
    "https://api.binance.com/api/v3",
    "https://api1.binance.com/api/v3",
    "https://api2.binance.com/api/v3",
    "https://api3.binance.com/api/v3",
]

# Strategy constants (mirrors signal_engine.py)
MIN_TREND_SLOPE    = 0.002
BUY_BTC_MIN        = 50
SELL_BTC_MAX       = 55
ULTRA_THRESHOLD    = 7.0
STRONG_THRESHOLD   = 5.0


@dataclass
class BacktestTrade:
    symbol:      str
    direction:   str        # BUY | SELL
    grade:       str
    entry_price: float
    tp1:         float
    tp2:         float
    tp3:         float
    sl:          float
    entry_time:  datetime
    exit_time:   datetime | None = None
    outcome:     str = "PENDING"   # TP1|TP2|TP3|SL|TIMEOUT
    exit_price:  float = 0.0
    pnl_pct:     float = 0.0
    confluence:  float = 0.0
    rsi_entry:   float = 0.0
    btc_score:   int = 50


@dataclass
class BacktestResult:
    symbol:         str
    period_days:    int
    total_trades:   int = 0
    wins:           int = 0
    losses:         int = 0
    timeouts:       int = 0
    win_rate:       float = 0.0
    avg_win_pct:    float = 0.0
    avg_loss_pct:   float = 0.0
    total_pnl_pct:  float = 0.0
    max_drawdown:   float = 0.0
    profit_factor:  float = 0.0
    best_trade:     float = 0.0
    worst_trade:    float = 0.0
    trades:         list = field(default_factory=list)
    equity_curve:   list = field(default_factory=list)

    # Grade breakdown
    ultra_wr:    float = 0.0
    strong_wr:   float = 0.0


class Backtester:

    def __init__(self,
                 tp1_pct:  float = 3.0,
                 tp2_pct:  float = 6.0,
                 tp3_pct:  float = 10.0,
                 sl_pct:   float = 3.0,
                 tp1_close: float = 0.33,
                 tp2_close: float = 0.33,
                 tp3_close: float = 0.34,
                 min_confluence: float = 5.0):
        self.tp1_pct       = tp1_pct
        self.tp2_pct       = tp2_pct
        self.tp3_pct       = tp3_pct
        self.sl_pct        = sl_pct
        self.tp1_close     = tp1_close
        self.tp2_close     = tp2_close
        self.tp3_close     = tp3_close
        self.min_confluence = min_confluence
        self._session      = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ── Public ────────────────────────────────────────────────

    def run(self, symbol: str, days: int = 180,
            progress_cb=None) -> BacktestResult:
        """
        Run backtest for one symbol over `days` of history.
        progress_cb(pct, message) called periodically.
        """
        log.info("Backtest: %s over %d days", symbol, days)
        result = BacktestResult(symbol=symbol, period_days=days)

        def prog(pct, msg):
            if progress_cb:
                progress_cb(pct, msg)

        prog(5, "Fetching daily candles...")
        df_1d = self._fetch(symbol, "1d", min(days + 50, 500))
        if df_1d is None or len(df_1d) < 30:
            log.warning("Insufficient daily data for %s", symbol)
            return result

        prog(20, "Fetching 4H candles...")
        df_4h = self._fetch(symbol, "4h", min(days * 6 + 100, 1000))
        if df_4h is None or len(df_4h) < 50:
            return result

        prog(35, "Fetching 1H candles...")
        df_1h = self._fetch(symbol, "1h", min(days * 24 + 100, 1500))
        if df_1h is None:
            df_1h = df_4h.copy()

        prog(50, "Running signal simulation...")

        # Cut to requested period
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_ms = int(cutoff.timestamp() * 1000)
        df_1d = df_1d[df_1d["open_time"] >= cutoff_ms].reset_index(drop=True)
        df_4h = df_4h[df_4h["open_time"] >= cutoff_ms].reset_index(drop=True)
        df_1h = df_1h[df_1h["open_time"] >= cutoff_ms].reset_index(drop=True)

        trades = self._simulate(symbol, df_1d, df_4h, df_1h, prog)
        result.trades = [t.__dict__ for t in trades]

        prog(90, "Computing statistics...")
        self._compute_stats(result, trades)
        prog(100, "Done")
        return result

    def run_multi(self, symbols: list[str], days: int = 90,
                  progress_cb=None) -> dict:
        """Run backtest across multiple symbols, return aggregated results."""
        all_results = {}
        for i, sym in enumerate(symbols):
            pct_base = int(i / len(symbols) * 90)
            def prog(p, m, base=pct_base, total=len(symbols)):
                if progress_cb:
                    progress_cb(base + int(p * 0.9 / total), f"{sym}: {m}")
            all_results[sym] = self.run(sym, days, progress_cb=prog)
            time.sleep(0.3)  # rate limit

        return all_results

    # ── Simulation ────────────────────────────────────────────

    def _simulate(self, symbol: str,
                  df_1d: pd.DataFrame,
                  df_4h: pd.DataFrame,
                  df_1h: pd.DataFrame,
                  prog) -> list[BacktestTrade]:
        """
        Walk-forward simulation. For each 4H candle:
          1. Check if any open trade hit TP or SL
          2. Check for new signal using data UP TO this candle
          3. No look-ahead: only use candles[0:i] for indicators
        """
        trades: list[BacktestTrade] = []
        open_trade: BacktestTrade | None = None
        total = len(df_4h)

        for i in range(50, total):
            if i % 100 == 0:
                prog(50 + int(i / total * 35), f"Simulating candle {i}/{total}")

            candle_4h   = df_4h.iloc[i]
            candle_time = datetime.fromtimestamp(
                int(candle_4h["open_time"]) / 1000, tz=timezone.utc
            )
            high  = float(candle_4h["high"])
            low   = float(candle_4h["low"])
            close = float(candle_4h["close"])

            # ── Check open trade ──────────────────────────────
            if open_trade is not None:
                outcome = self._check_exit(open_trade, high, low, candle_time)
                if outcome:
                    open_trade.outcome    = outcome
                    open_trade.exit_time  = candle_time
                    if open_trade.direction == "BUY":
                        if outcome == "TP1": open_trade.exit_price = open_trade.tp1
                        elif outcome == "TP2": open_trade.exit_price = open_trade.tp2
                        elif outcome == "TP3": open_trade.exit_price = open_trade.tp3
                        else: open_trade.exit_price = open_trade.sl
                    else:
                        if outcome == "TP1": open_trade.exit_price = open_trade.tp1
                        elif outcome == "TP2": open_trade.exit_price = open_trade.tp2
                        elif outcome == "TP3": open_trade.exit_price = open_trade.tp3
                        else: open_trade.exit_price = open_trade.sl
                    open_trade.pnl_pct = self._calc_pnl(open_trade)
                    trades.append(open_trade)
                    open_trade = None
                    continue

                # Timeout after 7 days
                if (candle_time - open_trade.entry_time).days >= 7:
                    open_trade.outcome    = "TIMEOUT"
                    open_trade.exit_time  = candle_time
                    open_trade.exit_price = close
                    open_trade.pnl_pct    = self._calc_pnl_price(open_trade, close)
                    trades.append(open_trade)
                    open_trade = None

            # ── Check for new signal (only when flat) ─────────
            if open_trade is not None:
                continue

            # Slice history up to current candle (no look-ahead)
            df_4h_hist = df_4h.iloc[max(0, i-100):i+1]
            df_1d_hist = df_1d[df_1d["open_time"] <= candle_4h["open_time"]].tail(100)
            df_1h_hist = df_1h[df_1h["open_time"] <= candle_4h["open_time"]].tail(60)

            if len(df_1d_hist) < 20 or len(df_4h_hist) < 20:
                continue

            sig = self._detect_signal(symbol, close, df_1d_hist, df_4h_hist, df_1h_hist)
            if sig:
                direction, confluence, grade, rsi_val, btc_score = sig
                entry = close
                if direction == "BUY":
                    tp1 = round(entry * (1 + self.tp1_pct / 100), 8)
                    tp2 = round(entry * (1 + self.tp2_pct / 100), 8)
                    tp3 = round(entry * (1 + self.tp3_pct / 100), 8)
                    sl  = round(entry * (1 - self.sl_pct  / 100), 8)
                else:
                    tp1 = round(entry * (1 - self.tp1_pct / 100), 8)
                    tp2 = round(entry * (1 - self.tp2_pct / 100), 8)
                    tp3 = round(entry * (1 - self.tp3_pct / 100), 8)
                    sl  = round(entry * (1 + self.sl_pct  / 100), 8)

                open_trade = BacktestTrade(
                    symbol=symbol, direction=direction, grade=grade,
                    entry_price=entry, tp1=tp1, tp2=tp2, tp3=tp3, sl=sl,
                    entry_time=candle_time, confluence=confluence,
                    rsi_entry=rsi_val, btc_score=btc_score,
                )

        # Close any open trade at end
        if open_trade and len(df_4h) > 0:
            last = float(df_4h.iloc[-1]["close"])
            open_trade.outcome    = "TIMEOUT"
            open_trade.exit_time  = datetime.now(timezone.utc)
            open_trade.exit_price = last
            open_trade.pnl_pct    = self._calc_pnl_price(open_trade, last)
            trades.append(open_trade)

        return trades

    def _detect_signal(self, symbol, price, df_1d, df_4h, df_1h) -> tuple | None:
        """Simplified signal detection for backtesting speed."""
        try:
            c1d = df_1d["close"]
            c4h = df_4h["close"]
            c1h = df_1h["close"] if len(df_1h) >= 10 else c4h

            rsi_d  = rsi(c1d)
            rsi_4h = rsi(c4h)
            rsi_1h = rsi(c1h)
            e20_d  = ema_value(c1d, 20)
            e50_d  = ema_value(c1d, 50)
            e20_4h = ema_value(c4h, 20)
            e50_4h = ema_value(c4h, 50)
            vr_4h  = volume_ratio(df_4h["volume"], 20)
            atr_4h_val = atr(df_4h, 14)
            obv    = obv_trend(df_4h)

            ema20_d_series = ema(c1d, 20)
            slope_d = self._slope(ema20_d_series, 5)

            ema20_4h_series = ema(c4h, 20)
            slope_4h = self._slope(ema20_4h_series, 5)

            # BTC score approximation from BTC RSI + EMA structure
            btc_score = 55  # neutral default in backtest (real bot uses live BTC)

            # ── BUY check ─────────────────────────────────────
            buy_score = 0.0
            if (price > e50_d and slope_d > MIN_TREND_SLOPE):
                if price > e20_d > e50_d: buy_score += 2.5
                elif price > e50_d:       buy_score += 1.0

                if 45 <= rsi_d <= 65:     buy_score += 1.2
                elif 35 <= rsi_d < 45:    buy_score += 0.8

                if slope_d > 0.008:       buy_score += 1.0
                elif slope_d > MIN_TREND_SLOPE: buy_score += 0.5

                if price > e20_4h > e50_4h and slope_4h > MIN_TREND_SLOPE:
                    buy_score += 1.5
                elif price > e20_4h: buy_score += 0.8

                if 40 <= rsi_4h <= 58:    buy_score += 1.0
                elif 35 <= rsi_4h < 40:   buy_score += 0.6

                if vr_4h < 0.75:          buy_score += 1.0
                elif vr_4h < 1.0:         buy_score += 0.5

                if obv == "rising":       buy_score += 0.8
                if 38 <= rsi_1h <= 60:    buy_score += 0.8

            # ── SELL check ────────────────────────────────────
            sell_score = 0.0
            if (price < e50_d and slope_d < -MIN_TREND_SLOPE):
                if price < e20_d < e50_d: sell_score += 2.5
                elif price < e50_d:       sell_score += 1.0

                if 48 <= rsi_d <= 62:     sell_score += 1.2
                if slope_d < -0.008:      sell_score += 1.0
                elif slope_d < -MIN_TREND_SLOPE: sell_score += 0.5

                if price < e20_4h < e50_4h and slope_4h < -MIN_TREND_SLOPE:
                    sell_score += 1.5
                elif price < e20_4h: sell_score += 0.8

                if 48 <= rsi_4h <= 65:    sell_score += 1.0
                if vr_4h >= 1.5:          sell_score += 1.0
                elif vr_4h >= 1.1:        sell_score += 0.5
                if obv == "falling":      sell_score += 0.8
                if rsi_1h >= 60:          sell_score += 0.8

            best_score = max(buy_score, sell_score)
            if best_score < self.min_confluence:
                return None

            direction = "BUY" if buy_score >= sell_score else "SELL"
            score = buy_score if direction == "BUY" else sell_score

            if score >= ULTRA_THRESHOLD:   grade = "ULTRA"
            elif score >= STRONG_THRESHOLD: grade = "STRONG"
            else:                          grade = "STANDARD"

            return direction, round(score, 2), grade, round(rsi_4h, 1), btc_score
        except Exception as e:
            log.debug("Signal detection error: %s", e)
            return None

    # ── Exit logic ────────────────────────────────────────────

    def _check_exit(self, trade: BacktestTrade, high: float, low: float,
                    t: datetime) -> str | None:
        is_buy = trade.direction == "BUY"
        if is_buy:
            if low  <= trade.sl:  return "SL"
            if high >= trade.tp3: return "TP3"
            if high >= trade.tp2: return "TP2"
            if high >= trade.tp1: return "TP1"
        else:
            if high >= trade.sl:  return "SL"
            if low  <= trade.tp3: return "TP3"
            if low  <= trade.tp2: return "TP2"
            if low  <= trade.tp1: return "TP1"
        return None

    def _calc_pnl(self, trade: BacktestTrade) -> float:
        return self._calc_pnl_price(trade, trade.exit_price)

    def _calc_pnl_price(self, trade: BacktestTrade, price: float) -> float:
        if trade.entry_price == 0:
            return 0.0
        if trade.direction == "BUY":
            return round((price - trade.entry_price) / trade.entry_price * 100, 2)
        return round((trade.entry_price - price) / trade.entry_price * 100, 2)

    # ── Statistics ────────────────────────────────────────────

    def _compute_stats(self, result: BacktestResult, trades: list[BacktestTrade]):
        if not trades:
            return

        closed = [t for t in trades if t.outcome != "PENDING"]
        wins   = [t for t in closed if t.outcome in ("TP1","TP2","TP3")]
        losses = [t for t in closed if t.outcome == "SL"]
        timeouts = [t for t in closed if t.outcome == "TIMEOUT"]

        result.total_trades = len(closed)
        result.wins         = len(wins)
        result.losses       = len(losses)
        result.timeouts     = len(timeouts)
        result.win_rate     = round(len(wins) / len(closed) * 100, 1) if closed else 0

        win_pnls  = [t.pnl_pct for t in wins]
        loss_pnls = [t.pnl_pct for t in losses]

        result.avg_win_pct   = round(float(np.mean(win_pnls)),  2) if win_pnls  else 0
        result.avg_loss_pct  = round(float(np.mean(loss_pnls)), 2) if loss_pnls else 0
        result.total_pnl_pct = round(sum(t.pnl_pct for t in closed), 2)
        result.best_trade    = round(max((t.pnl_pct for t in closed), default=0), 2)
        result.worst_trade   = round(min((t.pnl_pct for t in closed), default=0), 2)

        # Profit factor
        gross_win  = sum(p for p in win_pnls  if p > 0)
        gross_loss = abs(sum(p for p in loss_pnls if p < 0))
        result.profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else 99.0

        # Equity curve
        equity = 100.0
        curve  = [equity]
        for t in closed:
            equity += t.pnl_pct
            curve.append(round(equity, 2))
        result.equity_curve = curve

        # Max drawdown
        peak = 100.0
        max_dd = 0.0
        for val in curve:
            if val > peak: peak = val
            dd = (peak - val) / peak * 100
            if dd > max_dd: max_dd = dd
        result.max_drawdown = round(max_dd, 2)

        # Grade breakdown
        for grade in ("ULTRA", "STRONG"):
            g_trades = [t for t in closed if t.grade == grade]
            g_wins   = [t for t in g_trades if t.outcome in ("TP1","TP2","TP3")]
            wr = round(len(g_wins)/len(g_trades)*100, 1) if g_trades else 0
            if grade == "ULTRA":   result.ultra_wr  = wr
            if grade == "STRONG":  result.strong_wr = wr

    # ── Helpers ───────────────────────────────────────────────

    def _fetch(self, symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
        cols = ["open_time","open","high","low","close","volume",
                "close_time","quote_vol","trades","buy_base","buy_quote","ignore"]
        for base in _BINANCE_ENDPOINTS:
            try:
                r = self._session.get(
                    f"{base}/klines",
                    params={"symbol": symbol, "interval": interval, "limit": limit},
                    timeout=15,
                )
                if r.status_code == 451:
                    continue
                r.raise_for_status()
                df = pd.DataFrame(r.json(), columns=cols)
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c])
                return df
            except Exception as e:
                log.debug("Fetch error %s %s: %s", symbol, interval, e)
                continue
        return None

    @staticmethod
    def _slope(series: pd.Series, lookback: int = 5) -> float:
        if len(series) < lookback + 1:
            return 0.0
        recent = series.iloc[-lookback:]
        v0 = float(recent.iloc[0])
        if v0 == 0:
            return 0.0
        return float((recent.iloc[-1] - v0) / v0 / lookback)