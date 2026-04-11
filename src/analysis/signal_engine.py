"""
src/analysis/signal_engine.py — Institutional Grade Trading OS v4.1
─────────────────────────────────────────────────────────────────────
Core Philosophy:
  - Trade duration controlled by regime + momentum + liquidity structure
  - Scoring-based signals (0–100) → only ≥78 pass
  - Three dynamic modes: Sniper Scalp · Intraday Swing · Structural Swing

Pipeline per symbol:
  1. Regime detection   → Strong_Trend_Impulse | Trending | Choppy_Range
  2. Liquidity sweeps   → stop hunts, displacement candles
  3. Price action       → Engulfing, Hammer, Shooting Star, Rejection
  4. Confluence score   → weighted 0–100
  5. 1m Sniper filter   → final confirmation gate
  6. Signal build       → dynamic TP/SL/hold-time per regime
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from datetime import timezone
from typing import Literal

import numpy as np
import pandas as pd

from config import RiskConfig, ScanConfig, SignalConfig
from src.analysis.btc_strength import BtcStrength
from src.analysis.indicators import (
    atr, bollinger_bands, ema, ema_value,
    macd, obv_trend, rsi, stochastic_rsi, volume_ratio, williams_r,
)
from src.analysis.news_engine import NewsEngine
from src.analysis.trade_state_machine import TradeStateMachine
from src.data.binance_client import BinanceClient
from src.utils.logger import get_logger

log = get_logger(__name__)

SignalType = Literal["BUY", "SELL"]


@dataclass
class Signal:
    symbol:         str
    signal:         SignalType
    grade:          str
    confidence:     int
    action:         str
    price:          float
    gain_24h:       float
    rsi_1h:         float
    rsi_4h:         float
    rsi_daily:      float
    tp1:            float
    tp2:            float
    tp3:            float
    sl:             float
    atr:            float
    factors:        list[str]
    strategies_hit: list[str]
    btc_score:      int
    btc_trend:      str
    confluence:     float
    hold_time:      str
    news_sentiment: str
    stoch_rsi:      dict
    bb_pct_b:       float
    williams:       float
    regime:         str   = "Trending"
    state:          str   = "ACTIVE_INTRADAY"
    strategy:       str   = "TREND"        # TREND | MEAN_REV | BREAKOUT
    ml_prob:        float = 0.5            # ML win probability (for risk sizing)
    sniper_conf:    float = 1.0            # 1m sniper confidence score



def _wavetrend(df, n1: int = 10, n2: int = 21):
    """WaveTrend oscillator (LazyBear). Returns (wt1, wt2, cross_up, cross_down)."""
    if df is None or len(df) < max(n1, n2) + 5:
        return 0.0, 0.0, False, False
    try:
        ap  = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
        esa = ap.ewm(span=n1, adjust=False).mean()
        d   = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
        ci  = (ap - esa) / (0.015 * d.replace(0, 1e-10))
        wt1 = ci.ewm(span=n2, adjust=False).mean()
        wt2 = wt1.rolling(4).mean()
        w1, w2   = float(wt1.iloc[-1]), float(wt2.iloc[-1])
        w1p, w2p = float(wt1.iloc[-2]), float(wt2.iloc[-2])
        return w1, w2, (w1 > w2 and w1p <= w2p), (w1 < w2 and w1p >= w2p)
    except Exception:
        return 0.0, 0.0, False, False


def _tqi(df, er_len: int = 20, struct_len: int = 20, mom_len: int = 10):
    """
    Trend Quality Index — from SATS by WillyAlgoTrader.
    0.0 = pure chop/noise. 1.0 = perfect clean trend.
    Weights: ER=0.35, Structure=0.25, Momentum=0.20, Volatility=0.20
    """
    if df is None or len(df) < max(er_len, struct_len, mom_len, 100) + 5:
        return 0.5, 0.5, 0.5, 0.5   # neutral defaults
    try:
        cl = df["close"].astype(float)
        hi = df["high"].astype(float)
        lo = df["low"].astype(float)
        # Factor 1: Kaufman Efficiency Ratio
        directional = cl.diff(er_len).abs()
        total_path  = cl.diff().abs().rolling(er_len).sum()
        er = (directional / total_path.replace(0, 1e-10)).clip(0, 1)
        # Factor 2: Structure (price pinned to range edge = trend)
        hi_n   = hi.rolling(struct_len).max()
        lo_n   = lo.rolling(struct_len).min()
        pos    = (cl - lo_n) / (hi_n - lo_n + 1e-10)
        struct = ((pos - 0.5).abs() * 2).clip(0, 1)
        # Factor 3: Momentum persistence (% bars aligned with window)
        win_dir = cl.diff(mom_len)
        bar_dir = cl.diff()
        aligned = sum(
            ((win_dir > 0) & (bar_dir.shift(i) > 0) |
             (win_dir < 0) & (bar_dir.shift(i) < 0)).astype(float)
            for i in range(mom_len)
        )
        mom = (aligned / mom_len).clip(0, 1)
        # Factor 4: ATR volatility ratio vs baseline
        atr    = (hi - lo).rolling(14).mean()
        atr_bl = atr.rolling(100).mean()
        vol_f  = ((atr / atr_bl.replace(0, 1e-10) - 0.6) / 1.2).clip(0, 1)
        # Weighted TQI
        tqi_val = (er * 0.35 + struct * 0.25 + mom * 0.20 + vol_f * 0.20).clip(0, 1)
        return (float(tqi_val.iloc[-1]), float(er.iloc[-1]),
                float(struct.iloc[-1]), float(mom.iloc[-1]))
    except Exception:
        return 0.5, 0.5, 0.5, 0.5


def _cci_stoch(df, cci_len: int = 28, stoch_len: int = 28, sk: int = 3, sd: int = 3):
    """
    CCI Stochastic — from FxCodebase / PineScript.
    Returns (ma_val, buy_exit_signal, sell_exit_signal, in_oversold)
    Best entry: exit_oversold=True (MA crossed above 20) = BUY
    Best entry: exit_overbought=True (MA crossed below 80) = SELL
    """
    if df is None or len(df) < max(cci_len, stoch_len) + sk + sd + 5:
        return 50.0, False, False, False
    try:
        cl = df["close"].astype(float)
        hi = df["high"].astype(float)
        lo = df["low"].astype(float)
        # CCI
        tp  = (hi + lo + cl) / 3.0
        cci = (tp - tp.rolling(cci_len).mean()) / (0.015 * tp.rolling(cci_len).std() + 1e-10)
        # Stochastic of CCI
        cci_hi  = cci.rolling(stoch_len).max()
        cci_lo  = cci.rolling(stoch_len).min()
        stoch_k = (100 * (cci - cci_lo) / (cci_hi - cci_lo + 1e-10)).rolling(sk).mean()
        ma      = stoch_k.rolling(sd).mean()  # "D" line
        val     = float(ma.iloc[-1])
        prev    = float(ma.iloc[-2]) if len(ma) > 1 else val
        exit_os = val > 20 and prev <= 20   # crossed UP from oversold → BUY
        exit_ob = val < 80 and prev >= 80   # crossed DOWN from overbought → SELL
        in_os   = val < 20
        return val, exit_os, exit_ob, in_os
    except Exception:
        return 50.0, False, False, False


class SignalEngine:

    def __init__(
        self,
        binance: BinanceClient,
        sig_cfg: SignalConfig,
        risk_cfg: RiskConfig,
        scan_cfg: ScanConfig,
        news: NewsEngine | None = None,
    ):
        self._b    = binance
        self._s    = sig_cfg
        self._r    = risk_cfg
        self._scan = scan_cfg
        self._news = news
        self._sm   = TradeStateMachine()
        self._delayed_sym: str | None = None   # set when sniper rejects but score passes
        self._delayed_btc = None

    # ════════════════════════════════════════════════════════════════════
    # MAIN ANALYSIS PIPELINE
    # ════════════════════════════════════════════════════════════════════

    def analyze(self, ticker: dict, btc: BtcStrength) -> Signal | None:
        """
        Institutional Grade Trading OS — full pipeline.
        Returns Signal only if weighted score ≥ 78 AND 1m sniper confirms.
        """
        sym   = ticker["symbol"]
        gain  = float(ticker.get("priceChangePercent", 0))
        price = float(ticker.get("lastPrice", 0) or ticker.get("current_price", 0))
        if price <= 0:
            return None

        # ── Session gate: avoid 23:00–06:00 UTC ─────────────────────
        _h = datetime.datetime.now(timezone.utc).hour
        if 23 <= _h or _h < 6:
            return None   # low liquidity window — skip



        # ── Fetch candles ─────────────────────────────────────────────
        df_5m  = self._b.get_klines(sym, "5m",  150)
        df_1h  = self._b.get_klines(sym, "1h",   80)
        df_1m  = self._b.get_klines(sym, "1m",  100)
        df_4h  = self._b.get_klines(sym, "4h",   50)
        df_1d  = self._b.get_klines(sym, "1d",   30)
        df_1w  = self._b.get_klines(sym, "1w",   10)

        if df_5m.empty or len(df_5m) < 30: return None
        if df_1h.empty or len(df_1h) < 10: return None

        c5m = df_5m["close"].astype(float)
        c1h = df_1h["close"].astype(float)
        c4h = df_4h["close"].astype(float) if not df_4h.empty and len(df_4h) >= 14 else c1h
        c1d = df_1d["close"].astype(float) if not df_1d.empty and len(df_1d) >= 10 else c4h
        p   = float(c5m.iloc[-1])

        # ── ATR gate: need sufficient volatility ─────────────────────
        atr_5m_val = atr(df_5m, 14)
        atr_pct    = (atr_5m_val / p * 100) if p > 0 else 0
        if atr_pct < 0.3:
            return None   # dead market filter — 0.3% minimum ATR

        # ════════════════════════════════════════════════════════════
        # STEP 1: REGIME DETECTION
        # ════════════════════════════════════════════════════════════
        regime, adx = self._detect_regime(df_5m)

        # Choppy_Range: route to Mean Reversion (Strategy B)
        # Also try Breakout (Strategy C) in choppy markets — they often breakout
        if regime == "Choppy_Range":
            # Calculate RSI and volume before routing
            _c5m_ch  = df_5m["close"].astype(float)
            _rsi_5m  = rsi(_c5m_ch)
            _rsi_1h  = rsi(df_1h["close"].astype(float)) if not df_1h.empty else 50.0
            _vr_ch   = volume_ratio(df_5m["volume"], 20)
            _ob_ch   = self._b.get_orderbook_imbalance(sym)
            mr_sig = self._strategy_mean_reversion(
                sym=sym, p=p, gain=gain, df_5m=df_5m, df_1h=df_1h, df_4h=df_4h,
                df_1d=df_1d, df_1m=df_1m, c5m=_c5m_ch, atr_val=atr_5m_val,
                atr_pct=atr_pct, adx=adx, rsi_5m=_rsi_5m, rsi_1h=_rsi_1h,
                btc=btc, vr=_vr_ch, ob=_ob_ch,
            )
            if mr_sig:
                return mr_sig
            # Try breakout in choppy too (choppy → breakout = range expansion)
            bo_sig = self._strategy_breakout(
                sym=sym, p=p, gain=gain, df_5m=df_5m, df_1m=df_1m,
                c5m=_c5m_ch, atr_val=atr_5m_val, atr_pct=atr_pct,
                adx=adx, rsi_5m=_rsi_5m, rsi_1h=_rsi_1h,
                btc=btc, vr=_vr_ch, ob=_ob_ch, regime=regime,
            )
            return bo_sig

        # ════════════════════════════════════════════════════════════
        # STEP 2: INDICATORS
        # ════════════════════════════════════════════════════════════

        # EMA 9/21 on 5m — confirmed alignment
        e9s  = ema(c5m, 9); e21s = ema(c5m, 21)
        e9v  = ema_value(c5m, 9); e21v = ema_value(c5m, 21)
        try:
            _e9  = [float(e9s.iloc[i])  for i in [-1,-2,-3,-4,-5]]
            _e21 = [float(e21s.iloc[i]) for i in [-1,-2,-3,-4,-5]]
            ema_bull = _e9[0] > _e21[0] and _e9[1] > _e21[1]
            ema_bear = _e9[0] < _e21[0] and _e9[1] < _e21[1]
        except Exception:
            ema_bull = e9v > e21v
            ema_bear = e9v < e21v

        # Higher timeframe bias
        e50_1h  = ema_value(c1h, 50) if len(c1h) >= 50 else p
        e50_4h  = ema_value(c4h, 50) if len(c4h) >= 50 else p
        htf_bull = p > e50_1h and p > e50_4h
        htf_bear = p < e50_1h and p < e50_4h

        # RSI
        rsi_5m_val = rsi(c5m)
        rsi_1h_val = rsi(c1h)
        rsi_4h_val = rsi(c4h) if len(c4h) >= 14 else rsi_1h_val
        rsi_1d_val = rsi(c1d) if len(c1d) >= 14 else rsi_4h_val

        # RSI safe zones: 40–65 long, 35–60 short
        rsi_ok_long  = 32 <= rsi_5m_val <= 68   # wider zone for more signals
        rsi_ok_short = 32 <= rsi_5m_val <= 65

        # MACD on 5m — slope tells us momentum direction
        ml5, ms5   = macd(c5m)
        macd_slope = self._macd_slope(c5m)
        macd_bull  = ml5 > ms5
        macd_bear  = ml5 < ms5

        # Volume
        vr_5m     = volume_ratio(df_5m["volume"], 20)
        vol_surge = vr_5m >= 1.3

        # ════════════════════════════════════════════════════════════
        # STEP 3: PRICE ACTION PATTERNS (5m)
        # ════════════════════════════════════════════════════════════
        pa_bull, pa_bear, pa_label, pa_strong = self._price_action(df_5m)

        # ════════════════════════════════════════════════════════════
        # STEP 4: LIQUIDITY SWEEPS + DISPLACEMENT
        # ════════════════════════════════════════════════════════════
        sweep_bull, sweep_bear, sweep_label = self._liquidity_sweep(df_5m)

        # ════════════════════════════════════════════════════════════
        # STEP 5: ICT CONCEPTS (Order Blocks + Break of Structure only)
        # ════════════════════════════════════════════════════════════
        ict_bull, ict_bear, ict_label = self._ict_concepts(df_5m, p)

        # ════════════════════════════════════════════════════════════
        # STEP 6: ORDERBOOK IMBALANCE (real L2 microstructure)
        # ════════════════════════════════════════════════════════════
        ob = self._b.get_orderbook_imbalance(sym)
        ob_imbalance  = ob.get("imbalance", 0.0)
        ob_bias       = ob.get("bias", "NEUTRAL")
        ob_bull = ob_imbalance >  0.12   # more bids than asks
        ob_bear = ob_imbalance < -0.12   # more asks than bids

        # ════════════════════════════════════════════════════════════
        # STEP 7: CONFLUENCE SCORING (0–100)
        # ════════════════════════════════════════════════════════════

        # Determine candidate direction
        # Bull signals available
        bull_signals = sum([ema_bull, htf_bull, pa_bull, sweep_bull, ict_bull, ob_bull])
        bear_signals = sum([ema_bear, htf_bear, pa_bear, sweep_bear, ict_bear, ob_bear])

        if bull_signals == 0 and bear_signals == 0:
            return None
        if bull_signals == bear_signals:
            # Tie — check which MACD favors
            direction = "BUY" if macd_bull else "SELL"
        else:
            direction = "BUY" if bull_signals > bear_signals else "SELL"

        score = self._score(
            direction=direction,
            regime=regime, adx=adx,
            e9v=e9v, e21v=e21v, e50_1h=e50_1h,
            e50_4h=e50_4h, p=p,
            rsi_5m=rsi_5m_val,
            vr=vr_5m, vol_surge=vol_surge,
            pa_bull=pa_bull, pa_bear=pa_bear, pa_strong=pa_strong,
            sweep_bull=sweep_bull, sweep_bear=sweep_bear,
            ict_bull=ict_bull, ict_bear=ict_bear,
            ob_imbalance=ob_imbalance,
            macd_slope=macd_slope,
            btc_score=btc.score,
        )

        # Dynamic threshold — regime + market conditions adaptive
        if regime == "Strong_Trend_Impulse":
            _min_score = 60
        elif regime == "Trending":
            _min_score = 63
        else:
            _min_score = 75    # choppy — very strict (routes to MR strategy)

        # BTC market bonus: lower bar when BTC is very bullish/bearish
        # Strong market = better odds across the board
        _is_buy_dir = (btc.score >= 40)  # approximate
        if btc.score >= 80:
            _min_score -= 8   # VERY STRONG BULL — much easier to get BUYs through
        elif btc.score >= 65:
            _min_score -= 4
        elif btc.score <= 20:
            _min_score -= 8   # VERY STRONG BEAR — easier for SELLs
        elif btc.score <= 35:
            _min_score -= 4

        # Volatility boost: high ATR = cleaner moves = lower bar
        if atr_pct >= 1.5:  _min_score -= 3
        elif atr_pct >= 1.0: _min_score -= 2

        _min_score = max(48, _min_score)   # never go below 48

        if score < _min_score:
            log.info("  ❌ %s %s | score=%d/%d | regime=%s | rsi=%.0f | %s",
                     sym, direction, score, _min_score, regime, rsi_5m_val,
                     "⬆️ score too low" if score > _min_score - 10 else "🚫 weak setup")
            return None   # below threshold

        # ── WAVETREND OSCILLATOR ──────────────────────────────────────────
        wt1_5m, wt2_5m, wt_cross_bull, wt_cross_bear = _wavetrend(df_5m)
        wt1_1h, wt2_1h, _wt_cb_1h, _wt_cd_1h        = _wavetrend(df_1h)
        wt_gold_buy     = wt_cross_bull and wt1_5m < -40
        wt_gold_sell    = wt_cross_bear and wt1_5m >  40
        wt_confirm_buy  = wt1_5m < 0 and wt1_5m > wt2_5m
        wt_confirm_sell = wt1_5m > 0 and wt1_5m < wt2_5m
        wt_1h_bull      = wt1_1h < 20
        wt_1h_bear      = wt1_1h > -20

        # ── TQI — TREND QUALITY INDEX (SATS by WillyAlgoTrader) ──────────
        # Measures market quality: 0.0 = chop/noise, 1.0 = clean trend
        # High TQI → compress bands, tighten SL, increase confidence
        # Low TQI  → widen bands, reduce size, skip marginal setups
        tqi_val, tqi_er, tqi_struct, tqi_mom = _tqi(df_5m)
        tqi_1h, _, _, _                       = _tqi(df_1h)
        tqi_trending  = tqi_val >= 0.55   # high quality trend
        tqi_choppy    = tqi_val <= 0.30   # low quality / noise
        tqi_regime    = ("Trending" if tqi_val >= 0.55
                         else "Mixed" if tqi_val >= 0.35
                         else "Choppy")

        # ── CCI STOCHASTIC (FxCodebase) ───────────────────────────────────
        # More sensitive than RSI for scalp entry timing
        # exit_oversold=True → price bouncing up from OS zone = BUY entry
        # exit_overbought=True → price dropping from OB zone = SELL entry
        cci_val, cci_exit_os, cci_exit_ob, cci_in_os = _cci_stoch(df_5m)

        # ── BTC DIRECTION FILTER (hard gate based on market regime) ──────
        # BTC < 30 = bear market — only allow SELL signals (or very strong BUYs on strong trend)
        # BTC > 70 = bull market — only allow BUY signals
        if btc.score < 30 and direction == "BUY":
            if regime != "Strong_Trend_Impulse" or score < 80:
                log.info("  ❌ %s BUY blocked | BTC=%d (bear market)", sym, btc.score)
                return None
        if btc.score > 70 and direction == "SELL":
            if regime != "Strong_Trend_Impulse" or score < 80:
                log.info("  ❌ %s SELL blocked | BTC=%d (bull market)", sym, btc.score)
                return None

        # RSI — HARD BLOCKS on extremes, penalties on borderline
        rsi_penalty = 0
        if direction == "BUY":
            if rsi_5m_val > 80:                       # extreme overbought 5m → hard block
                log.info("  ❌ %s BUY blocked | 5m RSI=%.0f > 80", sym, rsi_5m_val)
                return None
            if rsi_5m_val > 70:   rsi_penalty += 10  # overbought on 5m
            if rsi_1h_val > 75:   rsi_penalty += 12  # parabolic on 1H
            if rsi_1h_val > 82:                       # extreme: hard block
                log.info("  ❌ %s BUY blocked | 1H RSI=%.0f > 82 (extreme top)", sym, rsi_1h_val)
                return None
        else:
            if rsi_5m_val < 20:                       # extreme oversold 5m → hard block
                log.info("  ❌ %s SELL blocked | 5m RSI=%.0f < 20", sym, rsi_5m_val)
                return None
            if rsi_5m_val < 30:   rsi_penalty += 10
            if rsi_1h_val < 25:   rsi_penalty += 12
            if rsi_1h_val < 18:                       # extreme: hard block
                log.info("  ❌ %s SELL blocked | 1H RSI=%.0f < 18 (extreme bottom)", sym, rsi_1h_val)
                return None
        # Apply RSI penalty to score
        score = max(0, score - rsi_penalty)
        if score < _min_score:
            log.info("  ❌ %s %s | RSI penalty -%d → score=%d/%d", sym, direction,
                     rsi_penalty, score, _min_score)
            return None

        # ════════════════════════════════════════════════════════════
        # WAVETREND FILTER — confirm or penalize based on WT state
        # ════════════════════════════════════════════════════════════
        wt_bonus   = 0
        wt_penalty = 0
        wt_label   = ""

        if direction == "BUY":
            if wt_gold_buy:                    # WT crossed up from oversold — GOLD signal
                wt_bonus  = 15
                wt_label  = "🌊 WT GOLD BUY"
            elif wt_confirm_buy and wt1_5m < 20:  # WT trending up, not overbought
                wt_bonus  = 8
                wt_label  = "🌊 WT confirm buy"
            elif wt1_5m > 53:                  # WT overbought — bad time to buy
                wt_penalty = 12
                wt_label  = "⚠️ WT overbought"
            elif wt_cross_bear:               # WT just turned down — against us
                wt_penalty = 10
                wt_label  = "⚠️ WT turning down"
            # 1H WT confirmation bonus
            if wt_1h_bull and wt_bonus > 0:
                wt_bonus += 5
                wt_label += " (1h aligned)"
        else:  # SELL
            if wt_gold_sell:
                wt_bonus  = 15
                wt_label  = "🌊 WT GOLD SELL"
            elif wt_confirm_sell and wt1_5m > -20:
                wt_bonus  = 8
                wt_label  = "🌊 WT confirm sell"
            elif wt1_5m < -53:
                wt_penalty = 12
                wt_label  = "⚠️ WT oversold"
            elif wt_cross_bull:
                wt_penalty = 10
                wt_label  = "⚠️ WT turning up"
            if wt_1h_bear and wt_bonus > 0:
                wt_bonus += 5
                wt_label += " (1h aligned)"

        score = max(0, score + wt_bonus - wt_penalty)
        if wt_label:
            log.info("  %s %s | WT1=%.1f WT2=%.1f | %s (score→%d)",
                     sym, direction, wt1_5m, wt2_5m, wt_label, score)

        if score < _min_score:
            log.info("  ❌ %s %s | WT penalty → score=%d/%d", sym, direction, score, _min_score)
            return None

        # ════════════════════════════════════════════════════════════
        # INDICATOR CORRELATION GATE
        # All three indicators (WT + TQI + CCI) must agree or be neutral.
        # If two indicators explicitly conflict with the direction → BLOCK.
        # This prevents false signals from individual indicator triggers.
        # ════════════════════════════════════════════════════════════
        conflicts = 0

        if direction == "BUY":
            if wt1_5m > 50 and not wt_gold_buy:      conflicts += 1  # WT overbought
            if tqi_choppy and not wt_gold_buy:         conflicts += 1  # TQI says chop
            if cci_val > 85:                           conflicts += 1  # CCI overbought
            if rsi_5m_val > 75:                        conflicts += 1  # RSI overbought
        else:
            if wt1_5m < -50 and not wt_gold_sell:     conflicts += 1  # WT oversold
            if tqi_choppy and not wt_gold_sell:         conflicts += 1  # TQI says chop
            if cci_val < 15:                           conflicts += 1  # CCI oversold
            if rsi_5m_val < 25:                        conflicts += 1  # RSI oversold

        if conflicts >= 2:
            log.info("  ❌ %s %s | %d indicator conflicts (WT=%.0f TQI=%.2f CCI=%.0f RSI=%.0f) → skip",
                     sym, direction, conflicts, wt1_5m, tqi_val, cci_val, rsi_5m_val)
            return None

        # ════════════════════════════════════════════════════════════
        # TQI FILTER — adjust score based on market quality
        # ════════════════════════════════════════════════════════════
        tqi_bonus   = 0
        tqi_penalty = 0
        if tqi_trending:                      # high quality trend
            tqi_bonus   = 10
            log.debug("  TQI=%.2f TRENDING → +10 bonus", tqi_val)
        elif tqi_choppy:                      # low quality chop
            tqi_penalty = 12
            log.debug("  TQI=%.2f CHOPPY → -12 penalty", tqi_val)
            # In choppy market, only allow TREND signals with strong WT confirmation
            if not (wt_gold_buy or wt_gold_sell):
                log.info("  ❌ %s %s | TQI=%.2f choppy + no WT gold → skip",
                         sym, direction, tqi_val)
                return None
        score = max(0, score + tqi_bonus - tqi_penalty)

        # ════════════════════════════════════════════════════════════
        # CCI STOCHASTIC — precise entry timing confirmation
        # ════════════════════════════════════════════════════════════
        cci_bonus   = 0
        cci_penalty = 0
        cci_label   = ""

        if direction == "BUY":
            if cci_exit_os:             # price exiting oversold → ideal BUY timing
                cci_bonus = 12
                cci_label = "📊 CCI exit oversold (BUY timing)"
            elif cci_in_os:             # still in oversold zone → accumulate
                cci_bonus = 6
                cci_label = "📊 CCI in oversold"
            elif cci_val > 80:          # overbought → bad BUY timing
                cci_penalty = 10
                cci_label   = "⚠️ CCI overbought"
        else:  # SELL
            if cci_exit_ob:
                cci_bonus = 12
                cci_label = "📊 CCI exit overbought (SELL timing)"
            elif cci_val > 80:
                cci_bonus = 5
                cci_label = "📊 CCI in overbought"
            elif cci_val < 20:
                cci_penalty = 10
                cci_label   = "⚠️ CCI oversold"

        score = max(0, score + cci_bonus - cci_penalty)
        if cci_label:
            log.info("  %s %s | CCI=%.1f | %s → score=%d",
                     sym, direction, cci_val, cci_label, score)

        if score < _min_score:
            log.info("  ❌ %s %s | CCI/TQI → score=%d/%d", sym, direction, score, _min_score)
            return None

        # ════════════════════════════════════════════════════════════
        # STEP 8: 1-MINUTE SNIPER CONFIRMATION
        # ════════════════════════════════════════════════════════════
        # ── SNIPER AS ENTRY SCALER (not blocker) ─────────────────────
        sniper_conf = 1.0
        if not df_1m.empty and len(df_1m) >= 10:
            sniper_conf = self._sniper_score_1m(df_1m, direction, regime, ob)
            # BTC extreme market boosts sniper confidence floor
            if btc.score >= 80 and direction == "BUY":
                sniper_conf = max(sniper_conf, 0.60)
            elif btc.score <= 20 and direction == "SELL":
                sniper_conf = max(sniper_conf, 0.60)
            if sniper_conf == 0.0:
                log.info("  ⏰ %s %s | score=%d ✅ sniper=0.0 — queuing delayed entry",
                         sym, direction, score)
                # Return a partial signal flagged for delayed retry
                # Scanner will queue it and retry for up to 3 cycles
                self._delayed_sym = sym
                self._delayed_btc = btc
                return None   # scanner checks _delayed_sym to queue
            size_lbl = ("FULL" if sniper_conf >= 0.85 else
                        "LARGE" if sniper_conf >= 0.70 else
                        "HALF"  if sniper_conf >= 0.50 else "SMALL")
            log.info("  🎯 %s %s | score=%d | sniper=%.0f%% → %s position",
                     sym, direction, score, sniper_conf*100, size_lbl)


        # PULLBACK MODE — Improvement 2.3: high WR when trend intact + RSI reset
        is_pullback = False
        try:
            pb_rsi_ok = 38 <= rsi_5m_val <= 57
            e21_dist  = abs(p - e21v) / p * 100
            if e21_dist < 0.35:
                if direction == 'BUY' and e9v > e21v and pb_rsi_ok:
                    is_pullback = True
                    log.info("  📍 %s BUY PULLBACK @ EMA21 RSI=%.0f", sym, rsi_5m_val)
                elif direction == 'SELL' and e9v < e21v and pb_rsi_ok:
                    is_pullback = True
                    log.info("  📍 %s SELL PULLBACK @ EMA21 RSI=%.0f", sym, rsi_5m_val)
        except Exception:
            pass

        # ════════════════════════════════════════════════════════════
        # STEP 9: BUILD SIGNAL WITH DYNAMIC REGIME-BASED TP/SL
        # ════════════════════════════════════════════════════════════

        # Collect factors for display
        factors = self._build_factors(
            direction=direction, regime=regime, adx=adx,
            e9v=e9v, e21v=e21v,
            rsi_5m=rsi_5m_val, vr=vr_5m, atr_pct=atr_pct,
            pa_label=pa_label, sweep_label=sweep_label, ict_label=ict_label,
            ob_bias=ob_bias, ob_imbalance=ob_imbalance,
            macd_slope=macd_slope, score=score,
            pa_bull=pa_bull, pa_bear=pa_bear, pa_strong=pa_strong,
            sweep_bull=sweep_bull, sweep_bear=sweep_bear,
            ict_bull=ict_bull, ict_bear=ict_bear,
        )

        news_sent = (self._news.get_sentiment_for(sym)
                     if self._news else {"label": "neutral"})

        ml_prob = 0.5  # ML removed

        # Score rating
        _rating = ("🟢🟢🟢 EXCELLENT" if score >= 90 else
                   "🟢🟢 STRONG"      if score >= 80 else
                   "🟢 GOOD"          if score >= 68 else "🟡 MARGINAL")
        log.info("  ✅ %s %s | score=%d/100 %s | regime=%s | rsi=%.0f | atr=%.2f%% ",
                 sym, direction, score, _rating, regime, rsi_5m_val, atr_pct)

        trend_sig = self._build(
            sig_type  = direction,
            sym       = sym,
            p         = p,
            gain      = gain,
            regime    = regime,
            score     = score,
            adx       = adx,
            atr_val   = atr_5m_val,
            rsi_1h    = rsi_1h_val,
            rsi_4h    = rsi_4h_val,
            rsi_1d    = rsi_1d_val,
            btc       = btc,
            factors   = factors,
            macd_slope= macd_slope,
            vr        = vr_5m,
            e50_4h    = e50_4h,
            news_sent = news_sent.get("label", "neutral"),
            df_5m       = df_5m,
            df_4h       = df_4h,
            df_1d       = df_1d,
            sniper_conf = sniper_conf,
            strategy    = "TREND",
        )
        if trend_sig:
            return trend_sig

        # If trend signal failed to build, try Strategy C (Breakout)
        return self._strategy_breakout(
            sym=sym, p=p, gain=gain, df_5m=df_5m, df_1m=df_1m,
            c5m=c5m, atr_val=atr_5m_val, atr_pct=atr_pct,
            adx=adx, rsi_5m=rsi_5m_val, rsi_1h=rsi_1h_val,
            btc=btc, vr=vr_5m, ob=ob, regime=regime,
        )


    # ════════════════════════════════════════════════════════════════════
    # STRATEGY B — MEAN REVERSION (captures sideways markets)
    # Activated when: regime = Choppy_Range (ADX < 18)
    # Logic: RSI extreme + Bollinger Band touch + low ATR
    # ════════════════════════════════════════════════════════════════════

    def _strategy_mean_reversion(self, sym, p, gain, df_5m, df_1h, df_4h,
                                  df_1d, df_1m, c5m, atr_val, atr_pct,
                                  adx, rsi_5m, rsi_1h, btc, vr, ob) -> "Signal | None":
        """
        Mean Reversion: buy oversold at lower BB, sell overbought at upper BB.
        Only fires in Choppy_Range (ADX < 18) + low/moderate ATR.
        """
        try:
            # ── Bollinger Bands (20-period, 2σ) ──────────────────────
            close = c5m
            bb_mid  = close.rolling(20).mean()
            bb_std  = close.rolling(20).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            bb_u  = float(bb_upper.iloc[-1])
            bb_l  = float(bb_lower.iloc[-1])
            bb_m  = float(bb_mid.iloc[-1])
            bb_rng = bb_u - bb_l
            pct_b = (p - bb_l) / bb_rng if bb_rng > 0 else 0.5   # 0=lower,1=upper

            # ── Mean reversion conditions ─────────────────────────────
            # BUY: RSI oversold + touching lower BB
            buy_rsi  = rsi_5m < 32
            buy_bb   = pct_b < 0.15           # near lower band
            # SELL: RSI overbought + touching upper BB
            sell_rsi = rsi_5m > 68
            sell_bb  = pct_b > 0.85           # near upper band

            # ADX must be low (truly sideways)
            if adx > 22:
                return None   # trend too strong for mean reversion

            # ATR must be moderate (not exploding)
            if atr_pct > 1.8:
                return None   # volatility too high, mean reversion fails

            if buy_rsi and buy_bb:
                direction = "BUY"
            elif sell_rsi and sell_bb:
                direction = "SELL"
            else:
                return None

            # BTC filter + WaveTrend confirmation for MR
            if direction == "BUY" and btc.score < 25:
                return None
            if direction == "SELL" and btc.score > 75:
                return None

            # WaveTrend: MR only valid when WT confirms the extreme
            wt1_mr, wt2_mr, wt_cb_mr, wt_cd_mr = _wavetrend(df_5m)
            if direction == "BUY"  and wt1_mr > -25: return None  # not oversold in WT
            if direction == "SELL" and wt1_mr <  25: return None  # not overbought in WT

            # Score for MR signal
            score_mr = int(40 + (32 - rsi_5m) * 0.8 if direction == "BUY"
                           else 40 + (rsi_5m - 68) * 0.8)
            score_mr = min(85, max(40, score_mr))



            # TP = mid BB, SL = 1.2× ATR (tight)
            tp1 = float(bb_m)
            tp2 = float(bb_u) if direction == "BUY" else float(bb_l)
            tp3 = tp2
            sl  = p - atr_val * 1.2 if direction == "BUY" else p + atr_val * 1.2
            # Hard SL cap
            max_sl = p * 0.015
            if abs(p - sl) > max_sl:
                sl = p - max_sl if direction == "BUY" else p + max_sl

            if tp1 <= p and direction == "BUY":   return None
            if tp1 >= p and direction == "SELL":  return None

            grade = "STANDARD"
            conf  = min(95, score_mr)

            log.info("  📊 %s %s MEAN-REVERSION | RSI=%.0f pct_b=%.2f score=%d ML=%.0f%%",
                     sym, direction, rsi_5m, pct_b, score_mr, ml_prob*100)

            from src.analysis.signal_engine import Signal as _Sig
            return self._build(
                sig_type=direction,
                sym=sym, p=p, gain=gain,
                regime="Choppy_Range", score=score_mr, adx=adx,
                atr_val=atr_val, rsi_1h=rsi_1h, rsi_4h=rsi_1h,
                rsi_1d=rsi_1h, btc=btc,
                factors=[f"📊 Mean Rev | RSI={rsi_5m:.0f} | BB%={pct_b:.2f}",
                         f"BB: L={bb_l:.5g} M={bb_m:.5g} U={bb_u:.5g}"],
                macd_slope=0.0, vr=vr, e50_4h=p,
                news_sent="neutral",
                df_5m=df_5m, df_4h=df_4h if not df_4h.empty else df_5m, df_1d=df_1d,
                sniper_conf=0.8, strategy="MEAN_REV",
            )
        except Exception as e:
            log.debug("Mean reversion error %s: %s", sym, e)
            return None

    # ════════════════════════════════════════════════════════════════════
    # STRATEGY C — BREAKOUT EXPANSION (fast signals, both regimes)
    # Called from analyze() after trend strategy in Strong_Trend + Trending
    # Logic: price breaks recent high/low + volume spike + strong candle
    # ════════════════════════════════════════════════════════════════════

    def _strategy_breakout(self, sym, p, gain, df_5m, df_1m, c5m,
                            atr_val, atr_pct, adx, rsi_5m, rsi_1h,
                            btc, vr, ob, regime) -> "Signal | None":
        """
        Breakout: price closes above N-candle high with volume spike.
        Quick TP (1–1.5× ATR), tight SL. Works in Strong_Trend + Trending.
        """
        try:
            if len(df_5m) < 22:
                return None

            highs = df_5m["high"].astype(float)
            lows  = df_5m["low"].astype(float)
            vols  = df_5m["volume"].astype(float)

            # Reference window: last 15 candles (excluding current)
            ref_high = float(highs.iloc[-16:-1].max())
            ref_low  = float(lows.iloc[-16:-1].min())

            last_c  = df_5m.iloc[-1]
            o0, h0, l0, cl0 = float(last_c["open"]), float(last_c["high"]),                                float(last_c["low"]),  float(last_c["close"])
            body       = abs(cl0 - o0)
            rng        = h0 - l0 + 1e-10
            body_ratio = body / rng

            # Volume must spike ≥ 1.5× average
            vol_avg   = float(vols.iloc[-21:-1].mean())
            vol_spike = float(vols.iloc[-1]) >= vol_avg * 1.5

            # Breakout candle must close strong (body > 60%)
            strong_candle = body_ratio > 0.60

            # Bullish breakout: close above ref_high
            bo_bull = cl0 > ref_high and cl0 > o0   # closed above AND bullish candle
            # Bearish breakout: close below ref_low
            bo_bear = cl0 < ref_low  and cl0 < o0

            if not (bo_bull or bo_bear):
                return None
            if not (vol_spike and strong_candle):
                return None

            direction = "BUY" if bo_bull else "SELL"

            # RSI must not be at extreme
            if direction == "BUY"  and rsi_5m > 78: return None
            if direction == "SELL" and rsi_5m < 22:  return None

            # BTC filter
            if direction == "BUY"  and btc.score < 30: return None
            if direction == "SELL" and btc.score > 70: return None

            # ATR expanding (breakout energy)
            atr_recent = atr(df_5m.iloc[-10:], 7) if len(df_5m) >= 10 else atr_val
            if atr_recent < atr_val * 0.8:
                return None   # ATR shrinking = false breakout

            score_bo = int(55 + (vols.iloc[-1] / max(vol_avg, 1) - 1.5) * 15)
            score_bo = min(90, max(50, score_bo))



            # Quick TP: 1.5× ATR, tight SL: 1.0× ATR
            tp1 = p + atr_val * 1.5 if direction == "BUY" else p - atr_val * 1.5
            tp2 = p + atr_val * 2.5 if direction == "BUY" else p - atr_val * 2.5
            tp3 = tp2
            sl  = p - atr_val * 1.0 if direction == "BUY" else p + atr_val * 1.0
            max_sl = p * 0.015
            if abs(p - sl) > max_sl:
                sl = p - max_sl if direction == "BUY" else p + max_sl

            log.info("  🚀 %s %s BREAKOUT | body=%.0f%% vol=%.1fx score=%d ML=%.0f%%",
                     sym, direction, body_ratio*100, vol_spike and vols.iloc[-1]/vol_avg or 1,
                     score_bo, ml_prob*100)

            vol_ratio_str = f"{float(vols.iloc[-1])/max(float(vol_avg),1e-10):.1f}"
            return self._build(
                sig_type=direction, sym=sym, p=p, gain=gain,
                regime=regime, score=score_bo, adx=adx,
                atr_val=atr_val, rsi_1h=rsi_1h, rsi_4h=rsi_1h,
                rsi_1d=rsi_1h, btc=btc,
                factors=[f"🚀 Breakout | vol={vol_ratio_str}× | body={body_ratio:.0%}",
                         f"Break {'above' if bo_bull else 'below'} {ref_high if bo_bull else ref_low:.5g}"],
                macd_slope=0.0, vr=vr, e50_4h=p,
                news_sent="neutral",
                df_5m=df_5m, df_4h=df_5m, df_1d=df_5m,
                sniper_conf=0.9, strategy="BREAKOUT",
            )
        except Exception as e:
            log.debug("Breakout strategy error %s: %s", sym, e)
            return None

    # ════════════════════════════════════════════════════════════════════
    # REGIME DETECTOR
    # ════════════════════════════════════════════════════════════════════

    def _detect_regime(self, df_5m: pd.DataFrame) -> tuple[str, float]:
        """ADX approximation from True Range expansion."""
        if len(df_5m) < 30:
            return "Trending", 20.0
        try:
            h = df_5m["high"].astype(float)
            l = df_5m["low"].astype(float)
            c = df_5m["close"].astype(float)
            tr = pd.concat([
                h - l,
                abs(h - c.shift(1)),
                abs(l - c.shift(1))
            ], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean()
            # ADX proxy: ATR range vs price range expansion
            atr_now  = float(atr14.iloc[-1]) if not atr14.empty else 0
            atr_past = float(atr14.iloc[-14]) if len(atr14) >= 14 else atr_now
            expansion = atr_now / max(atr_past, 1e-10)
            # Directional movement proxy
            dm_up   = (h - h.shift(1)).clip(lower=0)
            dm_down = (l.shift(1) - l).clip(lower=0)
            adx_proxy = float(dm_up.rolling(14).mean().iloc[-1] /
                              max(dm_down.rolling(14).mean().iloc[-1], 1e-10)) * 20
            adx = min(45.0, max(10.0, adx_proxy * expansion))
            if adx > 28:   return "Strong_Trend_Impulse", round(adx, 1)
            elif adx > 18: return "Trending",              round(adx, 1)
            else:          return "Choppy_Range",          round(adx, 1)
        except Exception:
            return "Trending", 20.0

    # ════════════════════════════════════════════════════════════════════
    # PRICE ACTION
    # ════════════════════════════════════════════════════════════════════

    def _price_action(self, df_5m: pd.DataFrame) -> tuple[bool, bool, str, bool]:
        """Detects Engulfing, Hammer, Shooting Star, rejection candles."""
        if len(df_5m) < 3:
            return False, False, "", False
        c0 = df_5m.iloc[-1]; c1 = df_5m.iloc[-2]
        o0,h0,l0,cl0 = float(c0["open"]),float(c0["high"]),float(c0["low"]),float(c0["close"])
        o1,h1,l1,cl1 = float(c1["open"]),float(c1["high"]),float(c1["low"]),float(c1["close"])
        body0  = abs(cl0 - o0)
        body1  = abs(cl1 - o1)
        rng0   = h0 - l0 + 1e-10
        uw0    = h0 - max(cl0, o0)
        lw0    = min(cl0, o0) - l0
        strong = (body0 / rng0) > 0.62   # strong displacement candle

        pa_bull = pa_bear = False
        label = ""

        # Bullish Engulfing
        if cl0 > o0 and cl1 < o1 and cl0 > o1 and o0 < cl1 and body0 > body1 * 0.8:
            pa_bull = True; label = "Bullish Engulfing"
        # Hammer / Bullish Pin Bar
        if lw0 >= 2.0 * body0 and lw0 >= 0.5 * rng0:
            pa_bull = True; label = "Hammer/Pin Bar"
        # Bullish Rejection
        if lw0 >= 1.5 * body0 and cl0 > (h0 + l0) / 2:
            pa_bull = True; label = label or "Bullish Rejection"

        # Bearish Engulfing
        if cl0 < o0 and cl1 > o1 and cl0 < o1 and o0 > cl1 and body0 > body1 * 0.8:
            pa_bear = True; label = "Bearish Engulfing"
        # Shooting Star / Bearish Pin Bar
        if uw0 >= 2.0 * body0 and uw0 >= 0.5 * rng0:
            pa_bear = True; label = "Shooting Star"
        # Bearish Rejection
        if uw0 >= 1.5 * body0 and cl0 < (h0 + l0) / 2:
            pa_bear = True; label = label or "Bearish Rejection"

        # Morning/Evening Star (3-candle)
        if len(df_5m) >= 4:
            c2 = df_5m.iloc[-3]
            o2,cl2 = float(c2["open"]), float(c2["close"])
            doji = abs(cl1 - o1) < rng0 * 0.3
            if cl2 < o2 and doji and cl0 > o0 and cl0 > (o2+cl2)/2:
                pa_bull = True; label = "Morning Star"
            if cl2 > o2 and doji and cl0 < o0 and cl0 < (o2+cl2)/2:
                pa_bear = True; label = "Evening Star"

        return pa_bull, pa_bear, label, strong

    # ════════════════════════════════════════════════════════════════════
    # LIQUIDITY SWEEPS
    # ════════════════════════════════════════════════════════════════════

    def _liquidity_sweep(self, df_5m: pd.DataFrame) -> tuple[bool, bool, str]:
        """Detects stop hunts + displacement (ICT liquidity sweep concept)."""
        if len(df_5m) < 8:
            return False, False, ""
        try:
            recent_high = float(df_5m["high"].iloc[-7:-1].max())
            recent_low  = float(df_5m["low"].iloc[-7:-1].min())
            last = df_5m.iloc[-1]; prev = df_5m.iloc[-2]
            lo, hi, cl, op = float(last["low"]), float(last["high"]), float(last["close"]), float(last["open"])
            rng = hi - lo + 1e-10

            # Bull sweep: wicked below swing low, closed strongly above
            bull = (lo <= recent_low * 0.9995 and
                    cl > op and
                    cl > float(prev["high"]) * 0.998 and
                    (cl - lo) / rng > 0.65)

            # Bear sweep: wicked above swing high, closed strongly below
            bear = (hi >= recent_high * 1.0005 and
                    cl < op and
                    cl < float(prev["low"]) * 1.002 and
                    (hi - cl) / rng > 0.65)

            label = "Liquidity Sweep + Displacement" if (bull or bear) else ""
            return bull, bear, label
        except Exception:
            return False, False, ""

    # ════════════════════════════════════════════════════════════════════
    # ICT CONCEPTS
    # ════════════════════════════════════════════════════════════════════

    def _ict_concepts(self, df_5m: pd.DataFrame, p: float) -> tuple[bool, bool, str]:
        """
        ICT Concepts — SIMPLIFIED per improvement plan:
        Keep ONLY: Order Block + Break of Structure
        Remove: FVG, OTE, ChoCH (added noise without ML contribution)
        """
        ict_bull = ict_bear = False
        labels = []
        try:
            # ── Order Block (proven high-value ICT concept) ───────────
            if len(df_5m) >= 10:
                for i in range(3, 10):
                    ob_o = float(df_5m.iloc[-i]["open"])
                    ob_c = float(df_5m.iloc[-i]["close"])
                    ob_h = float(df_5m.iloc[-i]["high"])
                    ob_l = float(df_5m.iloc[-i]["low"])
                    if ob_c < ob_o:  # bearish candle = bullish OB
                        hi_after = float(df_5m["high"].iloc[max(-i+1,-1):-1].max()) if i > 2 else p
                        if (hi_after - ob_h) / ob_h * 100 >= 0.4 and ob_l <= p <= ob_h * 1.002:
                            ict_bull = True; labels.append("Bull OB"); break
                    if ob_c > ob_o:  # bullish candle = bearish OB
                        lo_after = float(df_5m["low"].iloc[max(-i+1,-1):-1].min()) if i > 2 else p
                        if (ob_l - lo_after) / ob_l * 100 >= 0.4 and ob_l * 0.998 <= p <= ob_h:
                            ict_bear = True; labels.append("Bear OB"); break

            # ── Break of Structure (high reliability directional signal) ──
            if len(df_5m) >= 20:
                prev_h = float(df_5m["high"].iloc[-20:-5].max())
                prev_l = float(df_5m["low"].iloc[-20:-5].min())
                curr_h = float(df_5m["high"].iloc[-5:].max())
                curr_l = float(df_5m["low"].iloc[-5:].min())
                if curr_h > prev_h and p > prev_h:
                    ict_bull = True; labels.append("BOS ↑")
                if curr_l < prev_l and p < prev_l:
                    ict_bear = True; labels.append("BOS ↓")

        except Exception:
            pass

        return ict_bull, ict_bear, " + ".join(labels[:2]) if labels else ""

    # ════════════════════════════════════════════════════════════════════
    # CONFLUENCE SCORER (0–100)
    # ════════════════════════════════════════════════════════════════════

    def _score(self, direction: str, regime: str, adx: float,
               e9v: float, e21v: float, e50_1h: float,
               e50_4h: float, p: float,
               rsi_5m: float, vr: float, vol_surge: bool,
               pa_bull: bool, pa_bear: bool, pa_strong: bool,
               sweep_bull: bool, sweep_bear: bool,
               ict_bull: bool, ict_bear: bool,
               ob_imbalance: float, macd_slope: float,
               btc_score: int) -> int:
        """
        Weighted confluence score 0–100.
        Score ≥ 78 required to generate signal.
        """
        s = 0
        is_buy = direction == "BUY"

        # ── 0. Baseline — signals must agree (min quality bar) ────────
        # Count how many independent signal groups confirm direction
        confirms = sum([
            (e9v > e21v) if is_buy else (e9v < e21v),
            pa_bull if is_buy else pa_bear,
            sweep_bull if is_buy else sweep_bear,
            ict_bull if is_buy else ict_bear,
            ob_imbalance > 0.08 if is_buy else ob_imbalance < -0.08,
        ])
        if confirms >= 3: s += 8    # strong multi-confirmation
        elif confirms >= 2: s += 4  # moderate confirmation

        # ── 1. TOP-DOWN TREND ALIGNMENT — up to 40pts ─────────────────
        # 5m trend (entry timeframe)
        if is_buy:
            if e9v > e21v:    s += 12
        else:
            if e9v < e21v:    s += 12
        # 1H trend (primary filter) — MUST align
        if is_buy:
            if e21v > e50_1h: s += 14  # price above 1H EMA50
        else:
            if e21v < e50_1h: s += 14
        # 4H trend (higher timeframe bias)
        if is_buy:
            if e50_4h > 0 and p > e50_4h:  s += 14   # price above 4H EMA50
        else:
            if e50_4h > 0 and p < e50_4h:  s += 14
        # Counter-trend PENALTY — trading against HTF = high risk
        if is_buy and e21v < e50_1h:
            s -= 12  # 1H bearish trend = BUY counter-trend penalty
        if not is_buy and e21v > e50_1h:
            s -= 12  # 1H bullish trend = SELL counter-trend penalty

        # ── 2. Regime strength — up to 18pts ─────────────────────────
        if regime == "Strong_Trend_Impulse": s += 18
        elif regime == "Trending":           s += 12
        else:                                s += 5  # choppy: low weight

        # ── 3. Price Action — up to 18pts ────────────────────────────
        pa_ok = pa_bull if is_buy else pa_bear
        if pa_ok and pa_strong: s += 18
        elif pa_ok:             s += 11

        # ── 4. Liquidity Sweep — up to 16pts ─────────────────────────
        sw_ok = sweep_bull if is_buy else sweep_bear
        if sw_ok: s += 16

        # ── 5. Orderbook imbalance — up to 15pts ─────────────────────
        if is_buy  and ob_imbalance >  0.20: s += 15
        elif is_buy and ob_imbalance > 0.12: s += 10
        if not is_buy and ob_imbalance < -0.20: s += 15
        elif not is_buy and ob_imbalance < -0.12: s += 10

        # ── 6. Volume confirmation — up to 10pts ─────────────────────
        if vol_surge:              s += 10
        elif vr >= 1.1:            s += 5

        # ── 7. ICT concepts — up to 10pts ────────────────────────────
        ict_ok = ict_bull if is_buy else ict_bear
        if ict_ok: s += 10

        # ── 8. BTC alignment — up to 15pts ──────────────────────────
        if is_buy:
            if   btc_score >= 80: s += 15
            elif btc_score >= 65: s += 10
            elif btc_score >= 50: s += 5
            elif btc_score >= 40: s += 2
        else:
            if   btc_score <= 20: s += 15
            elif btc_score <= 35: s += 10
            elif btc_score <= 50: s += 5
            elif btc_score <= 60: s += 2
        # ── 9. MACD momentum penalty / bonus ─────────────────────────
        # Positive slope = momentum growing (good for long, bad for short)
        if is_buy:
            if macd_slope > 0.0005:    s += 6   # accelerating
            elif macd_slope < -0.001:   s -= 4   # decaying momentum (advisory)
        else:
            if macd_slope < -0.0005:   s += 6
            elif macd_slope > 0.001:    s -= 4

        return min(100, max(0, s))

    # ════════════════════════════════════════════════════════════════════
    # 1-MINUTE SNIPER FILTER
    # ════════════════════════════════════════════════════════════════════

    def _sniper_score_1m(self, df_1m: pd.DataFrame, direction: str,
                         regime: str, ob: dict) -> float:
        """
        Sniper as ENTRY SCALER — returns confidence 0.0–1.0 (not boolean).

        1.0 = full position (strong confirmation)
        0.7 = 70% position (moderate confirmation)
        0.4 = 40% position (weak — enter small, add on confirmation)
        0.0 = skip entirely (actively against us)

        This replaces the hard reject — every qualifying signal gets SOME execution.
        Only true counter-signals (OB opposing + EMA opposing) get 0.0.
        """
        try:
            c1m    = df_1m["close"].astype(float)
            e9_1m  = ema_value(c1m, 9)
            e21_1m = ema_value(c1m, 21)
            last   = df_1m.iloc[-1]
            o0, h0, l0, cl0 = (float(last[k]) for k in ["open","high","low","close"])
            body       = abs(cl0 - o0)
            rng        = h0 - l0 + 1e-10
            body_ratio = body / rng
            candle_dir = (cl0 >= o0) if direction == "BUY" else (cl0 <= o0)
            ob_imb     = ob.get("imbalance", 0.0)
            ema_ok     = (e9_1m > e21_1m) if direction == "BUY" else (e9_1m < e21_1m)
            ob_against = ((ob_imb < -0.20) if direction == "BUY" else (ob_imb > 0.20)) if abs(ob_imb) > 0.02 else False

            # If orderbook is strongly against AND EMA disagrees → skip
            if ob_against and not ema_ok:
                return 0.0

            # Score components: ema, candle body, candle direction, ob
            conf = 0.0
            if ema_ok:         conf += 0.40   # EMA aligned
            if candle_dir:     conf += 0.20   # candle closing right direction
            if body_ratio > 0.60: conf += 0.25  # strong candle
            elif body_ratio > 0.35: conf += 0.12  # moderate candle
            if not ob_against: conf += 0.15  # OB not opposing

            # Regime modifier
            if regime == "Strong_Trend_Impulse":
                conf = max(conf, 0.40)   # always at least 40% in strong trend

            return round(min(1.0, conf), 2)

        except Exception:
            return 0.70   # unavailable data → default 70%

    # ════════════════════════════════════════════════════════════════════
    # MACD SLOPE
    # ════════════════════════════════════════════════════════════════════

    def _macd_slope(self, closes: pd.Series, period: int = 5) -> float:
        try:
            ema12 = closes.ewm(span=12, adjust=False).mean()
            ema26 = closes.ewm(span=26, adjust=False).mean()
            _macd = ema12 - ema26
            _sig  = _macd.ewm(span=9, adjust=False).mean()
            hist  = _macd - _sig
            if len(hist) < period + 1: return 0.0
            return float(hist.iloc[-1] - hist.iloc[-period]) / period
        except Exception:
            return 0.0

    # ════════════════════════════════════════════════════════════════════
    # SIGNAL BUILDER — dynamic TP/SL/hold by regime
    # ════════════════════════════════════════════════════════════════════

    def _build(self, sig_type: str, sym: str, p: float, gain: float,
               regime: str, score: int, adx: float,
               atr_val: float, rsi_1h: float, rsi_4h: float, rsi_1d: float,
               btc: BtcStrength, factors: list, macd_slope: float, vr: float,
               e50_4h: float, news_sent: str, df_5m, df_4h, df_1d,
               pos_mult: float = 1.0,
               sniper_conf: float = 1.0, strategy: str = "TREND") -> Signal | None:

        is_buy = sig_type == "BUY"

        # ── Grade & confidence ────────────────────────────────────────
        if score >= 92:
            grade = f"ULTRA {'🟢🟢🟢' if is_buy else '🔴🔴🔴'}"
            base_conf = min(93, 85 + (score - 92))
        elif score >= 85:
            grade = f"STRONG {'🟢🟢' if is_buy else '🔴🔴'}"
            base_conf = min(87, 76 + (score - 85))
        else:
            grade = f"STANDARD {'🟢' if is_buy else '🔴'}"
            base_conf = min(78, 65 + (score - 78))

        # ML + sniper_conf + news adjust confidence
        ml_boost = 0
        sniper_boost = (5 if sniper_conf >= 0.85 else 2 if sniper_conf >= 0.70 else
                        0 if sniper_conf >= 0.50 else -3)
        news_adj = (3 if (is_buy  and news_sent == "positive") or
                         (not is_buy and news_sent == "negative") else
                   -3 if (is_buy  and news_sent == "negative") or
                          (not is_buy and news_sent == "positive") else 0)
        confidence = min(95, max(50, base_conf + ml_boost + sniper_boost + news_adj))

        # Strategy label prefix in grade
        strat_emoji = {"MEAN_REV": "📊", "BREAKOUT": "🚀", "TREND": ""}.get(strategy, "")
        if strat_emoji:
            grade = f"{strat_emoji} {grade}"

        # ── Regime-adaptive TP/SL ─────────────────────────────────────
        # ATR multipliers change per regime
        # ── R:R optimized per regime ──────────────────────────────────
        # Key insight: wider SL = fewer stop-outs, but TP must scale too
        # Target R:R ≥ 1.5:1 so breakeven WR is ≤ 40% (achievable with edge)
        if regime == "Strong_Trend_Impulse":
            sl_mult  = 1.0    # SL = 1.0× ATR — scalp mode, tight stop
            tp1_mult = 1.5    # TP1 = 1.5× ATR — R:R = 1.5:1
            tp2_mult = 2.5    # TP2 = 2.5× ATR
            tp3_mult = 4.0    # TP3 = 4.0× ATR — runner
            hold     = "5–15min (Scalp)"
        elif regime == "Trending":
            sl_mult  = 1.2    # SL = 1.2× ATR — tighter SL improves R:R
            tp1_mult = 2.0    # TP1 = 2.0× ATR — R:R = 1.67:1, breakeven 37%
            tp2_mult = 3.5    # TP2 = 3.5× ATR — R:R = 2.9:1
            tp3_mult = 5.5    # TP3 = 5.5× ATR — runner
            hold     = "15min–6h (Intraday)"
        else:
            sl_mult  = 0.8    # SL = 0.8× ATR — very tight, fast exit if wrong
            tp1_mult = 1.5    # TP1 = 1.5× ATR — R:R = 1.87:1, breakeven 35%
            tp2_mult = 2.5    # TP2 = 2.5× ATR
            tp3_mult = 4.0    # TP3 = 4.0× ATR
            hold     = "30s–8min (Sniper)"

        atr_buf = atr_val * sl_mult if atr_val > 0 else p * 0.008
        # Minimum SL: 0.6% for sniper, 0.8% for intraday, 1.0% for structural
        min_sl_pct = {"Strong_Trend_Impulse": 0.010, "Trending": 0.008, "Choppy_Range": 0.006}[regime]
        # Maximum SL: HARD CAP at 1.5% — prevents catastrophic losses
        max_sl_pct = 0.015
        min_sl     = p * min_sl_pct
        max_sl     = p * max_sl_pct
        atr_buf    = min(atr_buf, max_sl)

        # Structure-based SL: max(ATR stop, recent swing level)
        # Improvement 3.1 — align SL with actual market structure
        if is_buy:
            swing_low  = float(df_5m["low"].iloc[-15:].min()) if len(df_5m) >= 15 else 0
            struct_sl  = swing_low * 0.999 if swing_low > 0 else 0  # just below swing low
            atr_sl     = p - max(atr_buf, min_sl)
            # Use structure level if it's tighter than ATR but still within max
            sl_candidate = max(atr_sl, struct_sl)
            sl  = round(max(p - max_sl, sl_candidate), 8)  # never exceed max_sl_pct
            tp1 = round(p + atr_val * tp1_mult, 8)
            tp2 = round(p + atr_val * tp2_mult, 8)
            tp3 = round(p + atr_val * tp3_mult, 8)
            action = "Buy / Go Long"
        else:
            sl  = round(p + max(atr_buf, min_sl), 8)
            tp1 = round(p - atr_val * tp1_mult, 8)
            tp2 = round(p - atr_val * tp2_mult, 8)
            tp3 = round(p - atr_val * tp3_mult, 8)
            action = "Exit Long / Sell / Short"

        # ── Initial state from regime ─────────────────────────────────
        initial_state = self._sm.get_initial_state(regime)

        # ── Optional bonus scorer (adds to confidence, not gates) ─────
        try:
            c1h  = df_4h["close"].astype(float)
            srsi = stochastic_rsi(c1h) if len(c1h) >= 28 else {"k": 50, "d": 50}
            bb   = bollinger_bands(c1h) if len(c1h) >= 20 else {"pct_b": 0.5}
            wr   = williams_r(df_4h)    if not df_4h.empty and len(df_4h) >= 14 else -50
        except Exception:
            srsi = {"k": 50, "d": 50}; bb = {"pct_b": 0.5}; wr = -50

        return Signal(
            symbol=sym, signal=sig_type, grade=grade,
            confidence=confidence, action=action,
            price=p, gain_24h=gain,
            rsi_1h=rsi_1h, rsi_4h=rsi_4h, rsi_daily=rsi_1d,
            tp1=tp1, tp2=tp2, tp3=tp3, sl=sl,
            atr=atr_val,
            strategy=strategy,
            ml_prob=0.5,
            sniper_conf=sniper_conf,
            factors=factors,
            strategies_hit=[f"InstitutionalOS-v4.1 ({regime})"],
            btc_score=btc.score, btc_trend=btc.trend,
            confluence=round(score / 10, 1),
            hold_time=hold,
            news_sentiment=news_sent,
            stoch_rsi=srsi, bb_pct_b=bb.get("pct_b", 0.5), williams=float(wr),
            regime=regime, state=initial_state,
        )

    # ════════════════════════════════════════════════════════════════════
    # FACTOR LIST BUILDER
    # ════════════════════════════════════════════════════════════════════

    def _build_factors(self, direction, regime, adx, e9v, e21v, rsi_5m, vr, atr_pct,
                       pa_label, sweep_label, ict_label, ob_bias, ob_imbalance, macd_slope,
                       score, pa_bull, pa_bear, pa_strong, sweep_bull, sweep_bear,
                       ict_bull, ict_bear) -> list[str]:
        is_buy = direction == "BUY"
        f = [
            f"✅ Regime: {regime} (ADX≈{adx:.1f})",
            f"✅ Score: {score}/100 — threshold 78",
            f"✅ EMA9={e9v:.5g} {'>' if is_buy else '<'} EMA21={e21v:.5g}",
            f"✅ RSI {rsi_5m:.0f} | Vol {vr:.1f}x | ATR {atr_pct:.2f}%",
            f"✅ Orderbook: {ob_bias} (imbalance {ob_imbalance:+.2f})",
            f"✅ MACD slope: {macd_slope:+.5f}",
        ]
        if pa_label:
            f.append(f"✅ PA: {pa_label}{' 💪 strong' if pa_strong else ''}")
        if sweep_label and (sweep_bull if is_buy else sweep_bear):
            f.append(f"✅ Liquidity: {sweep_label}")
        if ict_label and (ict_bull if is_buy else ict_bear):
            f.append(f"✅ ICT: {ict_label}")
        return f

    # ════════════════════════════════════════════════════════════════════
    # SLOPE HELPER
    # ════════════════════════════════════════════════════════════════════

    @staticmethod
    def _slope(series: pd.Series, lookback: int = 5) -> float:
        if len(series) < lookback + 1: return 0.0
        recent = series.iloc[-lookback:]
        v0 = float(recent.iloc[0])
        return float((recent.iloc[-1] - v0) / v0 / lookback) if v0 != 0 else 0.0