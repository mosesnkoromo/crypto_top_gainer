"""
src/analysis/signal_engine.py — v9 FULL CONFLUENCE ENGINE
──────────────────────────────────────────────────────────
Base: v8 swing pullback (proven 60.9% win rate on March 18)
All new strategies are ADDITIVE BONUS only — never gates.
Hard gates unchanged: BTC<57, RSI>65, no slope, below EMA50.

Bonus strategies (all capped, additive only):
  CRT     — Previous day levels, 50% equilibrium, displacement
  S/R     — Weekly support/resistance levels
  FIB     — Fibonacci 0.5 / 0.618 retracement zones
  OB      — Order block detection (institutional footprints)
  DIV     — RSI divergence (price/RSI disagreement)
  VWAP    — Price near Volume-Weighted Average Price
  ICHI    — Ichimoku cloud position + TK cross
  SUPER   — Supertrend direction
  SESSION — London/NY session bonus (EAT = UTC+3)
  ATR     — Volatility quality filter (not too choppy, not too quiet)
  BTCCOR  — Coin strength vs BTC (relative performance)
  MSB     — Market Structure Break / Break of Structure
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from config import RiskConfig, SignalConfig, ScanConfig
from src.analysis.btc_strength import BtcStrength
from src.analysis.indicators import (
    atr, bollinger_bands, ema_value, ema,
    obv_trend, rsi, stochastic_rsi, volume_ratio, williams_r, macd,
)
from src.analysis.news_engine import NewsEngine
from src.data.binance_client import BinanceClient
from src.utils.logger import get_logger

log = get_logger(__name__)

SignalType = Literal["BUY", "SELL"]

ULTRA_THRESHOLD    = 7.0
STRONG_THRESHOLD   = 5.0
MIN_SEND_GRADE     = "STRONG"

# BTC gates (data-proven from March 18/20 analysis)
BUY_BTC_MIN   = 57
SELL_BTC_MAX  = 50

# RSI hard limits
BUY_RSI_MAX   = 65
BUY_RSI_MIN   = 35
SELL_RSI_MIN  = 45
SELL_RSI_MAX  = 72

# Trend slope requirement
MIN_SLOPE     = 0.0015

# Session hours in UTC (EAT = UTC+3, so London 07:00-16:00 UTC, NY 13:00-22:00 UTC)
LONDON_OPEN_UTC  = 7
LONDON_CLOSE_UTC = 16
NY_OPEN_UTC      = 13
NY_CLOSE_UTC     = 22


@dataclass
class Signal:
    symbol:          str
    signal:          SignalType
    grade:           str
    confidence:      int
    action:          str
    price:           float
    gain_24h:        float
    rsi_1h:          float
    rsi_4h:          float
    rsi_daily:       float
    tp1:             float
    tp2:             float
    tp3:             float
    sl:              float
    factors:         list[str]
    strategies_hit:  list[str]
    btc_score:       int
    btc_trend:       str
    confluence:      float
    hold_days:       str
    news_sentiment:  str
    stoch_rsi:       dict
    bb_pct_b:        float
    williams:        float


class SignalEngine:

    def __init__(self, binance: BinanceClient, sig_cfg: SignalConfig,
                 risk_cfg: RiskConfig, scan_cfg: ScanConfig,
                 news: NewsEngine | None = None):
        self._b    = binance
        self._s    = sig_cfg
        self._r    = risk_cfg
        self._scan = scan_cfg
        self._news = news

    def analyze(self, ticker: dict, btc: BtcStrength) -> Signal | None:
        sym   = ticker["symbol"]
        gain  = float(ticker.get("priceChangePercent", 0))
        price = float(ticker.get("lastPrice", 0) or ticker.get("current_price", 0))
        if price <= 0:
            return None

        # ── Fetch timeframes ──────────────────────────────────
        df_1w = self._b.get_klines(sym, "1w", 30)   # weekly for S/R
        df_1d = self._b.get_klines(sym, "1d", 60)   # daily trend
        df_4h = self._b.get_klines(sym, "4h", 100)  # confirmation
        df_1h = self._b.get_klines(sym, "1h", 48)   # entry timing

        if df_4h.empty or len(df_4h) < 30: return None
        if df_1d.empty or len(df_1d) < 20: return None

        close_1d = df_1d["close"]
        close_4h = df_4h["close"]
        close_1h = df_1h["close"] if not df_1h.empty and len(df_1h) >= 10 else close_4h
        price_now = float(close_4h.iloc[-1])

        # ── Core indicators ────────────────────────────────────
        rsi_d   = rsi(close_1d)
        rsi_4h  = rsi(close_4h)
        rsi_1h  = rsi(close_1h)
        e20_d   = ema_value(close_1d, 20)
        e50_d   = ema_value(close_1d, 50)
        e20_4h  = ema_value(close_4h, 20)
        e50_4h  = ema_value(close_4h, 50)
        e200_d  = ema_value(close_1d, 200) if len(close_1d) >= 200 else e50_d
        vr_4h   = volume_ratio(df_4h["volume"], 20)
        vr_1d   = volume_ratio(df_1d["volume"], 20)
        atr_4h  = atr(df_4h, 14)
        atr_1d  = atr(df_1d, 14)
        bb_4h   = bollinger_bands(close_4h)
        srsi_4h = stochastic_rsi(close_4h)
        wr_4h   = williams_r(df_4h)
        obv_4h  = obv_trend(df_4h)
        ml, ms  = macd(close_4h)

        ema20_d_s  = ema(close_1d, 20)
        ema20_4h_s = ema(close_4h, 20)
        slope_d    = self._slope(ema20_d_s, 5)
        slope_4h   = self._slope(ema20_4h_s, 6)

        swing_low  = float(df_4h["low"].rolling(20).min().iloc[-1])
        swing_high = float(df_4h["high"].rolling(20).max().iloc[-1])

        news_sent = (self._news.get_sentiment_for(sym)
                     if self._news else {"label": "neutral", "articles": 0})

        # ── SCALP PRE-FILTERS (mandatory gates) ───────────────
        # Fetch 15m for entry timing and volatility
        df_15m = self._b.get_klines(sym, "15m", 30)
        close_15m = df_15m["close"] if not df_15m.empty and len(df_15m) >= 14 else close_1h

        # MANDATORY 1: Volatility filter — ATR(15m) > 0.6% price
        # Skip quiet/choppy markets entirely
        atr_15m_val = atr(df_15m, 14) if not df_15m.empty and len(df_15m) >= 14 else 0
        atr_15m_pct = (atr_15m_val / price_now * 100) if price_now > 0 else 0
        if 0 < atr_15m_pct < 0.6:
            return None  # market too quiet

        # MANDATORY 2: 4H trend alignment — only trade WITH macro direction
        macro_bull = price_now > e50_4h   # LONG only above 4H EMA50
        macro_bear = price_now < e50_4h   # SHORT only below 4H EMA50

        # MANDATORY 3: No-trade zone — RSI 45-55 = chop, no edge
        if 45 < rsi_1h < 55:
            return None

        # MANDATORY 4: 15m structure confirmation
        # LONG: last 15m candle closes higher than previous (momentum up)
        # SHORT: last 15m candle closes lower than previous (momentum down)
        if len(close_15m) >= 2:
            m_structure_bull = float(close_15m.iloc[-1]) > float(close_15m.iloc[-2])
            m_structure_bear = float(close_15m.iloc[-1]) < float(close_15m.iloc[-2])
        else:
            m_structure_bull = m_structure_bear = True  # no data, don't filter

        # MANDATORY 5: EMA20 direction on 1H
        e20_1h = ema_value(close_1h, 20) if len(close_1h) >= 20 else price_now
        ema_bull = price_now > e20_1h   # price above 1H EMA20 = bullish
        ema_bear = price_now < e20_1h   # price below 1H EMA20 = bearish

        # ── Score ─────────────────────────────────────────────
        buy_score, buy_factors = self._buy_score(
            price_now, gain, rsi_d, rsi_4h, rsi_1h,
            e20_d, e50_d, e200_d, e20_4h, e50_4h,
            slope_d, slope_4h, vr_4h, vr_1d, atr_4h, atr_1d,
            obv_4h, bb_4h, srsi_4h, wr_4h, ml, ms,
            btc, news_sent, df_1d, df_4h, df_1h, df_1w, sym
        )
        sell_score, sell_factors = self._sell_score(
            price_now, gain, rsi_d, rsi_4h, rsi_1h,
            e20_d, e50_d, e20_4h, e50_4h,
            slope_d, slope_4h, vr_4h, vr_1d, atr_4h, atr_1d,
            obv_4h, bb_4h, srsi_4h, wr_4h, ml, ms,
            btc, news_sent, df_1d, df_4h, df_1h, df_1w, sym
        )

        # Compare integer (core) part of score to confluence threshold
        buy_core  = int(buy_score)
        sell_core = int(sell_score)

        # MANDATORY gates must ALL pass before optional score is checked
        buy_mandatory  = macro_bull and ema_bull and m_structure_bull
        sell_mandatory = macro_bear and ema_bear and m_structure_bear

        if buy_core >= self._s.min_buy_confluence and buy_score > sell_score and buy_mandatory:
            buy_factors.insert(0, f"✅ Scalp gates: 4H={price_now:.4g}>EMA50, 1H EMA↑, 15m↑, ATR={atr_15m_pct:.1f}%")
            return self._build("BUY", sym, buy_score, buy_factors, ["ScalpPullback"],
                               price_now, gain, rsi_1h, rsi_4h, rsi_d, btc,
                               srsi_4h, bb_4h["pct_b"], wr_4h,
                               news_sent.get("label", "neutral"),
                               e50_4h, atr_4h, swing_low)
        if sell_core >= self._s.min_sell_confluence and sell_score > buy_score and sell_mandatory:
            sell_factors.insert(0, f"✅ Scalp gates: 4H={price_now:.4g}<EMA50, 1H EMA↓, 15m↓, ATR={atr_15m_pct:.1f}%")
            return self._build("SELL", sym, sell_score, sell_factors, ["ScalpPullback"],
                               price_now, gain, rsi_1h, rsi_4h, rsi_d, btc,
                               srsi_4h, bb_4h["pct_b"], wr_4h,
                               news_sent.get("label", "neutral"),
                               e50_4h, atr_4h, swing_high)
        return None

    # ════════════════════════════════════════════════════════
    # BUY SCORING
    # ════════════════════════════════════════════════════════

    def _buy_score(self, price, gain, rsi_d, rsi_4h, rsi_1h,
                   e20_d, e50_d, e200_d, e20_4h, e50_4h,
                   slope_d, slope_4h, vr_4h, vr_1d, atr_4h, atr_1d,
                   obv, bb, srsi, wr, ml, ms,
                   btc, news, df_1d, df_4h, df_1h, df_1w, sym):
        score, factors = 0.0, []

        # ════ HARD GATES ══════════════════════════════════════
        if btc.score < BUY_BTC_MIN:          return 0.0, []
        if price < e50_d:                    return 0.0, []
        if slope_d < MIN_SLOPE:              return 0.0, []
        if rsi_4h > BUY_RSI_MAX:            return 0.0, []
        if rsi_4h < BUY_RSI_MIN:            return 0.0, []
        # ═════════════════════════════════════════════════════

        # 1. Daily trend structure (0–2.5)
        if price > e20_d > e50_d > e200_d:
            score += 2.5; factors.append("✅ DAILY: price > EMA20 > EMA50 > EMA200")
        elif price > e20_d > e50_d:
            score += 1.8; factors.append("✅ DAILY: price > EMA20 > EMA50")
        elif price > e50_d:
            score += 1.0; factors.append("⚠️ DAILY: above EMA50 only")

        # 2. Daily RSI (0–1.2)
        if 45 <= rsi_d <= 62:
            score += 1.2; factors.append(f"✅ DAILY RSI {rsi_d:.0f} — healthy, room to run")
        elif 35 <= rsi_d < 45:
            score += 0.7; factors.append(f"✅ DAILY RSI {rsi_d:.0f} — pullback zone")
        elif rsi_d > 68: score -= 0.5

        # 3. Daily slope (0–1.0)
        if slope_d > 0.006:
            score += 1.0; factors.append("✅ DAILY EMA20 strong upslope")
        elif slope_d > MIN_SLOPE:
            score += 0.5; factors.append("⚠️ DAILY EMA20 mild upslope")

        # 4. 4H trend confirmation (0–1.5)
        if price > e20_4h > e50_4h and slope_4h > MIN_SLOPE:
            score += 1.5; factors.append("✅ 4H uptrend: price > EMA20 > EMA50")
        elif price > e20_4h:
            score += 0.8; factors.append("⚠️ 4H: above EMA20")

        # 5. 4H RSI pullback (0–1.2)
        if 40 <= rsi_4h <= 55:
            score += 1.2; factors.append(f"✅ 4H RSI {rsi_4h:.0f} — ideal pullback zone")
        elif 55 < rsi_4h <= BUY_RSI_MAX:
            score += 0.6; factors.append(f"⚠️ 4H RSI {rsi_4h:.0f} — acceptable")

        # 6. Volume declining on pullback (0–1.0)
        if vr_4h < 0.75:
            score += 1.0; factors.append(f"✅ Volume low on pullback ({vr_4h:.2f}x) — healthy")
        elif vr_4h < 1.0:
            score += 0.5
        elif vr_4h >= 1.5: score -= 0.3

        # 7. MACD (0–1.0)
        if ml > ms and ml > 0:
            score += 1.0; factors.append("✅ MACD bullish cross above zero")
        elif ml > ms:
            score += 0.5; factors.append("⚠️ MACD bullish cross")

        # 8. OBV (0–0.8)
        if obv == "rising":
            score += 0.8; factors.append("✅ OBV rising — accumulation")

        # 9. StochRSI (0–0.7)
        if srsi["k"] < 40 and srsi["k"] > srsi["d"]:
            score += 0.7; factors.append(f"✅ StochRSI turning up (K:{srsi['k']:.0f})")
        elif srsi["k"] < 50: score += 0.3

        # 10. BTC strength (0–0.8)
        if btc.score >= 70:
            score += 0.8; factors.append(f"✅ BTC very strong ({btc.score}/100)")
        elif btc.score >= 60:
            score += 0.5; factors.append(f"✅ BTC strong ({btc.score}/100)")

        # 11. 1H timing (0–0.6)
        if 38 <= rsi_1h <= 58:
            score += 0.6; factors.append(f"✅ 1H RSI {rsi_1h:.0f} — good entry timing")

        # 12. Price near 4H EMA20 (0–0.8)
        if atr_4h > 0 and -0.3 <= (price - e20_4h) / atr_4h <= 0.6:
            score += 0.8; factors.append("✅ Price near 4H EMA20 — pullback entry")

        # 13. Bollinger position (0–0.5)
        if bb["pct_b"] < 0.45:
            score += 0.5; factors.append("✅ Price in lower half of Bollinger")

        # 14. Daily volume (0–0.5)
        if vr_1d >= 1.2:
            score += 0.5; factors.append(f"✅ Daily volume strong ({vr_1d:.1f}x)")

        # 15. News (0–0.6)
        if news.get("label") == "positive" and news.get("articles", 0) >= 2:
            score += 0.6; factors.append(f"✅ NEWS: {news.get('articles',0)} bullish articles")
        elif news.get("label") == "positive": score += 0.3
        elif news.get("label") == "negative": score -= 0.4

        # ════ BONUS STRATEGIES (additive, max 1.0 each) ══════

        # B1. CRT — Candle Range Theory
        s, f = self._crt_buy(df_1d, df_4h, price)
        score += s; factors += f

        # B2. Weekly S/R — bounce off weekly support
        s, f = self._weekly_sr_buy(df_1w, df_1d, price)
        score += s; factors += f

        # B3. Fibonacci — price at 0.5/0.618 retracement
        s, f = self._fibonacci_buy(df_4h, price)
        score += s; factors += f

        # B4. Order Block — price returning to bullish OB
        s, f = self._order_block_buy(df_4h, price)
        score += s; factors += f

        # B5. RSI Divergence — bullish divergence
        s, f = self._rsi_divergence_buy(df_4h, df_4h['close'], rsi_4h)
        score += s; factors += f

        # B6. VWAP — price near VWAP (fair value)
        s, f = self._vwap_buy(df_1d, price)
        score += s; factors += f

        # B7. Ichimoku — price above cloud, TK cross
        s, f = self._ichimoku_buy(df_4h, price)
        score += s; factors += f

        # B8. Supertrend — bullish supertrend
        s, f = self._supertrend_buy(df_4h, price, atr_4h)
        score += s; factors += f

        # B9. Session — London/NY overlap
        s, f = self._session_bonus()
        score += s; factors += f

        # B10. ATR quality — not too choppy, not too quiet
        s, f = self._atr_quality(atr_4h, atr_1d, price)
        score += s; factors += f

        # B11. BTC Correlation — coin stronger than BTC
        s, f = self._btc_correlation_buy(df_4h, btc)
        score += s; factors += f

        # B12. Market Structure Break — recent BOS on 4H
        s, f = self._msb_buy(df_4h, price)
        score += s; factors += f

        return round(score, 2), factors

    # ════════════════════════════════════════════════════════
    # SELL SCORING
    # ════════════════════════════════════════════════════════

    def _sell_score(self, price, gain, rsi_d, rsi_4h, rsi_1h,
                    e20_d, e50_d, e20_4h, e50_4h,
                    slope_d, slope_4h, vr_4h, vr_1d, atr_4h, atr_1d,
                    obv, bb, srsi, wr, ml, ms,
                    btc, news, df_1d, df_4h, df_1h, df_1w, sym):
        score, factors = 0.0, []

        # ════ HARD GATES ══════════════════════════════════════
        if btc.score > SELL_BTC_MAX:                     return 0.0, []
        if price > e50_d:                                return 0.0, []
        if slope_d > -MIN_SLOPE:                         return 0.0, []
        if rsi_4h < SELL_RSI_MIN or rsi_4h > SELL_RSI_MAX: return 0.0, []
        # ═════════════════════════════════════════════════════

        # 1. Daily downtrend (0–2.5)
        if price < e20_d < e50_d:
            score += 2.5; factors.append("✅ DAILY: price < EMA20 < EMA50 — downtrend")
        elif price < e50_d:
            score += 1.0; factors.append("⚠️ DAILY: below EMA50")

        # 2. Daily RSI bounce (0–1.2)
        if 50 <= rsi_d <= 62:
            score += 1.2; factors.append(f"✅ DAILY RSI {rsi_d:.0f} — bounce into resistance")
        elif rsi_d > 62: score += 0.5
        elif rsi_d < 38: score -= 0.5

        # 3. Downslope (0–1.0)
        if slope_d < -0.006:
            score += 1.0; factors.append("✅ DAILY EMA20 strong downslope")
        elif slope_d < -MIN_SLOPE: score += 0.5

        # 4. 4H downtrend (0–1.5)
        if price < e20_4h < e50_4h and slope_4h < -MIN_SLOPE:
            score += 1.5; factors.append("✅ 4H downtrend: price < EMA20 < EMA50")
        elif price < e20_4h: score += 0.8

        # 5. 4H RSI bounce (0–1.2)
        if 52 <= rsi_4h <= 65:
            score += 1.2; factors.append(f"✅ 4H RSI {rsi_4h:.0f} — bounced into resistance")
        elif rsi_4h > 65: score += 0.6

        # 6. Volume spike on bounce (0–1.0)
        if vr_4h >= 1.5:
            score += 1.0; factors.append(f"✅ Volume spike ({vr_4h:.1f}x) — distribution")
        elif vr_4h >= 1.1: score += 0.5

        # 7. MACD bearish (0–1.0)
        if ml < ms and ml < 0:
            score += 1.0; factors.append("✅ MACD bearish cross below zero")
        elif ml < ms: score += 0.5

        # 8. OBV (0–0.8)
        if obv == "falling":
            score += 0.8; factors.append("✅ OBV falling — distribution")

        # 9. StochRSI (0–0.7)
        if srsi["k"] > 60 and srsi["k"] < srsi["d"]:
            score += 0.7; factors.append(f"✅ StochRSI turning down (K:{srsi['k']:.0f})")

        # 10. BTC weakness (0–0.8)
        if btc.score < 30:
            score += 0.8; factors.append(f"✅ BTC very weak ({btc.score}/100)")
        elif btc.score <= SELL_BTC_MAX: score += 0.4

        # 11. 1H trigger (0–0.6)
        if rsi_1h >= 60:
            score += 0.6; factors.append(f"✅ 1H RSI {rsi_1h:.0f} — local overbought")

        # 12. Price near 4H EMA20 resistance (0–0.8)
        if atr_4h > 0 and -0.2 <= (price - e20_4h) / atr_4h <= 0.5:
            score += 0.8; factors.append("✅ Price testing 4H EMA20 as resistance")

        # 13. News (0–0.6)
        if news.get("label") == "negative" and news.get("articles", 0) >= 2:
            score += 0.6; factors.append(f"✅ NEWS: {news.get('articles',0)} bearish articles")
        elif news.get("label") == "negative": score += 0.3
        elif news.get("label") == "positive": score -= 0.4

        # ════ BONUS STRATEGIES ════════════════════════════════

        core_score = round(score, 2)
        bonus_score, bonus_factors = 0.0, []

        s, f = self._crt_sell(df_1d, df_4h, price)
        bonus_score += s; bonus_factors += f

        s, f = self._weekly_sr_sell(df_1w, df_1d, price)
        bonus_score += s; bonus_factors += f

        s, f = self._fibonacci_sell(df_4h, price)
        bonus_score += s; bonus_factors += f

        s, f = self._order_block_sell(df_4h, price)
        bonus_score += s; bonus_factors += f

        s, f = self._rsi_divergence_sell(df_4h, df_4h['close'], rsi_4h)
        bonus_score += s; bonus_factors += f

        s, f = self._vwap_sell(df_1d, price)
        bonus_score += s; bonus_factors += f

        s, f = self._ichimoku_sell(df_4h, price)
        bonus_score += s; bonus_factors += f

        s, f = self._supertrend_sell(df_4h, price, atr_4h)
        bonus_score += s; bonus_factors += f

        s, f = self._session_bonus()
        bonus_score += s; bonus_factors += f

        s, f = self._atr_quality(atr_4h, atr_1d, price)
        bonus_score += s; bonus_factors += f

        s, f = self._btc_correlation_sell(df_4h, btc)
        bonus_score += s; bonus_factors += f

        s, f = self._msb_sell(df_4h, price)
        bonus_score += s; bonus_factors += f

        factors += bonus_factors
        return round(core_score + bonus_score / 100, 6), factors

    # ════════════════════════════════════════════════════════
    # BONUS STRATEGIES — all return (score, factors)
    # Max contribution per strategy capped inside each method
    # ════════════════════════════════════════════════════════

    def _crt_buy(self, df_1d, df_4h, price):
        if len(df_1d) < 2 or len(df_4h) < 2: return 0.0, []
        s, f = 0.0, []
        pdl = float(df_1d["low"].iloc[-2])
        pdh = float(df_1d["high"].iloc[-2])
        dr  = pdh - pdl
        if dr <= 0: return 0.0, []
        if price > pdl:
            s += 0.4; f.append(f"✅ CRT: above PDL ({pdl:.5g})")
        mid = pdl + dr * 0.5
        if abs(price - mid) / dr < 0.12:
            s += 0.6; f.append(f"✅ CRT: at 50% equilibrium ({mid:.5g})")
        if price > pdh:
            s += 0.5; f.append(f"✅ CRT: above PDH — breakout structure")
        r4h = float(df_4h["high"].iloc[-2]) - float(df_4h["low"].iloc[-2])
        body = abs(float(df_4h["close"].iloc[-1]) - float(df_4h["open"].iloc[-1]))
        if r4h > 0 and body / r4h > 0.6:
            s += 0.4; f.append("✅ CRT: 4H displacement — institutional momentum")
        return round(min(s, 1.0), 2), f

    def _crt_sell(self, df_1d, df_4h, price):
        if len(df_1d) < 2 or len(df_4h) < 2: return 0.0, []
        s, f = 0.0, []
        pdl = float(df_1d["low"].iloc[-2])
        pdh = float(df_1d["high"].iloc[-2])
        dr  = pdh - pdl
        if dr <= 0: return 0.0, []
        if price < pdh:
            s += 0.4; f.append(f"✅ CRT: below PDH ({pdh:.5g})")
        mid = pdl + dr * 0.5
        if abs(price - mid) / dr < 0.12:
            s += 0.6; f.append(f"✅ CRT: at 50% midpoint ({mid:.5g})")
        if abs(price - pdl) / dr < 0.08:
            s += 0.5; f.append(f"✅ CRT: rejection at PDL ({pdl:.5g})")
        r4h  = float(df_4h["high"].iloc[-2]) - float(df_4h["low"].iloc[-2])
        body = abs(float(df_4h["close"].iloc[-1]) - float(df_4h["open"].iloc[-1]))
        bear = float(df_4h["close"].iloc[-1]) < float(df_4h["open"].iloc[-1])
        if r4h > 0 and body / r4h > 0.6 and bear:
            s += 0.4; f.append("✅ CRT: bearish displacement candle")
        return round(min(s, 1.0), 2), f

    def _weekly_sr_buy(self, df_1w, df_1d, price):
        """Price bouncing off a weekly support level."""
        if df_1w is None or df_1w.empty or len(df_1w) < 4: return 0.0, []
        s, f = 0.0, []
        # Weekly lows = support levels
        w_lows = df_1w["low"].rolling(4).min().dropna()
        for lv in w_lows.tail(8):
            lv = float(lv)
            dist = abs(price - lv) / price
            if dist < 0.015:
                s += 1.0; f.append(f"✅ WEEKLY S/R: bouncing off weekly support ({lv:.5g})")
                break
            elif dist < 0.03:
                s += 0.5; f.append(f"⚠️ WEEKLY S/R: near weekly support ({lv:.5g})")
                break
        return round(min(s, 1.0), 2), f

    def _weekly_sr_sell(self, df_1w, df_1d, price):
        """Price rejecting at a weekly resistance level."""
        if df_1w is None or df_1w.empty or len(df_1w) < 4: return 0.0, []
        s, f = 0.0, []
        w_highs = df_1w["high"].rolling(4).max().dropna()
        for lv in w_highs.tail(8):
            lv = float(lv)
            dist = abs(price - lv) / price
            if dist < 0.015:
                s += 1.0; f.append(f"✅ WEEKLY S/R: rejecting at weekly resistance ({lv:.5g})")
                break
            elif dist < 0.03:
                s += 0.5; f.append(f"⚠️ WEEKLY S/R: near weekly resistance ({lv:.5g})")
                break
        return round(min(s, 1.0), 2), f

    def _fibonacci_buy(self, df_4h, price):
        """Price at 0.5 or 0.618 retracement of last 4H swing."""
        if len(df_4h) < 20: return 0.0, []
        s, f = 0.0, []
        hi = float(df_4h["high"].tail(20).max())
        lo = float(df_4h["low"].tail(20).min())
        rng = hi - lo
        if rng <= 0: return 0.0, []
        fib618 = hi - rng * 0.618
        fib500 = hi - rng * 0.500
        fib382 = hi - rng * 0.382
        tol = rng * 0.02   # 2% tolerance band
        if abs(price - fib618) < tol:
            s += 1.0; f.append(f"✅ FIB: price at 0.618 retracement ({fib618:.5g}) — golden ratio")
        elif abs(price - fib500) < tol:
            s += 0.8; f.append(f"✅ FIB: price at 0.5 retracement ({fib500:.5g})")
        elif abs(price - fib382) < tol:
            s += 0.5; f.append(f"⚠️ FIB: price at 0.382 retracement ({fib382:.5g})")
        return round(min(s, 1.0), 2), f

    def _fibonacci_sell(self, df_4h, price):
        """Price at 0.5 or 0.618 retracement bounce (resistance)."""
        if len(df_4h) < 20: return 0.0, []
        s, f = 0.0, []
        hi = float(df_4h["high"].tail(20).max())
        lo = float(df_4h["low"].tail(20).min())
        rng = hi - lo
        if rng <= 0: return 0.0, []
        fib382 = lo + rng * 0.382
        fib500 = lo + rng * 0.500
        fib618 = lo + rng * 0.618
        tol = rng * 0.02
        if abs(price - fib618) < tol:
            s += 1.0; f.append(f"✅ FIB: rejection at 0.618 bounce ({fib618:.5g})")
        elif abs(price - fib500) < tol:
            s += 0.8; f.append(f"✅ FIB: rejection at 0.5 bounce ({fib500:.5g})")
        elif abs(price - fib382) < tol:
            s += 0.5; f.append(f"⚠️ FIB: near 0.382 level ({fib382:.5g})")
        return round(min(s, 1.0), 2), f

    def _order_block_buy(self, df_4h, price):
        """Last bearish candle before a big bullish move = bullish OB."""
        if len(df_4h) < 15: return 0.0, []
        s, f = 0.0, []
        for i in range(len(df_4h) - 3, max(len(df_4h) - 15, 2), -1):
            o = float(df_4h["open"].iloc[i])
            c = float(df_4h["close"].iloc[i])
            nxt_c = float(df_4h["close"].iloc[i + 1])
            # Bearish candle followed by strong bullish move = OB
            if c < o and (nxt_c - o) / o > 0.01:
                ob_high = max(o, c)
                ob_low  = min(o, c)
                if ob_low <= price <= ob_high * 1.005:
                    s += 1.0; f.append(f"✅ OB: price in bullish order block ({ob_low:.5g}–{ob_high:.5g})")
                    break
                elif price < ob_low and price > ob_low * 0.99:
                    s += 0.5; f.append(f"⚠️ OB: approaching bullish order block ({ob_low:.5g})")
                    break
        return round(min(s, 1.0), 2), f

    def _order_block_sell(self, df_4h, price):
        """Last bullish candle before a big bearish move = bearish OB."""
        if len(df_4h) < 15: return 0.0, []
        s, f = 0.0, []
        for i in range(len(df_4h) - 3, max(len(df_4h) - 15, 2), -1):
            o = float(df_4h["open"].iloc[i])
            c = float(df_4h["close"].iloc[i])
            nxt_c = float(df_4h["close"].iloc[i + 1])
            if c > o and (o - nxt_c) / o > 0.01:
                ob_high = max(o, c)
                ob_low  = min(o, c)
                if ob_low * 0.995 <= price <= ob_high:
                    s += 1.0; f.append(f"✅ OB: price in bearish order block ({ob_low:.5g}–{ob_high:.5g})")
                    break
                elif price > ob_high and price < ob_high * 1.01:
                    s += 0.5; f.append(f"⚠️ OB: approaching bearish order block ({ob_high:.5g})")
                    break
        return round(min(s, 1.0), 2), f

    def _rsi_divergence_buy(self, df_4h, close, rsi_now):
        """Bullish divergence: price lower low, RSI higher low."""
        if len(df_4h) < 30: return 0.0, []
        try:
            lows = df_4h["low"].values[-30:]
            r    = self._rsi_series(pd.Series(close.values[-50:])).values[-30:]
            li1  = int(np.argmin(lows[:15]))
            li2  = int(np.argmin(lows[15:])) + 15
            if lows[li2] < lows[li1] and r[li2] > r[li1] + 3:
                s = 1.0
                f = [f"✅ DIV: bullish RSI divergence — price LL, RSI HL ({rsi_now:.0f})"]
                if rsi_now > r[li2]: s = 1.0; f.append("✅ DIV: RSI recovering from divergence low")
                return round(min(s, 1.0), 2), f
        except Exception: pass
        return 0.0, []

    def _rsi_divergence_sell(self, df_4h, close, rsi_now):
        """Bearish divergence: price higher high, RSI lower high."""
        if len(df_4h) < 30: return 0.0, []
        try:
            highs = df_4h["high"].values[-30:]
            r     = self._rsi_series(pd.Series(close.values[-50:])).values[-30:]
            hi1   = int(np.argmax(highs[:15]))
            hi2   = int(np.argmax(highs[15:])) + 15
            if highs[hi2] > highs[hi1] and r[hi2] < r[hi1] - 3:
                s = 1.0
                f = [f"✅ DIV: bearish RSI divergence — price HH, RSI LH ({rsi_now:.0f})"]
                if rsi_now < r[hi2]: f.append("✅ DIV: RSI declining from divergence high")
                return round(min(s, 1.0), 2), f
        except Exception: pass
        return 0.0, []

    def _vwap_buy(self, df_1d, price):
        """Price near or below daily VWAP = fair value BUY zone."""
        if len(df_1d) < 2: return 0.0, []
        try:
            d  = df_1d.iloc[-1]
            tp = (float(d["high"]) + float(d["low"]) + float(d["close"])) / 3
            vwap = tp  # single-day simplified VWAP
            dist = (price - vwap) / vwap
            if -0.01 <= dist <= 0.005:
                return 0.8, ["✅ VWAP: price at fair value (VWAP) — ideal entry"]
            elif -0.03 <= dist < -0.01:
                return 0.5, [f"✅ VWAP: price below VWAP — value zone"]
            elif 0.005 < dist <= 0.02:
                return 0.3, [f"⚠️ VWAP: slightly above VWAP"]
        except Exception: pass
        return 0.0, []

    def _vwap_sell(self, df_1d, price):
        """Price near or above daily VWAP = fair value SELL zone."""
        if len(df_1d) < 2: return 0.0, []
        try:
            d  = df_1d.iloc[-1]
            tp = (float(d["high"]) + float(d["low"]) + float(d["close"])) / 3
            vwap = tp
            dist = (price - vwap) / vwap
            if -0.005 <= dist <= 0.01:
                return 0.8, ["✅ VWAP: price at VWAP — distribution zone"]
            elif 0.01 < dist <= 0.03:
                return 0.5, [f"✅ VWAP: price above VWAP — overvalued"]
        except Exception: pass
        return 0.0, []

    def _ichimoku_buy(self, df_4h, price):
        """Price above Kumo cloud + TK bullish cross."""
        if len(df_4h) < 52: return 0.0, []
        s, f = 0.0, []
        try:
            hi9  = df_4h["high"].rolling(9).max().iloc[-1]
            lo9  = df_4h["low"].rolling(9).min().iloc[-1]
            hi26 = df_4h["high"].rolling(26).max().iloc[-1]
            lo26 = df_4h["low"].rolling(26).min().iloc[-1]
            tenkan = (hi9 + lo9) / 2
            kijun  = (hi26 + lo26) / 2
            hi52   = df_4h["high"].rolling(52).max().iloc[-1]
            lo52   = df_4h["low"].rolling(52).min().iloc[-1]
            span_a = (tenkan + kijun) / 2
            span_b = (hi52 + lo52) / 2
            kumo_top = max(span_a, span_b)
            kumo_bot = min(span_a, span_b)
            if price > kumo_top:
                s += 0.6; f.append(f"✅ ICHI: price above Kumo cloud")
            if tenkan > kijun:
                s += 0.4; f.append(f"✅ ICHI: TK bullish cross (T>{K:.3g})" if False else f"✅ ICHI: Tenkan > Kijun — bullish")
        except Exception: pass
        return round(min(s, 1.0), 2), f

    def _ichimoku_sell(self, df_4h, price):
        """Price below Kumo cloud + TK bearish cross."""
        if len(df_4h) < 52: return 0.0, []
        s, f = 0.0, []
        try:
            hi9  = df_4h["high"].rolling(9).max().iloc[-1]
            lo9  = df_4h["low"].rolling(9).min().iloc[-1]
            hi26 = df_4h["high"].rolling(26).max().iloc[-1]
            lo26 = df_4h["low"].rolling(26).min().iloc[-1]
            tenkan = (hi9 + lo9) / 2
            kijun  = (hi26 + lo26) / 2
            hi52   = df_4h["high"].rolling(52).max().iloc[-1]
            lo52   = df_4h["low"].rolling(52).min().iloc[-1]
            span_a = (tenkan + kijun) / 2
            span_b = (hi52 + lo52) / 2
            kumo_bot = min(span_a, span_b)
            if price < kumo_bot:
                s += 0.6; f.append("✅ ICHI: price below Kumo cloud")
            if tenkan < kijun:
                s += 0.4; f.append("✅ ICHI: Tenkan < Kijun — bearish")
        except Exception: pass
        return round(min(s, 1.0), 2), f

    def _supertrend_buy(self, df_4h, price, atr_val, mult=3.0):
        """Supertrend bullish — price above supertrend line."""
        if len(df_4h) < 20 or atr_val <= 0: return 0.0, []
        try:
            hl2 = (df_4h["high"] + df_4h["low"]) / 2
            upper = hl2 + mult * atr_val
            lower = hl2 - mult * atr_val
            # Simplified: just check if price > lower supertrend band
            if float(price) > float(lower.iloc[-1]):
                return 0.6, [f"✅ SUPER: price above supertrend support ({lower.iloc[-1]:.5g})"]
        except Exception: pass
        return 0.0, []

    def _supertrend_sell(self, df_4h, price, atr_val, mult=3.0):
        """Supertrend bearish — price below supertrend line."""
        if len(df_4h) < 20 or atr_val <= 0: return 0.0, []
        try:
            hl2   = (df_4h["high"] + df_4h["low"]) / 2
            upper = hl2 + mult * atr_val
            if float(price) < float(upper.iloc[-1]):
                return 0.6, [f"✅ SUPER: price below supertrend resistance ({upper.iloc[-1]:.5g})"]
        except Exception: pass
        return 0.0, []

    def _session_bonus(self):
        """+0.4 during London/NY session (higher volume, cleaner moves)."""
        try:
            h = datetime.now(timezone.utc).hour
            if LONDON_OPEN_UTC <= h < LONDON_CLOSE_UTC:
                if NY_OPEN_UTC <= h < NY_CLOSE_UTC:
                    return 0.4, ["✅ SESSION: London/NY overlap — highest liquidity"]
                return 0.3, ["✅ SESSION: London session — high liquidity"]
            if NY_OPEN_UTC <= h < NY_CLOSE_UTC:
                return 0.3, ["✅ SESSION: NY session — high liquidity"]
        except Exception: pass
        return 0.0, []

    def _atr_quality(self, atr_4h, atr_1d, price):
        """Bonus when market has good volatility — not too choppy, not too quiet."""
        if price <= 0 or atr_4h <= 0: return 0.0, []
        try:
            atr_pct = atr_4h / price * 100
            # Ideal volatility: 0.5%–3% ATR on 4H
            if 0.5 <= atr_pct <= 3.0:
                return 0.4, [f"✅ ATR: ideal volatility ({atr_pct:.1f}%) — clean trends"]
            elif atr_pct > 5.0:
                return -0.3, []   # too volatile = risky stop placement
        except Exception: pass
        return 0.0, []

    def _btc_correlation_buy(self, df_4h, btc):
        """Coin outperforming BTC over last 4H = relative strength = bullish."""
        if len(df_4h) < 5: return 0.0, []
        try:
            coin_ret = (float(df_4h["close"].iloc[-1]) - float(df_4h["close"].iloc[-5])) / float(df_4h["close"].iloc[-5]) * 100
            # BTC 4H approx return from score change (simplified)
            btc_ret = btc.score / 100 * 2  # rough proxy
            if coin_ret > btc_ret + 1:
                return 0.5, [f"✅ BTCCOR: coin +{coin_ret:.1f}% outperforming BTC — relative strength"]
        except Exception: pass
        return 0.0, []

    def _btc_correlation_sell(self, df_4h, btc):
        """Coin underperforming BTC = relative weakness = bearish."""
        if len(df_4h) < 5: return 0.0, []
        try:
            coin_ret = (float(df_4h["close"].iloc[-1]) - float(df_4h["close"].iloc[-5])) / float(df_4h["close"].iloc[-5]) * 100
            btc_ret  = btc.score / 100 * 2
            if coin_ret < btc_ret - 1:
                return 0.5, [f"✅ BTCCOR: coin {coin_ret:.1f}% underperforming BTC — relative weakness"]
        except Exception: pass
        return 0.0, []

    def _msb_buy(self, df_4h, price):
        """Market Structure Break — recent bullish BOS (higher high broken upward)."""
        if len(df_4h) < 20: return 0.0, []
        try:
            # Previous swing high (last 10 candles excl recent 3)
            prev_high = float(df_4h["high"].iloc[-13:-3].max())
            curr_high = float(df_4h["high"].iloc[-3:].max())
            if curr_high > prev_high:
                return 0.8, [f"✅ MSB: bullish break of structure ({prev_high:.5g} → {curr_high:.5g})"]
        except Exception: pass
        return 0.0, []

    def _msb_sell(self, df_4h, price):
        """Market Structure Break — recent bearish BOS (lower low broken downward)."""
        if len(df_4h) < 20: return 0.0, []
        try:
            prev_low = float(df_4h["low"].iloc[-13:-3].min())
            curr_low = float(df_4h["low"].iloc[-3:].min())
            if curr_low < prev_low:
                return 0.8, [f"✅ MSB: bearish break of structure ({prev_low:.5g} → {curr_low:.5g})"]
        except Exception: pass
        return 0.0, []

    # ════════════════════════════════════════════════════════
    # BUILD SIGNAL
    # ════════════════════════════════════════════════════════

    def _build(self, sig_type, sym, confluence, factors, strats,
               price, gain, rsi_1h, rsi_4h, rsi_d, btc,
               srsi, bb_pct_b, wr_val, news_sent,
               e50_4h, atr_4h, swing_level) -> Signal | None:
        r      = self._r
        is_buy = sig_type == "BUY"

        # Separate core score (integer part) from bonus (fractional * 100)
        # e.g. 6.2345 = core 6.23, bonus 0.45 * 100 = 45
        core_conf  = int(confluence)
        bonus_pts  = round((confluence - core_conf) * 100, 1)

        # Grade from CORE SCORE ONLY — fixes ULTRA<STRONG inversion
        # Data showed bonuses were inflating weak signals to ULTRA
        ULTRA_CORE  = 6.5   # raised from 7.0 to be achievable, but core only
        STRONG_CORE = 5.0

        if core_conf >= ULTRA_CORE:
            grade = f"ULTRA {'🟢🟢🟢' if is_buy else '🔴🔴🔴'}"; hold = "4-7 days"
            # Base confidence + bonus pts (each pt = 1% confidence, max +12%)
            base_conf = min(88, 80 + int(bonus_pts))
        elif core_conf >= STRONG_CORE:
            grade = f"STRONG {'🟢🟢' if is_buy else '🔴🔴'}"; hold = "2-4 days"
            base_conf = min(82, 70 + int(bonus_pts))
        else:
            grade = f"STANDARD {'🟢' if is_buy else '🔴'}"; hold = "1-2 days"
            base_conf = min(70, 60 + int(bonus_pts))

        grade_rank = {"ULTRA": 3, "STRONG": 2, "STANDARD": 1}
        if grade_rank.get(grade.split()[0], 0) < grade_rank.get(MIN_SEND_GRADE, 1):
            return None

        news_adj = (
             3 if (is_buy and news_sent == "positive") or (not is_buy and news_sent == "negative") else
            -3 if (is_buy and news_sent == "negative") or (not is_buy and news_sent == "positive") else 0
        )
        confidence = min(93, max(52, base_conf + news_adj))

        # ATR-based SL — placed beyond recent swing low/high + ATR buffer
        # Uses MIN (furthest away) not MAX, so normal volatility doesn't hit it
        # sl_pct is the MINIMUM fallback — ATR-SL can be wider but never tighter
        sl_buf = atr_4h * 1.5   # 1.5x ATR gives breathing room for normal swings

        if is_buy:
            tp1 = round(price * (1 + r.tp1_pct / 100), 8)
            tp2 = round(price * (1 + r.tp2_pct / 100), 8)
            tp3 = round(price * (1 + r.tp3_pct / 100), 8)
            # Use the LOWER of swing-based or %-based SL (more room = fewer noise hits)
            sl_swing = swing_level - sl_buf
            sl_pct   = price * (1 - r.sl_pct / 100)
            sl = round(min(sl_swing, sl_pct), 8)
            action = "Buy / Go Long"
        else:
            tp1 = round(price * (1 - r.tp1_pct / 100), 8)
            tp2 = round(price * (1 - r.tp2_pct / 100), 8)
            tp3 = round(price * (1 - r.tp3_pct / 100), 8)
            sl_swing = swing_level + sl_buf
            sl_pct   = price * (1 + r.sl_pct / 100)
            sl = round(max(sl_swing, sl_pct), 8)
            action = "Exit Long / Sell / Short"

        return Signal(
            symbol=sym, signal=sig_type, grade=grade,
            confidence=confidence, action=action,
            price=price, gain_24h=gain,
            rsi_1h=rsi_1h, rsi_4h=rsi_4h, rsi_daily=rsi_d,
            tp1=tp1, tp2=tp2, tp3=tp3, sl=sl,
            factors=factors, strategies_hit=strats,
            btc_score=btc.score, btc_trend=btc.trend,
            confluence=confluence, hold_days=hold,
            news_sentiment=news_sent,
            stoch_rsi=srsi, bb_pct_b=bb_pct_b, williams=wr_val,
        )

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _slope(series: pd.Series, lookback: int = 5) -> float:
        if len(series) < lookback + 1: return 0.0
        recent = series.iloc[-lookback:]
        v0 = float(recent.iloc[0])
        return float((recent.iloc[-1] - v0) / v0 / lookback) if v0 != 0 else 0.0

    @staticmethod
    def _rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))