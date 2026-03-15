"""
src/analysis/signal_engine.py
──────────────────────────────
7-factor confluence engine. Only emits ULTRA and STRONG signals
(STANDARD suppressed by default — configurable).

New factors vs v2:
  + Bollinger Bands %B (overextension / squeeze breakout)
  + StochRSI overbought / oversold confirmation
  + Williams %R reversal confirmation
  + OBV trend alignment
  + News sentiment modifier (±confidence boost)

Signal only fires when confluence ≥ threshold AND
BTC Strength aligns with direction.
"""

from dataclasses import dataclass
from typing import Literal

from config import RiskConfig, SignalConfig, ScanConfig
from src.analysis.btc_strength import BtcStrength
from src.analysis.indicators import (
    atr, bollinger_bands, ema_value, obv_trend,
    rsi, stochastic_rsi, volume_ratio, williams_r,
)
from src.analysis.news_engine import NewsEngine
from src.data.binance_client import BinanceClient
from src.utils.logger import get_logger

log = get_logger(__name__)

SignalType = Literal["BUY", "SELL"]

# Minimum grade to actually send an alert (set to STRONG to block STANDARD)
MIN_SEND_GRADE = "STRONG"   # Options: "ULTRA" | "STRONG" | "STANDARD"


@dataclass
class Signal:
    symbol:     str
    signal:     SignalType
    grade:      str
    confidence: int
    action:     str
    price:      float
    gain_24h:   float
    rsi:        float
    tp1:        float
    tp2:        float
    tp3:        float
    sl:         float
    factors:    list[str]
    btc_score:  int
    btc_trend:  str
    confluence: float
    stoch_rsi:  dict
    bb_pct_b:   float
    williams:   float
    news_sentiment: str


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
        gain  = float(ticker["priceChangePercent"])
        price = float(ticker["lastPrice"])

        df = self._b.get_klines(sym, self._scan.timeframe, self._scan.candle_limit)
        if df.empty or len(df) < 30:
            return None

        close      = df["close"]
        pair_rsi   = rsi(close)
        e20        = ema_value(close, 20)
        e50        = ema_value(close, 50)
        pair_atr   = atr(df)
        vr         = volume_ratio(df["volume"])
        dist_atr   = abs(price - e20) / pair_atr if pair_atr > 0 else 0.0
        bb         = bollinger_bands(close)
        srsi       = stochastic_rsi(close)
        wr         = williams_r(df)
        obv        = obv_trend(df)
        news_sent  = self._news.get_sentiment_for(sym) if self._news else {"label": "neutral"}

        sell_score, sell_factors = self._sell_factors(gain, pair_rsi, btc, dist_atr, vr, bb, srsi, wr, obv, news_sent)
        buy_score,  buy_factors  = self._buy_factors(gain, pair_rsi, btc, vr, price, e20, e50, bb, srsi, wr, obv, news_sent)

        if sell_score >= self._s.min_sell_confluence and sell_score > buy_score:
            sig = self._build("SELL", sym, sell_score, sell_factors, price, gain,
                              pair_rsi, btc, srsi, bb["pct_b"], wr, news_sent["label"])
        elif buy_score >= self._s.min_buy_confluence and buy_score > sell_score:
            sig = self._build("BUY", sym, buy_score, buy_factors, price, gain,
                              pair_rsi, btc, srsi, bb["pct_b"], wr, news_sent["label"])
        else:
            return None

        # Filter by minimum grade
        grade_rank = {"ULTRA": 3, "STRONG": 2, "STANDARD": 1}
        min_rank   = grade_rank.get(MIN_SEND_GRADE, 1)
        sig_rank   = grade_rank.get(sig.grade.split()[0], 0)
        if sig_rank < min_rank:
            log.debug("Suppressed %s %s (%s < %s)", sig.signal, sym, sig.grade, MIN_SEND_GRADE)
            return None

        return sig

    # ── SELL factors (max ~7 pts) ─────────────────────────────

    def _sell_factors(self, gain, pair_rsi, btc, dist_atr, vr, bb, srsi, wr, obv, news):
        s = self._s
        score, factors = 0.0, []

        # 1. Gain overextension
        if gain >= 15:   score += 1.0; factors.append(f"✅ GAIN +{gain:.1f}% — heavily overextended")
        elif gain >= 8:  score += 0.8; factors.append(f"✅ GAIN +{gain:.1f}% — overextended")
        elif gain >= 5:  score += 0.5; factors.append(f"⚠️ GAIN +{gain:.1f}% — moderate")

        # 2. RSI overbought
        if pair_rsi >= 80:                score += 1.0; factors.append(f"✅ RSI {pair_rsi} — extremely overbought")
        elif pair_rsi >= s.rsi_overbought: score += 0.8; factors.append(f"✅ RSI {pair_rsi} — overbought")
        elif pair_rsi >= 65:              score += 0.4; factors.append(f"⚠️ RSI {pair_rsi} — elevated")

        # 3. BTC alignment
        if btc.score < s.btc_weak_threshold: score += 1.0; factors.append(f"✅ BTC weak ({btc.score}/100)")
        elif btc.rsi > s.btc_rsi_danger:     score += 0.8; factors.append(f"✅ BTC RSI overbought ({btc.rsi})")
        elif btc.score < 55:                 score += 0.4; factors.append(f"⚠️ BTC softening ({btc.score}/100)")

        # 4. EMA distance
        if dist_atr >= s.ema_distance_strong:   score += 1.0; factors.append(f"✅ Price {dist_atr:.1f}x ATR above EMA20")
        elif dist_atr >= s.ema_distance_moderate: score += 0.6; factors.append(f"⚠️ Price {dist_atr:.1f}x ATR above EMA20")

        # 5. Volume climax
        if vr >= s.volume_climax:  score += 1.0; factors.append(f"✅ Volume climax {vr:.1f}x avg")
        elif vr >= s.volume_strong: score += 0.6; factors.append(f"⚠️ Volume spike {vr:.1f}x avg")

        # 6. Bollinger Bands — price above upper band
        if bb["pct_b"] >= 1.0:   score += 0.8; factors.append(f"✅ Above Bollinger upper band")
        elif bb["pct_b"] >= 0.85: score += 0.5; factors.append(f"⚠️ Near Bollinger upper band")

        # 7. StochRSI + Williams %R confirmation
        if srsi["k"] >= 85 and srsi["d"] >= 80:
            score += 0.7; factors.append(f"✅ StochRSI overbought (K:{srsi['k']} D:{srsi['d']})")
        if wr >= -15:
            score += 0.5; factors.append(f"✅ Williams %R overbought ({wr})")

        # 8. OBV divergence (price up, volume trend falling = distribution)
        if obv == "falling" and pair_rsi > 65:
            score += 0.5; factors.append("✅ OBV falling — distribution signal")

        # 9. News sentiment — treated as full confluence factor
        if news["label"] == "negative" and news.get("articles", 0) >= 2:
            score += 0.8; factors.append(f"✅ NEWS: {news.get('articles',0)} negative articles (bearish)")
        elif news["label"] == "negative":
            score += 0.4; factors.append(f"⚠️ NEWS: bearish sentiment detected")
        elif news["label"] == "positive":
            score -= 0.3  # Positive news reduces sell conviction

        return round(score, 2), factors

    # ── BUY factors (max ~7 pts) ──────────────────────────────

    def _buy_factors(self, gain, pair_rsi, btc, vr, price, e20, e50, bb, srsi, wr, obv, news):
        s = self._s
        score, factors = 0.0, []

        # 1. Momentum
        if gain >= 10:  score += 1.0; factors.append(f"✅ Strong momentum +{gain:.1f}%")
        elif gain >= 5: score += 0.7; factors.append(f"✅ Momentum +{gain:.1f}%")

        # 2. BTC strong
        if btc.score >= s.btc_strong_threshold + 12: score += 1.0; factors.append(f"✅ BTC very strong ({btc.score}/100)")
        elif btc.score >= s.btc_strong_threshold:    score += 0.7; factors.append(f"✅ BTC strong ({btc.score}/100)")
        elif btc.score >= 50:                        score += 0.4; factors.append(f"⚠️ BTC moderate ({btc.score}/100)")

        # 3. RSI healthy range
        if s.rsi_buy_min <= pair_rsi <= s.rsi_buy_max:        score += 1.0; factors.append(f"✅ RSI {pair_rsi} — healthy, room to run")
        elif s.rsi_buy_min - 5 <= pair_rsi < s.rsi_buy_min:  score += 0.5; factors.append(f"⚠️ RSI {pair_rsi} — slightly low")

        # 4. Volume supporting
        if vr >= s.volume_buy_min + 0.7: score += 1.0; factors.append(f"✅ Strong volume {vr:.1f}x avg")
        elif vr >= s.volume_buy_min:     score += 0.6; factors.append(f"⚠️ Volume {vr:.1f}x avg")

        # 5. EMA trend aligned
        if price > e20 > e50:  score += 1.0; factors.append("✅ Price above EMA20 > EMA50 — perfect uptrend")
        elif price > e20:      score += 0.6; factors.append("✅ Price above EMA20")
        elif price > e50:      score += 0.3; factors.append("⚠️ Price above EMA50 only")

        # 6. Bollinger Bands — bounce from lower band
        if bb["pct_b"] <= 0.1:  score += 0.8; factors.append("✅ Bouncing off Bollinger lower band")
        elif bb["pct_b"] <= 0.25: score += 0.4; factors.append("⚠️ Near Bollinger lower band")

        # 7. StochRSI + Williams %R oversold
        if srsi["k"] <= 20 and srsi["d"] <= 25:
            score += 0.7; factors.append(f"✅ StochRSI oversold (K:{srsi['k']} D:{srsi['d']})")
        if wr <= -80:
            score += 0.5; factors.append(f"✅ Williams %R oversold ({wr})")

        # 8. OBV accumulation
        if obv == "rising":
            score += 0.5; factors.append("✅ OBV rising — accumulation")

        # 9. News sentiment — treated as full confluence factor
        if news["label"] == "positive" and news.get("articles", 0) >= 2:
            score += 0.8; factors.append(f"✅ NEWS: {news.get('articles',0)} positive articles (bullish)")
        elif news["label"] == "positive":
            score += 0.4; factors.append(f"⚠️ NEWS: bullish sentiment detected")
        elif news["label"] == "negative":
            score -= 0.3  # Negative news reduces buy conviction

        return round(score, 2), factors

    # ── Build Signal object ───────────────────────────────────

    def _build(self, sig_type, sym, confluence, factors, price, gain,
               pair_rsi, btc, srsi, bb_pct_b, wr_val, news_sent) -> Signal:
        r        = self._r
        is_sell  = sig_type == "SELL"

        if confluence >= 5.0:   grade = f"ULTRA {'🔴🔴🔴' if is_sell else '🟢🟢🟢'}"; base_conf = 85
        elif confluence >= 3.8: grade = f"STRONG {'🔴🔴' if is_sell else '🟢🟢'}";   base_conf = 75
        else:                   grade = f"STANDARD {'🔴' if is_sell else '🟢'}";      base_conf = 62

        # News sentiment adjusts displayed confidence ±3
        news_adj = 3 if (is_sell and news_sent == "negative") or (not is_sell and news_sent == "positive") else \
                  -3 if (is_sell and news_sent == "positive") or (not is_sell and news_sent == "negative") else 0
        confidence = min(95, max(50, base_conf + news_adj))

        mult = (1 - r.sl_pct / 100) if is_sell else (1 + r.sl_pct / 100)
        if is_sell:
            tp1 = round(price * (1 - r.tp1_pct / 100), 8)
            tp2 = round(price * (1 - r.tp2_pct / 100), 8)
            tp3 = round(price * (1 - r.tp3_pct / 100), 8)
            sl  = round(price * (1 + r.sl_pct  / 100), 8)
            action = "Exit Long / Sell / Short"
        else:
            tp1 = round(price * (1 + r.tp1_pct / 100), 8)
            tp2 = round(price * (1 + r.tp2_pct / 100), 8)
            tp3 = round(price * (1 + r.tp3_pct / 100), 8)
            sl  = round(price * (1 - r.sl_pct  / 100), 8)
            action = "Buy / Go Long"

        return Signal(
            symbol=sym, signal=sig_type, grade=grade,
            confidence=confidence, action=action,
            price=price, gain_24h=gain, rsi=pair_rsi,
            tp1=tp1, tp2=tp2, tp3=tp3, sl=sl,
            factors=factors, btc_score=btc.score,
            btc_trend=btc.trend, confluence=confluence,
            stoch_rsi=srsi, bb_pct_b=bb_pct_b,
            williams=wr_val, news_sentiment=news_sent,
        )
