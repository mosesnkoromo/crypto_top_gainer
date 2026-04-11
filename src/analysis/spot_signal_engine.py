"""
src/analysis/spot_signal_engine.py — 5m Scalp Spot Engine
───────────────────────────────────────────────────────────
BUY-only spot scalping on 5m charts.
Same signal logic as futures but LONG-only.
No leverage — you own the coin.
OCO orders: TP1 (0.6%) + SL stop-limit.
Hold: 5-15 minutes max.

Entry gates (all must pass):
  1. EMA9 > EMA21 on 5m (bullish momentum)
  2. RSI on 5m: 28–65 (not overbought)
  3. MACD bullish on 5m
  4. Volume spike 1.2x OR bullish candle pattern
  5. 4H EMA50 macro bull (price above it)
  6. ATR(5m) ≥ 0.2% (market moving)
  7. RSI not in chop zone 48-52
"""

from dataclasses import dataclass
from typing import Literal
import pandas as pd

from src.data.binance_client import BinanceClient
from src.analysis.btc_strength import BtcStrength
from src.analysis.indicators import atr, ema_value, ema, macd, rsi, volume_ratio
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SpotSignal:
    symbol:      str
    signal:      Literal["BUY"]
    grade:       str
    confidence:  int
    action:      str
    price:       float
    gain_24h:    float
    tp1:         float
    tp2:         float
    tp3:         float
    sl:          float
    atr:         float
    factors:     list
    btc_score:   int
    hold_time:   str = "5-15 min"


class SpotSignalEngine:

    def __init__(self, binance: BinanceClient, btc_threshold: int = 25):
        self._b         = binance
        self._btc_min   = btc_threshold

    def analyze(self, ticker: dict, btc: BtcStrength) -> SpotSignal | None:
        """5m scalp BUY-only signal for spot trading."""
        sym   = ticker["symbol"]
        gain  = float(ticker.get("priceChangePercent", 0))
        price = float(ticker.get("lastPrice", 0) or ticker.get("current_price", 0))
        if price <= 0:
            return None

        # BTC gate: don't buy spot in extreme collapse
        if btc.score < self._btc_min:
            return None

        # Fetch candles
        df_5m = self._b.get_klines(sym, "5m", 60)
        df_4h = self._b.get_klines(sym, "4h", 50)

        if df_5m.empty or len(df_5m) < 30:
            return None

        close_5m  = df_5m["close"]
        close_4h  = df_4h["close"] if not df_4h.empty and len(df_4h) >= 20 else close_5m
        price_now = float(close_5m.iloc[-1])

        # Gate 1: Volatility — ATR ≥ 0.2%
        atr_5m_val = atr(df_5m, 14)
        atr_pct    = (atr_5m_val / price_now * 100) if price_now > 0 else 0
        if 0 < atr_pct < 0.2:
            return None

        # Gate 2: 4H macro bull (price above EMA50)
        e50_4h = ema_value(close_4h, 50) if len(close_4h) >= 50 else price_now
        if price_now <= e50_4h:
            return None  # macro bearish — no spot buy

        # Gate 3: EMA 9 > EMA 21 on 5m
        e9  = ema_value(close_5m, 9)
        e21 = ema_value(close_5m, 21)
        if e9 <= e21:
            return None  # not bullish momentum

        # Gate 4: RSI on 5m in buy zone
        rsi_5m = rsi(close_5m)
        if not (28 <= rsi_5m <= 65):
            return None
        if 48 < rsi_5m < 52:
            return None  # chop zone

        # Gate 5: MACD bullish on 5m
        ml_5m, ms_5m = macd(close_5m)
        if ml_5m <= ms_5m:
            return None

        # Gate 6: Volume OR price action pattern
        vr_5m = volume_ratio(df_5m["volume"], 20)
        # Quick bullish candle check
        last  = df_5m.iloc[-1]
        o5, h5, l5, c5 = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
        body  = abs(c5 - o5)
        wick  = l5 and (min(c5,o5) - l5)
        pa_bull = (c5 > o5 and wick >= body * 1.5) or (  # hammer
                   len(df_5m) > 1 and c5 > o5 and          # engulfing
                   float(df_5m.iloc[-2]["close"]) < float(df_5m.iloc[-2]["open"]) and
                   c5 > float(df_5m.iloc[-2]["open"]) and o5 < float(df_5m.iloc[-2]["close"]))
        if vr_5m < 1.2 and not pa_bull:
            return None  # no volume AND no pattern

        # Build signal
        factors = [
            f"✅ 5m EMA9={e9:.5g} > EMA21={e21:.5g}",
            f"✅ 5m RSI={rsi_5m:.0f} (buy zone 28–65)",
            f"✅ 5m MACD bullish",
            f"✅ 4H macro bull (price > EMA50={e50_4h:.5g})",
            f"✅ ATR={atr_pct:.1f}% (active market)",
        ]
        if vr_5m >= 1.2:
            factors.append(f"✅ Volume {vr_5m:.1f}x")
        if pa_bull:
            factors.append("✅ Bullish candle pattern")

        # Grade by confluence count
        score = sum([
            vr_5m >= 1.3,
            pa_bull,
            rsi_5m < 50,        # room to run
            atr_pct >= 0.5,     # strong volatility
            btc.score >= 55,    # BTC supportive
        ])
        if score >= 4:
            grade, conf = "STRONG 🟢🟢", 72
        elif score >= 2:
            grade, conf = "STANDARD 🟢", 62
        else:
            grade, conf = "STANDARD 🟢", 57

        # TP/SL levels (same as futures scalp)
        tp1 = round(price_now * 1.006, 8)   # 0.6%
        tp2 = round(price_now * 1.012, 8)   # 1.2%
        tp3 = round(price_now * 1.020, 8)   # 2.0%
        sl  = round(price_now - atr_5m_val * 1.2, 8) if atr_5m_val > 0               else round(price_now * 0.994, 8)        # ATR-based or 0.6%

        return SpotSignal(
            symbol=sym, signal="BUY", grade=grade,
            confidence=conf, action="Spot BUY · OCO TP1+SL",
            price=price_now, gain_24h=gain,
            tp1=tp1, tp2=tp2, tp3=tp3, sl=sl,
            atr=atr_5m_val, factors=factors,
            btc_score=btc.score,
        )