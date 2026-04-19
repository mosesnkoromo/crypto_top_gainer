"""
src/analysis/vwap_scalping_engine.py – VWAP + RSI(3) + EMA(8) Scalping Strategy
"""
from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import Optional

from src.utils.logger import get_logger
from src.analysis.indicators import vwap

log = get_logger(__name__)


@dataclass
class VWAPScalpSignal:
    symbol: str
    direction: str      # "BUY" or "SELL"
    price: float
    confidence: int     # 0-100
    tp1: float
    sl: float
    atr: float
    strategy: str = "VWAP_RSI_EMA"


class VWAPScalpingEngine:
    """
    VWAP + RSI(3) + EMA(8) scalping.
    Designed for 1m or 2m charts.
    """

    def __init__(self, binance_client):
        self._b = binance_client

    def analyze(self, ticker: dict) -> VWAPScalpSignal | None:
        sym = ticker.get("symbol")
        if not sym:
            return None

        # Use 2-minute candles
        df = self._get_df(sym, "2m", 100)
        if df is None or len(df) < 50:
            return None

        price = float(df["close"].astype(float).iloc[-1])

        # Calculate indicators
        close = df["close"].astype(float)
        ema8 = close.ewm(span=8, adjust=False).mean()
        vwap_series = vwap(df)
        rsi3 = self._rsi(df, 3)

        # Current values
        ema8_curr = float(ema8.iloc[-1])
        vwap_curr = float(vwap_series.iloc[-1])
        rsi3_curr = float(rsi3.iloc[-1])

        # ATR for SL
        atr = self._atr(df, 14)

        # Overextension filter
        vwap_distance_pct = abs(price - vwap_curr) / vwap_curr * 100
        if vwap_distance_pct > 1.5:
            log.debug("  ⏭️  %s: overextended from VWAP (%.2f%%) — skip", sym, vwap_distance_pct)
            return None

        direction = None

        # Long signal
        if price > vwap_curr and price > ema8_curr and rsi3_curr < 30:
            direction = "BUY"
            confidence = int(min(100, 100 - rsi3_curr))
        # Short signal
        elif price < vwap_curr and price < ema8_curr and rsi3_curr > 70:
            direction = "SELL"
            confidence = int(min(100, rsi3_curr))
        else:
            return None

        # TP and SL (tight for scalping)
        sl_dist = atr * 1.2
        tp_dist = atr * 1.0   # 1:1 risk-reward

        if direction == "BUY":
            sl = price - sl_dist
            tp1 = price + tp_dist
        else:
            sl = price + sl_dist
            tp1 = price - tp_dist

        # Enforce minimum SL of 0.5%
        min_sl_dist = price * 0.005
        if abs(price - sl) < min_sl_dist:
            if direction == "BUY":
                sl = price - min_sl_dist
            else:
                sl = price + min_sl_dist

        log.debug("VWAP Scalp signal: %s %s @ %.6g (conf=%d)", sym, direction, price, confidence)

        return VWAPScalpSignal(
            symbol=sym,
            direction=direction,
            price=price,
            confidence=confidence,
            tp1=tp1,
            sl=sl,
            atr=atr,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_df(self, sym, interval, limit):
        try:
            df = self._b.get_klines(sym, interval, limit)
            if df is None or df.empty:
                return None
            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception:
            return None

    def _rsi(self, df, period=3) -> pd.Series:
        close = df["close"].astype(float)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))

    def _atr(self, df, period=14) -> float:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])