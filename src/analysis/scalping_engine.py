"""
src/analysis/scalping_engine.py – EMA Crossover + RSI Scalping Strategy
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ScalpSignal:
    symbol: str
    direction: str      # "BUY" or "SELL"
    price: float
    confidence: int     # 0-100
    tp1: float
    sl: float
    atr: float
    strategy: str = "EMA_Cross_RSI"


class ScalpingEngine:
    """
    Simple EMA crossover scalping with RSI filter.
    Designed for 1m or 2m charts.
    """

    def __init__(self, binance_client):
        self._b = binance_client

    def analyze(self, ticker: dict) -> ScalpSignal | None:
        sym = ticker.get("symbol")
        if not sym:
            return None

        # Use 2-minute candles for faster signals
        df = self._get_df(sym, "2m", 100)
        if df is None or len(df) < 50:
            return None

        price = float(df["close"].astype(float).iloc[-1])

        # Calculate EMAs
        close = df["close"].astype(float)
        ema5 = close.ewm(span=5, adjust=False).mean()
        ema13 = close.ewm(span=13, adjust=False).mean()

        # RSI
        rsi = self._rsi(df, 14)

        # ATR for SL
        atr = self._atr(df, 14)

        # Crossover detection (current vs previous)
        ema5_prev = float(ema5.iloc[-2])
        ema13_prev = float(ema13.iloc[-2])
        ema5_curr = float(ema5.iloc[-1])
        ema13_curr = float(ema13.iloc[-1])
        rsi_curr = float(rsi.iloc[-1])

        direction = None
        if ema5_prev <= ema13_prev and ema5_curr > ema13_curr and rsi_curr > 50:
            direction = "BUY"
        elif ema5_prev >= ema13_prev and ema5_curr < ema13_curr and rsi_curr < 50:
            direction = "SELL"
        else:
            return None

        # Confidence based on RSI strength
        if direction == "BUY":
            confidence = int(min(100, rsi_curr))
        else:
            confidence = int(min(100, 100 - rsi_curr))

        # TP and SL (tight, quick)
        sl_dist = atr * 1.5
        tp_dist = atr * 1.0   # 1:1 risk-reward

        if direction == "BUY":
            sl = price - sl_dist
            tp1 = price + tp_dist
        else:
            sl = price + sl_dist
            tp1 = price - tp_dist

        log.debug("Scalp signal: %s %s @ %.6g (conf=%d)", sym, direction, price, confidence)

        return ScalpSignal(
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

    def _rsi(self, df, period=14) -> pd.Series:
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