"""
src/analysis/indicators.py
───────────────────────────
Pure, stateless technical indicator functions.
"""

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1: return 50.0
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    val   = 100.0 - (100.0 / (1.0 + rs))
    last  = val.iloc[-1]
    return round(float(last), 1) if not np.isnan(last) else 50.0


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def ema_value(series: pd.Series, period: int) -> float:
    return float(ema(series, period).iloc[-1])


def atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1: return 0.0
    h, l, c = df["high"], df["low"], df["close"]
    tr  = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return float(val) if not np.isnan(val) else 0.0


def macd(series: pd.Series, fast=12, slow=26, signal=9) -> tuple[float, float]:
    m   = ema(series, fast) - ema(series, slow)
    sig = ema(m, signal)
    return float(m.iloc[-1]), float(sig.iloc[-1])


def volume_ratio(volume_series: pd.Series, lookback: int = 20) -> float:
    if len(volume_series) < lookback: return 1.0
    avg = float(volume_series.rolling(lookback).mean().iloc[-1])
    cur = float(volume_series.iloc[-1])
    return round(cur / avg, 2) if avg > 0 else 1.0


def bollinger_bands(series: pd.Series, period: int = 20, std: float = 2.0) -> dict:
    """Returns upper, middle, lower bands and %B position (0=lower, 1=upper)."""
    if len(series) < period:
        mid = float(series.iloc[-1])
        return {"upper": mid, "mid": mid, "lower": mid, "pct_b": 0.5, "squeeze": False}
    rolling = series.rolling(period)
    mid    = float(rolling.mean().iloc[-1])
    sigma  = float(rolling.std().iloc[-1])
    upper  = mid + std * sigma
    lower  = mid - std * sigma
    price  = float(series.iloc[-1])
    band_w = upper - lower
    pct_b  = (price - lower) / band_w if band_w > 0 else 0.5
    # Squeeze = bands very narrow (< 3% of price)
    squeeze = band_w < price * 0.03
    return {
        "upper":   round(upper, 8),
        "mid":     round(mid,   8),
        "lower":   round(lower, 8),
        "pct_b":   round(pct_b, 3),
        "squeeze": squeeze,
    }


def stochastic_rsi(series: pd.Series, rsi_period=14, stoch_period=14) -> dict:
    """StochRSI — %K and %D. Values 0-100. >80=overbought, <20=oversold."""
    if len(series) < rsi_period + stoch_period:
        return {"k": 50.0, "d": 50.0}
    # Calculate RSI series
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(rsi_period).mean()
    loss  = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi_s = 100 - (100 / (1 + rs))
    # Stochastic on RSI
    low_rsi  = rsi_s.rolling(stoch_period).min()
    high_rsi = rsi_s.rolling(stoch_period).max()
    diff     = (high_rsi - low_rsi).replace(0, np.nan)
    k        = ((rsi_s - low_rsi) / diff) * 100
    d        = k.rolling(3).mean()
    k_val = float(k.iloc[-1]) if not np.isnan(k.iloc[-1]) else 50.0
    d_val = float(d.iloc[-1]) if not np.isnan(d.iloc[-1]) else 50.0
    return {"k": round(k_val, 1), "d": round(d_val, 1)}


def williams_r(df: pd.DataFrame, period: int = 14) -> float:
    """Williams %R. Range -100 to 0. Above -20=overbought, below -80=oversold."""
    if len(df) < period: return -50.0
    high  = df["high"].rolling(period).max()
    low   = df["low"].rolling(period).min()
    close = df["close"]
    diff  = (high - low).replace(0, np.nan)
    wr    = ((high - close) / diff) * -100
    val   = float(wr.iloc[-1])
    return round(val, 1) if not np.isnan(val) else -50.0


def obv_trend(df: pd.DataFrame, period: int = 10) -> str:
    """On-Balance Volume trend: 'rising', 'falling', or 'flat'."""
    if len(df) < period + 2: return "flat"
    close = df["close"]
    vol   = df["volume"]
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv   = (direction * vol).cumsum()
    recent = obv.iloc[-period:]
    slope  = np.polyfit(range(len(recent)), recent.values, 1)[0]
    if slope > 0: return "rising"
    if slope < 0: return "falling"
    return "flat"
