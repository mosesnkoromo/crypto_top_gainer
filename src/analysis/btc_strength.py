"""
src/analysis/btc_strength.py — BTC Strength Engine v4.2
════════════════════════════════════════════════════════
Full top-down HTF analysis on BTC itself:
  1W → 1D → 4H → 1H cascade (EMA21 vs EMA50 per timeframe)
  RSI on 1H as fine sentiment meter
  ADX on 4H as trend strength

Score 0–100:
  0–29   VERY WEAK BEAR 🔴🔴
  30–44  BEAR 🔴
  45–55  NEUTRAL ⚪
  56–69  BULL 🟢
  70–84  STRONG BULL 🟢🟢
  85–100 VERY STRONG BULL 🟢🟢🟢

Signal engine uses this score:
  BUY  only when score ≥ 30
  SELL only when score ≤ 70
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class BtcStrength:
    score: int       # 0–100
    trend: str       # label e.g. "VERY STRONG BULL 🟢🟢🟢"
    rsi:   float     # RSI-1H value
    price: float     # current BTC price
    # HTF details
    htf_score:    int    = 0      # raw HTF vote sum (-9 to +9)
    htf_labels:   str   = ""     # e.g. "1W↑ 1D↑ 4H↑ 1H↑"
    adx_4h:       float = 20.0   # ADX on 4H
    is_bull:      bool  = False  # score >= 56
    is_bear:      bool  = False  # score <= 44

    def to_dict(self) -> dict:
        return {
            "score":      self.score,
            "trend":      self.trend,
            "rsi":        round(self.rsi, 1),
            "price":      self.price,
            "htf_score":  self.htf_score,
            "htf_labels": self.htf_labels,
            "adx_4h":     self.adx_4h,
            "is_bull":    self.is_bull,
            "is_bear":    self.is_bear,
        }


class BtcStrengthEngine:
    """
    Calculates BTC market strength using multi-timeframe top-down analysis.
    Replaces the old RSI-only approach.
    """

    SYMBOL = "BTCUSDT"

    def __init__(self, binance_client, scan_cfg=None):
        self._b   = binance_client
        self._cfg = scan_cfg

    def calculate(self) -> BtcStrength:
        try:
            # Fetch BTC candles across all timeframes
            df_1w = self._get_df("1w",  20)
            df_1d = self._get_df("1d",  60)
            df_4h = self._get_df("4h",  60)
            df_1h = self._get_df("1h",  80)

            # Current price
            price = 0.0
            if df_1h is not None and not df_1h.empty:
                price = float(df_1h["close"].astype(float).iloc[-1])

            # ── 1H RSI ────────────────────────────────────────────────
            rsi_1h = self._rsi(df_1h, 14) if df_1h is not None else 50.0

            # ── Top-down HTF votes ────────────────────────────────────
            #   1W = weight 3, 1D = weight 3, 4H = weight 2, 1H = weight 1
            s1w, l1w = self._tf_vote(df_1w, "1W", 3)
            s1d, l1d = self._tf_vote(df_1d, "1D", 3)
            s4h, l4h = self._tf_vote(df_4h, "4H", 2)
            s1h, l1h = self._tf_vote(df_1h, "1H", 1)

            htf_raw   = s1w + s1d + s4h + s1h    # range -9 to +9
            htf_str   = f"{l1w} {l1d} {l4h} {l1h}"

            # ── ADX 4H (trend strength) ───────────────────────────────
            adx_4h = self._adx(df_4h, 14) if df_4h is not None else 20.0

            # ── Convert to 0-100 score ────────────────────────────────
            # HTF raw (-9 to +9) → base 0-100
            # htf_raw = +9 → base = 82, +0 = 50, -9 = 18
            htf_base = int(50 + htf_raw * 3.5)   # each point ≈ 3.5

            # RSI adjustment: pull score toward RSI sentiment
            # RSI 70+ = +5, RSI 30- = -5, otherwise proportional
            if   rsi_1h >= 70: rsi_adj = +6
            elif rsi_1h >= 55: rsi_adj = +3
            elif rsi_1h >= 45: rsi_adj =  0
            elif rsi_1h >= 30: rsi_adj = -3
            else:              rsi_adj = -6

            # ADX boost: strong trend (ADX>30) increases confidence by ±4
            if adx_4h >= 40:
                adx_adj = +4 if htf_raw > 0 else -4
            elif adx_4h >= 25:
                adx_adj = +2 if htf_raw > 0 else -2
            else:
                adx_adj = 0  # weak/no trend — neutral

            score = int(np.clip(htf_base + rsi_adj + adx_adj, 0, 100))

            # ── Label ─────────────────────────────────────────────────
            if   score >= 85: trend = "VERY STRONG BULL 🟢🟢🟢"
            elif score >= 70: trend = "STRONG BULL 🟢🟢"
            elif score >= 56: trend = "BULL 🟢"
            elif score >= 45: trend = "NEUTRAL ⚪"
            elif score >= 30: trend = "BEAR 🔴"
            else:             trend = "VERY WEAK BEAR 🔴🔴"

            result = BtcStrength(
                score=score, trend=trend, rsi=rsi_1h, price=price,
                htf_score=htf_raw, htf_labels=htf_str, adx_4h=round(adx_4h, 1),
                is_bull=(score >= 56), is_bear=(score <= 44),
            )

            log.info("BTC Strength: %d/100 — %s | HTF:%s | RSI=%.1f | ADX4H=%.0f",
                     score, trend, htf_str, rsi_1h, adx_4h)
            return result

        except Exception as e:
            log.warning("BTC strength error: %s", e)
            return BtcStrength(score=50, trend="NEUTRAL ⚪", rsi=50.0, price=0.0)

    # ─────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────

    def _get_df(self, interval: str, limit: int) -> pd.DataFrame | None:
        try:
            df = self._b.get_klines(self.SYMBOL, interval, limit)
            if df is None or df.empty:
                return None
            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception:
            return None

    def _tf_vote(self, df: pd.DataFrame | None, label: str, weight: int):
        """EMA21 vs EMA50 vote per timeframe."""
        try:
            if df is None or len(df) < 52:
                return 0, f"{label}~"
            c   = df["close"].astype(float)
            e21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
            e50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1])
            pr  = float(c.iloc[-1])
            if   pr > e50 and e21 > e50: return +weight, f"{label}↑"
            elif pr < e50 and e21 < e50: return -weight, f"{label}↓"
            else:                        return  0,       f"{label}~"
        except Exception:
            return 0, f"{label}~"

    def _rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        try:
            c    = df["close"].astype(float)
            d    = c.diff()
            gain = d.clip(lower=0).rolling(period).mean()
            loss = (-d.clip(upper=0)).rolling(period).mean()
            return float((100 - 100 / (1 + gain / loss.replace(0, 1e-10))).iloc[-1])
        except Exception:
            return 50.0

    def _adx(self, df: pd.DataFrame, period: int = 14) -> float:
        try:
            if len(df) < period * 2:
                return 20.0
            h   = df["high"].astype(float)
            l   = df["low"].astype(float)
            c   = df["close"].astype(float)
            tr  = ((h - l).combine((h - c.shift(1)).abs(), max)
                          .combine((l - c.shift(1)).abs(), max))
            atr = tr.rolling(period).mean()
            dm_up   = (h - h.shift(1)).clip(lower=0)
            dm_down = (l.shift(1) - l).clip(lower=0)
            di_up   = (dm_up.rolling(period).mean()   / atr.replace(0, 1e-10)) * 100
            di_down = (dm_down.rolling(period).mean() / atr.replace(0, 1e-10)) * 100
            dx      = (abs(di_up - di_down) / (di_up + di_down + 1e-10)) * 100
            return float(dx.rolling(period).mean().iloc[-1])
        except Exception:
            return 20.0