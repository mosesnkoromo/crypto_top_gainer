"""
src/analysis/spot_signal_engine.py — Weekly Oversold Bounce (Spot)
────────────────────────────────────────────────────────────────────
A completely separate spot trading engine. Nothing shared with futures.

Strategy: Weekly Oversold Bounce
  - Weekly RSI < 38 (coin is genuinely cheap vs history)
  - Daily RSI turning up from oversold (recovery starting)
  - Price at or near weekly support (structure holds)
  - BTC not in total collapse (score > 25 — very lenient for spot)
  - Volume dried up on the dip (sellers exhausted)
  - Enter on daily RSI cross above 35 or StochRSI turn
  - Hold 1–3 weeks through normal noise — no tight SL
  - SL: below weekly swing low (structural, not %)
  - TP1/TP2/TP3: previous weekly resistance levels (real targets)

Why this works for spot:
  - You OWN the coin — a 5% dip is noise, not a margin call
  - Weekly oversold means real value — not just a 4H blip
  - No hard % SL — structural SL below weekly low prevents wash-outs
  - 1-3 week hold lets the move develop naturally
"""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import pandas as pd

from src.data.binance_client import BinanceClient
from src.analysis.btc_strength import BtcStrength
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SpotSignal:
    symbol:         str
    grade:          str          # PRIME / GOOD / WATCH
    confidence:     int
    price:          float
    gain_24h:       float

    # Weekly context
    weekly_rsi:     float
    daily_rsi:      float
    rsi_4h:         float
    weekly_low:     float        # structural SL reference
    weekly_high:    float        # resistance reference

    # Targets — based on market structure, not %
    tp1:            float        # prev weekly resistance 1
    tp2:            float        # prev weekly resistance 2
    tp3:            float        # prev weekly resistance 3 / prev high
    sl:             float        # below weekly swing low
    sl_pct:         float        # SL% from entry (for display)
    tp1_pct:        float
    tp2_pct:        float
    tp3_pct:        float

    hold_weeks:     str          # "1–2 weeks" / "2–3 weeks"
    factors:        list[str]
    entry_type:     str          # "Immediate" / "Wait for daily RSI > 35"
    btc_score:      int
    score:          float        # raw confluence score
    volume_note:    str          # observation on volume


# ── BTC gate for spot (very lenient — you're buying the dip) ──
SPOT_BTC_MIN  = 25    # only skip if BTC is total collapse
SPOT_BTC_WARN = 45    # below this: caution, smaller size

# ── RSI thresholds ─────────────────────────────────────────────
WEEKLY_RSI_MAX  = 38   # must be genuinely oversold on weekly
WEEKLY_RSI_MIN  = 15   # below 15 = extreme fear, skip — could keep dropping
DAILY_RSI_MAX   = 52   # daily must not have already recovered too much
RSI_4H_MAX      = 60   # 4H should not be overbought at entry


class SpotSignalEngine:

    def __init__(self, binance: BinanceClient):
        self._b = binance

    def analyze(self, ticker: dict, btc: BtcStrength) -> SpotSignal | None:
        sym   = ticker["symbol"]
        price = float(ticker.get("lastPrice", 0) or ticker.get("current_price", 0))
        gain  = float(ticker.get("priceChangePercent", 0))
        if price <= 0:
            return None

        # ── Fetch weekly + daily + 4H ──────────────────────────
        try:
            df_1w = self._b.get_klines(sym, "1w", 52)   # 1 year weekly
            df_1d = self._b.get_klines(sym, "1d", 90)   # 3 months daily
            df_4h = self._b.get_klines(sym, "4h", 60)   # entry timing
        except Exception as e:
            log.debug("Spot: %s data error: %s", sym, e)
            return None

        if df_1w.empty or len(df_1w) < 20: return None
        if df_1d.empty or len(df_1d) < 30: return None

        # ── Weekly indicators ──────────────────────────────────
        w_close  = df_1w["close"]
        w_rsi    = self._rsi(w_close)
        w_low_20 = float(df_1w["low"].tail(20).min())   # 20-week low = structural support
        w_high_20= float(df_1w["high"].tail(20).max())
        w_vol_ma = df_1w["volume"].tail(10).mean()
        w_vol_now= float(df_1w["volume"].iloc[-1])

        # ── Daily indicators ───────────────────────────────────
        d_close  = df_1d["close"]
        d_rsi    = self._rsi(d_close)
        d_rsi_5d_ago = self._rsi(d_close.iloc[:-5]) if len(d_close) > 10 else d_rsi
        d_vol_ma = df_1d["volume"].tail(20).mean()
        d_vol_now= float(df_1d["volume"].iloc[-1])
        d_ema20  = float(d_close.ewm(span=20).mean().iloc[-1])
        d_ema50  = float(d_close.ewm(span=50).mean().iloc[-1])

        # ── 4H for entry timing ────────────────────────────────
        rsi_4h = self._rsi(df_4h["close"]) if not df_4h.empty and len(df_4h) >= 15 else 50.0
        srsi_4h= self._stoch_rsi(df_4h["close"]) if not df_4h.empty and len(df_4h) >= 20 else {"k": 50, "d": 50}

        # ── HARD GATES ─────────────────────────────────────────
        if btc.score < SPOT_BTC_MIN:    return None   # total market collapse
        if w_rsi > WEEKLY_RSI_MAX:      return None   # not oversold enough
        if w_rsi < WEEKLY_RSI_MIN:      return None   # extreme fear — skip
        if d_rsi > DAILY_RSI_MAX:       return None   # daily already recovered too much
        if rsi_4h > RSI_4H_MAX:         return None   # 4H overbought — don't chase

        # ── Score ──────────────────────────────────────────────
        score, factors = 0.0, []

        # 1. Weekly RSI depth (0–2.5) — deeper = more oversold = better bounce
        if w_rsi <= 25:
            score += 2.5; factors.append(f"✅ WEEKLY RSI {w_rsi:.0f} — extreme oversold, high bounce probability")
        elif w_rsi <= 30:
            score += 2.0; factors.append(f"✅ WEEKLY RSI {w_rsi:.0f} — deeply oversold")
        elif w_rsi <= 35:
            score += 1.4; factors.append(f"✅ WEEKLY RSI {w_rsi:.0f} — oversold zone")
        else:
            score += 0.8; factors.append(f"⚠️ WEEKLY RSI {w_rsi:.0f} — approaching oversold")

        # 2. Daily RSI recovering (0–2.0) — the turn is the signal
        d_rsi_rising = d_rsi > d_rsi_5d_ago + 2
        if 30 <= d_rsi <= 45 and d_rsi_rising:
            score += 2.0; factors.append(f"✅ DAILY RSI {d_rsi:.0f} — turning up from oversold (recovery started)")
        elif 25 <= d_rsi < 30 and d_rsi_rising:
            score += 1.5; factors.append(f"✅ DAILY RSI {d_rsi:.0f} — just turning from extreme low")
        elif d_rsi <= 35:
            score += 1.0; factors.append(f"⚠️ DAILY RSI {d_rsi:.0f} — oversold, not yet turning")
        elif d_rsi <= 45:
            score += 0.5; factors.append(f"⚠️ DAILY RSI {d_rsi:.0f} — mild recovery")

        # 3. Price near weekly support (0–2.0)
        dist_from_low = (price - w_low_20) / price
        if dist_from_low <= 0.03:
            score += 2.0; factors.append(f"✅ WEEKLY SUPPORT: price within 3% of 20-week low ({w_low_20:.5g})")
        elif dist_from_low <= 0.08:
            score += 1.3; factors.append(f"✅ WEEKLY SUPPORT: price near 20-week low ({w_low_20:.5g})")
        elif dist_from_low <= 0.15:
            score += 0.6; factors.append(f"⚠️ WEEKLY SUPPORT: moderately above 20-week low")

        # 4. Volume drying up = sellers exhausted (0–1.5)
        if w_vol_now < w_vol_ma * 0.6:
            score += 1.5; factors.append(f"✅ WEEKLY VOLUME: {w_vol_now/w_vol_ma:.1f}x avg — sellers exhausted (very dry)")
        elif w_vol_now < w_vol_ma * 0.8:
            score += 1.0; factors.append(f"✅ WEEKLY VOLUME: {w_vol_now/w_vol_ma:.1f}x avg — volume drying up")
        elif w_vol_now < w_vol_ma:
            score += 0.5; factors.append(f"⚠️ WEEKLY VOLUME: {w_vol_now/w_vol_ma:.1f}x avg — slightly below avg")

        # 5. Daily volume also declining (0–0.8)
        if d_vol_now < d_vol_ma * 0.7:
            score += 0.8; factors.append("✅ DAILY VOLUME: low on pullback — no panic selling")
        elif d_vol_now < d_vol_ma:
            score += 0.4; factors.append("⚠️ DAILY VOLUME: below average")

        # 6. Price below daily EMA50 (0–1.0) — oversold relative to trend
        if price < d_ema50:
            score += 1.0; factors.append(f"✅ DAILY: price below EMA50 ({d_ema50:.5g}) — discounted from trend")
        elif price < d_ema20:
            score += 0.5; factors.append(f"⚠️ DAILY: price below EMA20 — short-term pullback")

        # 7. 4H StochRSI turning up (0–1.0) — entry timing
        if srsi_4h["k"] < 25 and srsi_4h["k"] > srsi_4h["d"]:
            score += 1.0; factors.append(f"✅ 4H StochRSI turning up (K:{srsi_4h['k']:.0f}) — entry timing confirmed")
        elif srsi_4h["k"] < 40:
            score += 0.5; factors.append(f"⚠️ 4H StochRSI {srsi_4h['k']:.0f} — not yet turning")

        # 8. BTC support (0–1.0)
        if btc.score >= 55:
            score += 1.0; factors.append(f"✅ BTC {btc.score}/100 — market supports recovery")
        elif btc.score >= 40:
            score += 0.5; factors.append(f"⚠️ BTC {btc.score}/100 — neutral, trade carefully")
        else:
            factors.append(f"⚠️ BTC {btc.score}/100 — weak, use smaller position size")

        # ── Grade ──────────────────────────────────────────────
        if score >= 7.0:
            grade = "PRIME 💎"; confidence = min(85, 72 + int(score - 7))
        elif score >= 5.0:
            grade = "GOOD 🟢"; confidence = min(78, 62 + int(score - 5))
        elif score >= 3.5:
            grade = "WATCH 👀"; confidence = 55
        else:
            return None   # not convincing enough

        # ── Structural targets (key resistance levels) ─────────
        # TP levels = previous weekly highs (real resistance, not % targets)
        w_highs = df_1w["high"].values
        # Find the last 3 meaningful resistance levels above current price
        resistance_levels = sorted(
            [float(h) for h in w_highs[-20:] if h > price * 1.02],
            key=lambda x: abs(x - price)
        )

        if len(resistance_levels) >= 3:
            tp1 = round(resistance_levels[0], 8)
            tp2 = round(resistance_levels[1], 8)
            tp3 = round(resistance_levels[2], 8)
        elif len(resistance_levels) == 2:
            tp1 = round(resistance_levels[0], 8)
            tp2 = round(resistance_levels[1], 8)
            tp3 = round(price * 1.20, 8)   # fallback: 20% from entry
        elif len(resistance_levels) == 1:
            tp1 = round(resistance_levels[0], 8)
            tp2 = round(price * 1.12, 8)
            tp3 = round(price * 1.20, 8)
        else:
            # No resistance found above — use % targets as fallback
            tp1 = round(price * 1.08, 8)
            tp2 = round(price * 1.15, 8)
            tp3 = round(price * 1.25, 8)

        # SL = below 20-week low — structural, not tight %
        sl = round(w_low_20 * 0.97, 8)   # 3% below 20-week support
        sl_pct  = round((price - sl) / price * 100, 1)
        tp1_pct = round((tp1 - price) / price * 100, 1)
        tp2_pct = round((tp2 - price) / price * 100, 1)
        tp3_pct = round((tp3 - price) / price * 100, 1)

        # ── Hold and entry guidance ────────────────────────────
        if d_rsi_rising and srsi_4h["k"] > srsi_4h["d"]:
            entry_type = "Enter now — daily RSI turning up"
            hold_weeks = "1–2 weeks"
        elif d_rsi <= 32:
            entry_type = "Wait for 4H StochRSI to turn up, then enter"
            hold_weeks = "2–3 weeks"
        else:
            entry_type = "Enter in stages — DCA over 2-3 days"
            hold_weeks = "2–3 weeks"

        vol_note = (
            "Sellers exhausted — ideal entry"     if w_vol_now < w_vol_ma * 0.6 else
            "Volume drying — good sign"           if w_vol_now < w_vol_ma * 0.8 else
            "Watch for volume to drop further"
        )

        log.info("SPOT %s: %s (score %.1f, weekly RSI %.0f, daily RSI %.0f)",
                 grade.split()[0], sym, score, w_rsi, d_rsi)

        return SpotSignal(
            symbol=sym, grade=grade, confidence=confidence,
            price=price, gain_24h=gain,
            weekly_rsi=round(w_rsi, 1), daily_rsi=round(d_rsi, 1), rsi_4h=round(rsi_4h, 1),
            weekly_low=w_low_20, weekly_high=w_high_20,
            tp1=tp1, tp2=tp2, tp3=tp3, sl=sl,
            sl_pct=sl_pct, tp1_pct=tp1_pct, tp2_pct=tp2_pct, tp3_pct=tp3_pct,
            hold_weeks=hold_weeks, factors=factors,
            entry_type=entry_type, btc_score=btc.score,
            score=round(score, 2), volume_note=vol_note,
        )

    # ── Indicators ──────────────────────────────────────────────

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> float:
        if len(series) < period + 1: return 50.0
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))
        return round(float(rsi.iloc[-1]), 2)

    @staticmethod
    def _stoch_rsi(series: pd.Series, period: int = 14, smooth: int = 3) -> dict:
        if len(series) < period * 2: return {"k": 50.0, "d": 50.0}
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))
        min_r = rsi.rolling(period).min()
        max_r = rsi.rolling(period).max()
        rng   = (max_r - min_r).replace(0, np.nan)
        k_raw = (rsi - min_r) / rng * 100
        k     = k_raw.rolling(smooth).mean()
        d     = k.rolling(smooth).mean()
        return {"k": round(float(k.iloc[-1]), 1), "d": round(float(d.iloc[-1]), 1)}