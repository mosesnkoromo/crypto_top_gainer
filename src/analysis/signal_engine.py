"""
src/analysis/signal_engine.py — Signal Engine v4.2
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal, Optional

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
    gain_24h:       float = 0.0
    rsi_1h:         float = 0.0
    rsi_4h:         float = 0.0
    rsi_daily:      float = 0.0
    tp1:            float = 0.0
    tp2:            float = 0.0
    tp3:            float = 0.0
    sl:             float = 0.0
    atr:            float = 0.0
    factors:        list  = field(default_factory=list)
    strategies_hit: list  = field(default_factory=list)
    btc_score:      int   = 50
    btc_trend:      str   = ""
    confluence:     float = 0.0
    hold_time:      str   = ""
    news_sentiment: str   = "neutral"
    regime:         str   = "Trending"
    sniper_conf:    float = 1.0
    tqi:            float = 0.5
    displacement_ratio: float = 0.0
    position_size:  float = 1.0
    tag:            str   = "STANDARD"   # STANDARD | CONFIRMATION | CAPITULATION


class SignalEngine:
    """
    Production-grade Signal Engine v4.2
    Pipeline:
      1. Fetch candles (5m, 15m, 1h, 4h, 1d, 1w, 1m)
      2. Regime via TQI
      3. Top-down HTF cascade 1W→1D→4H→1H→15m (hard gate)
      4. BTC + RSI hard filters
      5. Confluence scoring (WaveTrend + MACD + RSI + BTC + HTF bonus)
      6. 1m Sniper gate
      7. Grade: ULTRA / STRONG / STANDARD (scanner-compatible)
    """

    def __init__(self, binance_client):
        self._b                 = binance_client
        self.signals_this_cycle = 0

    def analyze(self, ticker: dict, btc) -> Signal | None:
        sym = ticker.get("symbol")
        if not sym:
            return None
        try:
            df_5m  = self._get_df(sym, "5m",  150)
            df_15m = self._get_df(sym, "15m",  80)
            df_1h  = self._get_df(sym, "1h",   80)
            df_4h  = self._get_df(sym, "4h",   60)
            df_1d  = self._get_df(sym, "1d",   60)
            df_1w  = self._get_df(sym, "1w",   20)
            df_1m  = self._get_df(sym, "1m",   50)

            if df_5m is None or len(df_5m) < 100:
                log.debug("  ⏭️  %s: insufficient candle data — skip", sym)
                return None

            price = float(df_5m["close"].astype(float).iloc[-1])

            # Step 1: Regime
            tqi_val = self._tqi(df_5m)
            if   tqi_val > 0.75: regime = "Strong_Trend_Impulse"
            elif tqi_val > 0.50: regime = "Trending"
            else:                regime = "Choppy_Range"

            tqi_min = {"Strong_Trend_Impulse": 0.15, "Trending": 0.20, "Choppy_Range": 0.30}
            if tqi_val < tqi_min[regime]:
                log.info("  ⏭️  %s: TQI=%.2f too choppy for %s — skip", sym, tqi_val, regime)
                return None

            # Step 2: HTF cascade
            htf_score, htf_labels, is_major_bull, is_major_bear, adx_4h = \
                self._top_down(df_1w, df_1d, df_4h, df_1h, df_15m)
            htf_str = " ".join(htf_labels)
            log.info("  📊 %s HTF: %s | score=%+d | ADX4H=%.0f | major_bull=%s major_bear=%s",
                     sym, htf_str, htf_score, adx_4h, is_major_bull, is_major_bear)

            # Step 3: Indicators
            disp_ratio, strong_disp = self._displacement(df_5m)
            wt1, wt2, wt_up, wt_dn  = self._wavetrend(df_5m)
            rsi_5m   = self._rsi(df_5m, 14)
            rsi_1h   = self._rsi(df_1h, 14)
            macd_h   = self._macd_hist(df_5m)
            btc_score = int(getattr(btc, "score", 50))
            atr_val  = float((df_5m["high"].astype(float) -
                               df_5m["low"].astype(float)).rolling(14).mean().iloc[-1])
            cci_val  = self._cci(df_5m, 20)

            # ── Volume Analysis ──────────────────────────────────────
            vol_ratio, obv_rising, vol_spike = self._volume_analysis(df_5m, df_1h)

            # Step 4: Direction — HTF decides WHAT to trade, LTF decides WHEN
            # ─────────────────────────────────────────────────────────
            # Derive bias from HTF cascade (already computed above)
            if   htf_score >= 3:   htf_direction = "BUY"    # 1D+4H or more aligned up
            elif htf_score <= -3:  htf_direction = "SELL"   # 1D+4H or more aligned down
            else:                  htf_direction = "NEUTRAL" # mixed/sideways

            # LTF readiness: 5m MACD + WaveTrend confirm timing
            ltf_buy  = macd_h > 0 and (wt1 > wt2 or wt_up)
            ltf_sell = macd_h < 0 and (wt1 < wt2 or wt_dn)
            ltf_buy_weak  = macd_h > 0   # MACD only, no WT confirmation
            ltf_sell_weak = macd_h < 0

            if htf_direction == "BUY":
                # HTF says BUY — only look for BUY entries
                if ltf_buy:
                    direction = "BUY"     # strong: MACD + WT aligned
                elif ltf_buy_weak:
                    direction = "BUY"     # weak: MACD only (WT lagging)
                else:
                    # LTF is bearish in a bullish HTF — pullback in progress, wait
                    log.info("  ⏭️  %s: HTF bullish (score=%+d) but LTF bearish — pullback, skip",
                             sym, htf_score)
                    return None

            elif htf_direction == "SELL":
                # HTF says SELL — only look for SELL entries
                if ltf_sell:
                    direction = "SELL"
                elif ltf_sell_weak:
                    direction = "SELL"
                else:
                    log.info("  ⏭️  %s: HTF bearish (score=%+d) but LTF bullish — pullback, skip",
                             sym, htf_score)
                    return None

            else:
                # HTF NEUTRAL (range/mixed) — allow both directions based on LTF only
                # This handles Choppy_Range regime mean-reversion trades
                if   ltf_buy  or ltf_buy_weak:   direction = "BUY"
                elif ltf_sell or ltf_sell_weak:  direction = "SELL"
                else:
                    log.info("  ⏭️  %s: MACD flat (hist=%.4f) — no direction → skip", sym, macd_h)
                    return None

            # ── CAPITULATION OVERRIDE ──────────────────────────────
            # BTC RSI < 25 (extreme oversold) = market panic/capitulation
            # These are the HIGHEST edge mean-reversion BUY setups
            # Allow BUY signals on oversold pairs even during pullback
            btc_rsi = btc.rsi if hasattr(btc, "rsi") else 50
            if btc_rsi < 25 and direction == "SELL" and rsi_5m < 25:
                # Don't SELL into extreme oversold during capitulation
                log.info("  ⏭️  %s: Capitulation mode (BTC RSI=%.0f) — skip SELL at oversold RSI=%.0f",
                         sym, btc_rsi, rsi_5m)
                return None

            # Velocity check: only enter capitulation BUY when RSI is RISING (panic ending)
            # RSI=12 dropping = still dangerous; RSI=12→24 rising = momentum shift = buy edge
            btc_rsi_prev = getattr(btc, "rsi_prev", btc_rsi)  # fallback to current if unavailable
            btc_rsi_rising = btc_rsi > btc_rsi_prev or btc_rsi > 20   # rising or recovered above 20

            if btc_rsi < 25 and htf_direction == "BUY" and not (ltf_buy or ltf_buy_weak):
                # Allow BUY only when: pair oversold AND BTC RSI rising (velocity confirms)
                if rsi_5m < 30 and btc_rsi_rising:
                    log.info("  📍 %s: Capitulation BUY (BTC RSI=%.0f↑, pair RSI=%.0f) — override pullback",
                             sym, btc_rsi, rsi_5m)
                    direction = "BUY"
                elif rsi_5m < 30 and not btc_rsi_rising:
                    log.info("  ⏭️  %s: Capitulation — BTC RSI still falling (%.0f), wait",
                             sym, btc_rsi)
                    return None

            is_buy = direction == "BUY"

            # Step 4b: INDICATOR CORRELATION GATE
            # Count how many indicators conflict with the trade direction.
            # If ≥2 indicators disagree, the setup is contradictory — skip.
            #
            # Conflict rules:
            #   BUY:  WT overbought (>50)            → conflict
            #         TQI choppy (<0.30) + no WT gold → conflict
            #         CCI overbought (>85)            → conflict
            #         RSI overbought (>75)            → conflict
            #   SELL: WT oversold (<-50)              → conflict
            #         TQI choppy (<0.30) + no WT gold → conflict
            #         CCI oversold (<-85)             → conflict
            #         RSI oversold (<25)              → conflict
            conflicts = 0
            conflict_detail = []
            if is_buy:
                if wt1 > 50:
                    conflicts += 1; conflict_detail.append(f"WT={wt1:.0f}↑overbought")
                if tqi_val < 0.30 and not wt_up:
                    conflicts += 1; conflict_detail.append(f"TQI={tqi_val:.2f}↓choppy+noGold")
                if cci_val > 85:
                    conflicts += 1; conflict_detail.append(f"CCI={cci_val:.0f}↑overbought")
                if rsi_5m > 75:
                    conflicts += 1; conflict_detail.append(f"RSI={rsi_5m:.0f}↑overbought")
            else:
                if wt1 < -50:
                    conflicts += 1; conflict_detail.append(f"WT={wt1:.0f}↓oversold")
                if tqi_val < 0.30 and not wt_dn:
                    conflicts += 1; conflict_detail.append(f"TQI={tqi_val:.2f}↓choppy+noGold")
                if cci_val < -85:
                    conflicts += 1; conflict_detail.append(f"CCI={cci_val:.0f}↓oversold")
                if rsi_5m < 25:
                    conflicts += 1; conflict_detail.append(f"RSI={rsi_5m:.0f}↓oversold")

            # Conflicts now penalise score instead of blocking
            # Hard block only on 4+ conflicts (truly contradictory signal)
            conflict_penalty = conflicts * 4   # -4 pts per conflict
            if conflicts >= 4:
                log.info("  ❌ %s %s | %d indicator conflicts (%s) — too contradictory → skip",
                         sym, direction, conflicts, " ".join(conflict_detail))
                return None
            elif conflicts > 0:
                log.info("  ⚠️  %s %s | %d indicator conflict(s) (%s) → score penalty -%d",
                         sym, direction, conflicts, " ".join(conflict_detail), conflict_penalty)

            # Step 5: HTF gate — hard block only when ALL major TFs strongly against
            # (1W+1D+4H all bearish = true bear market, no BUY; vice versa for SELL)
            # Softer misalignment converts to score penalty instead of hard block
            htf_penalty = 0
            if is_buy:
                # Only hard block if 1W+1D+4H all bearish (total ≤ -9)
                if htf_score <= -9:
                    log.info("  ❌ %s BUY | ALL major TFs bearish (score=%+d) — %s → blocked",
                             sym, htf_score, htf_str)
                    return None
                # Soft: penalise for each bearish major TF
                if is_major_bear:  htf_penalty += 10  # 1D+4H both bearish
                elif htf_score < 0: htf_penalty += abs(htf_score) * 1.5
            else:
                # Only hard block if 1W+1D+4H all bullish (total ≥ +9)
                if htf_score >= 9:
                    log.info("  ❌ %s SELL | ALL major TFs bullish (score=%+d) — %s → blocked",
                             sym, htf_score, htf_str)
                    return None
                if is_major_bull:  htf_penalty += 10
                elif htf_score > 0: htf_penalty += htf_score * 1.5

            # Step 6: BTC filter
            if is_buy  and btc_score < 30:
                log.info("  ❌ %s BUY blocked | BTC=%d (bear market)", sym, btc_score)
                return None
            if not is_buy and btc_score > 70:
                log.info("  ❌ %s SELL blocked | BTC=%d (bull market)", sym, btc_score)
                return None

            # Step 7: RSI extreme blocks — hard block only at truly extreme levels
            # Strong_Trend_Impulse: allow overbought continuation (BTC/SOL breakouts)
            # Other regimes: penalise but don't fully block until extreme
            rsi_penalty = 0
            if is_buy:
                if rsi_5m > 90 or rsi_1h > 88:   # truly extreme — block
                    log.info("  ❌ %s BUY | RSI extreme (5m=%.0f 1h=%.0f) → skip", sym, rsi_5m, rsi_1h)
                    return None
                elif rsi_5m > 78 and regime != "Strong_Trend_Impulse":
                    rsi_penalty = (rsi_5m - 78) * 1.5   # e.g. RSI=82 → -6 pts
                elif rsi_5m > 82:   # strong trend but still very high
                    rsi_penalty = (rsi_5m - 82) * 1.0
            else:
                if rsi_5m < 10 or rsi_1h < 12:   # truly extreme — block
                    log.info("  ❌ %s SELL | RSI extreme (5m=%.0f 1h=%.0f) → skip", sym, rsi_5m, rsi_1h)
                    return None
                elif rsi_5m < 22 and regime != "Strong_Trend_Impulse":
                    rsi_penalty = (22 - rsi_5m) * 1.5
                elif rsi_5m < 18:
                    rsi_penalty = (18 - rsi_5m) * 1.0

            # Step 8: Confluence score
            score = 40.0
            if   regime == "Strong_Trend_Impulse": score += 15
            elif regime == "Trending":             score += 10
            else:                                  score +=  5

            if strong_disp: score += 12

            if is_buy  and wt_up:    score += 15
            if not is_buy and wt_dn: score += 15
            if is_buy  and wt1 > wt2: score += 5
            if not is_buy and wt1 < wt2: score += 5
            if is_buy  and wt1 > 50:  score -= 8
            if not is_buy and wt1 < -50: score -= 8

            if abs(macd_h) > 0.05: score += 8

            # Volume confirmation scoring
            if vol_spike:
                # Volume spike (3×+ average) — confirms breakout move
                score += 8 if is_buy else 8
                log.debug("  %s: volume spike %.1f× avg → +8", sym, vol_ratio)
            elif vol_ratio >= 1.5:
                # Above-average volume confirms direction
                score += 5
            elif vol_ratio < 0.7:
                # Very low volume = weak move, likely to reverse
                score -= 5
                log.debug("  %s: thin volume %.1f× avg → -5", sym, vol_ratio)

            # OBV trend — money flow direction
            if is_buy and obv_rising:
                score += 6   # buyers accumulating = confirms BUY
            elif not is_buy and not obv_rising:
                score += 6   # sellers distributing = confirms SELL
            elif is_buy and not obv_rising:
                score -= 4   # selling pressure on BUY signal = caution
            elif not is_buy and obv_rising:
                score -= 4   # buying pressure on SELL signal = caution

            if is_buy  and 35 < rsi_5m < 68: score += 8
            if not is_buy and 32 < rsi_5m < 65: score += 8

            if is_buy  and btc_score >= 60: score += 6
            if not is_buy and btc_score <= 40: score += 6

            if is_buy:
                htf_bonus = max(0, min(20, int(htf_score * 1.5)))
            else:
                htf_bonus = max(0, min(20, int(-htf_score * 1.5)))
            score += htf_bonus
            score  = min(100.0, score)

            # Apply accumulated gate penalties before threshold check
            score -= htf_penalty      # HTF misalignment penalty (0–15)
            score -= rsi_penalty      # RSI extreme penalty (0–12)
            score -= conflict_penalty # Indicator conflict penalty (0–12)
            score = max(0.0, min(100.0, score))

            # Relaxed thresholds — 60/65/72 (was: 62/68/80)
            threshold = {"Strong_Trend_Impulse": 60, "Trending": 65, "Choppy_Range": 72}[regime]
            if score < threshold:
                log.info("  ❌ %s %s | score=%.0f/%.0f | regime=%s | rsi=%.0f | 🚫 weak setup",
                         sym, direction, score, threshold, regime, rsi_5m)
                return None

            # Step 9: Sniper
            sniper = self._sniper_1m(df_1m, is_buy)
            # Relax sniper during capitulation — recovery moves start weak then gain momentum
            _btc_rsi_snap = getattr(btc, "rsi", 50)
            sniper_min = 0.20 if _btc_rsi_snap < 32 else 0.35
            if sniper < sniper_min:
                log.info("  ❌ %s %s | sniper=%.0f%% too weak (min=%.0f%%) → skip",
                         sym, direction, sniper*100, sniper_min*100)
                return None

            # Step 10: Grade (MUST match scanner's ULTRA/STRONG/STANDARD)
            if   score >= 88: grade = "ULTRA";    position_size = 1.0
            elif score >= 75: grade = "STRONG";   position_size = 0.75
            else:             grade = "STANDARD"; position_size = 0.50

            # Step 11: TP/SL
            sl_mult = {"Strong_Trend_Impulse": 1.2, "Trending": 1.5, "Choppy_Range": 1.8}[regime]
            sl_dist = atr_val * sl_mult
            # Minimum SL distance: 0.3% of price (prevents "SL too tight" on low-vol pairs)
            min_sl_dist = price * 0.003
            sl_dist = max(sl_dist, min_sl_dist)
            # R:R enforcement: TP1 must be ≥ 1.5× SL to ensure positive expectancy
            # Old: TP1=0.8×SL (R:R<1 → negative E even at 70% WR)
            # New: TP1=1.5×SL (R:R=1.5 → positive E at just 40% WR)
            if is_buy:
                sl  = price - sl_dist
                tp1 = price + sl_dist * 1.5   # was 0.8 — enforces min R:R 1.5
                tp2 = price + sl_dist * 2.5   # was 1.8
                tp3 = price + sl_dist * 4.0   # was 3.0 — runner
            else:
                sl  = price + sl_dist
                tp1 = price - sl_dist * 1.5
                tp2 = price - sl_dist * 2.5
                tp3 = price - sl_dist * 4.0

            if   score >= 88: _rating = "🟢🟢🟢 EXCELLENT"
            elif score >= 75: _rating = "🟢🟢 STRONG"
            elif score >= 62: _rating = "🟢 GOOD"
            else:             _rating = "🟡 MARGINAL"

            # Dynamic risk tag
            btc_rsi_chk = getattr(btc, "rsi", 50)
            if btc_rsi_chk < 25:
                signal_tag = "CAPITULATION"
            elif grade in ("ULTRA", "STRONG"):
                signal_tag = "CONFIRMATION"
            else:
                signal_tag = "STANDARD"

            # Strategy type label for log clarity
            _strategy = ("TREND CONTINUATION" if regime == "Strong_Trend_Impulse" else
                         "SWING"              if regime == "Trending" else
                         "MEAN REVERSION")

            log.info(
                "  ✅ %s %s | score=%.0f/100 %s | %s | regime=%s | rsi=%.0f | atr=%.2f%% | "
                "vol=%.1f× %s | HTF:%s(%+d) ADX4H=%.0f | penalties: htf=%.0f rsi=%.0f conf=%.0f",
                sym, direction, score, _rating, _strategy, regime, rsi_5m,
                atr_val / price * 100,
                vol_ratio, "📈OBV↑" if obv_rising else "📉OBV↓",
                htf_str, htf_score, adx_4h,
                htf_penalty, rsi_penalty, conflict_penalty)

            self.signals_this_cycle += 1
            return Signal(
                symbol=sym, signal=direction, grade=grade, tag=signal_tag,
                confidence=int(score),
                action="ENTER_LONG" if is_buy else "ENTER_SHORT",
                price=price, rsi_1h=rsi_1h, rsi_4h=rsi_1h, rsi_daily=rsi_1h,
                tp1=tp1, tp2=tp2, tp3=tp3, sl=sl, atr=atr_val,
                factors=(["HTF_CASCADE", "REGIME"] +
                         (["DISPLACEMENT"] if strong_disp else [])),
                strategies_hit=["TQI_REGIME", "WAVETREND"],
                btc_score=btc_score, btc_trend=getattr(btc, "trend", ""),
                confluence=score, hold_time="5-20min",
                regime=regime, sniper_conf=sniper, tqi=tqi_val,
                displacement_ratio=disp_ratio, position_size=position_size,
            )
        except Exception as e:
            log.warning("SignalEngine error %s: %s", sym, e)
            return None

    # ── TOP-DOWN HTF CASCADE ────────────────────────────────────────

    def _top_down(self, df_1w, df_1d, df_4h, df_1h, df_15m):
        def vote(df, label, w):
            try:
                if df is None or len(df) < 52: return 0, f"{label}~"
                c   = df["close"].astype(float)
                e21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
                e50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1])
                pr  = float(c.iloc[-1])
                if   pr > e50 and e21 > e50: return +w, f"{label}↑"
                elif pr < e50 and e21 < e50: return -w, f"{label}↓"
                else:                        return  0,  f"{label}~"
            except Exception:
                return 0, f"{label}~"

        s1w,  l1w  = vote(df_1w,  "1W",  3)
        s1d,  l1d  = vote(df_1d,  "1D",  3)
        s4h,  l4h  = vote(df_4h,  "4H",  3)
        s1h,  l1h  = vote(df_1h,  "1H",  2)
        s15m, l15m = vote(df_15m, "15m", 1)

        htf_score     = s1w + s1d + s4h + s1h + s15m
        is_major_bull = (s1d > 0 and s4h > 0)
        is_major_bear = (s1d < 0 and s4h < 0)

        adx_4h = 20.0
        try:
            if df_4h is not None and len(df_4h) >= 28:
                h  = df_4h["high"].astype(float)
                l  = df_4h["low"].astype(float)
                c4 = df_4h["close"].astype(float)
                tr = ((h - l).combine((h - c4.shift(1)).abs(), max)
                             .combine((l - c4.shift(1)).abs(), max))
                atr14   = tr.rolling(14).mean()
                dm_up   = (h - h.shift(1)).clip(lower=0)
                dm_dn   = (l.shift(1) - l).clip(lower=0)
                di_up   = (dm_up.rolling(14).mean() / atr14.replace(0, 1e-10)) * 100
                di_dn   = (dm_dn.rolling(14).mean() / atr14.replace(0, 1e-10)) * 100
                dx      = (abs(di_up - di_dn) / (di_up + di_dn + 1e-10)) * 100
                adx_4h  = float(dx.rolling(14).mean().iloc[-1])
        except Exception:
            pass

        return htf_score, [l1w, l1d, l4h, l1h, l15m], is_major_bull, is_major_bear, round(adx_4h, 1)

    # ── INDICATORS ─────────────────────────────────────────────────

    def _get_df(self, sym, interval, limit):
        try:
            df = self._b.get_klines(sym, interval, limit)
            if df is None or df.empty: return None
            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception:
            return None

    def _tqi(self, df, er_len=20, struct_len=20, mom_len=10) -> float:
        try:
            cl  = df["close"].astype(float)
            hi  = df["high"].astype(float)
            lo  = df["low"].astype(float)
            er  = (cl.diff(er_len).abs() /
                   cl.diff().abs().rolling(er_len).sum().replace(0, 1e-10)).clip(0, 1)
            hi_n = hi.rolling(struct_len).max()
            lo_n = lo.rolling(struct_len).min()
            struct = (((cl - lo_n) / (hi_n - lo_n + 1e-10) - 0.5).abs() * 2).clip(0, 1)
            win_dir = cl.diff(mom_len); bar_dir = cl.diff()
            aligned = sum(
                (((win_dir > 0) & (bar_dir.shift(i) > 0)) |
                 ((win_dir < 0) & (bar_dir.shift(i) < 0))).astype(float)
                for i in range(mom_len)
            )
            mom   = (aligned / mom_len).clip(0, 1)
            atr_v = (hi - lo).rolling(14).mean()
            vol_f = ((atr_v / atr_v.rolling(100).mean().replace(0, 1e-10) - 0.6) / 1.2).clip(0, 1)
            return float((er * 0.35 + struct * 0.25 + mom * 0.20 + vol_f * 0.20).clip(0, 1).iloc[-1])
        except Exception:
            return 0.5

    def _wavetrend(self, df, n1=10, n2=21):
        try:
            if len(df) < n2 + 5: return 0.0, 0.0, False, False
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

    def _rsi(self, df, period=14) -> float:
        try:
            c    = df["close"].astype(float)
            d    = c.diff()
            gain = d.clip(lower=0).rolling(period).mean()
            loss = (-d.clip(upper=0)).rolling(period).mean()
            return float((100 - 100 / (1 + gain / loss.replace(0, 1e-10))).iloc[-1])
        except Exception:
            return 50.0

    def _macd_hist(self, df) -> float:
        try:
            c    = df["close"].astype(float)
            fast = c.ewm(span=8,  adjust=False).mean()
            slow = c.ewm(span=17, adjust=False).mean()
            line = fast - slow
            return float((line - line.ewm(span=9, adjust=False).mean()).iloc[-1])
        except Exception:
            return 0.0

    def _volume_analysis(self, df_5m: pd.DataFrame,
                          df_1h: pd.DataFrame | None = None):
        """
        Returns (vol_ratio, obv_rising, vol_spike)
        vol_ratio:   current 5m volume / 20-bar moving average (e.g. 1.8 = 80% above avg)
        obv_rising:  On-Balance-Volume trend is up (money flowing in)
        vol_spike:   volume is ≥ 3× average (major institutional move or news)
        """
        try:
            vols  = df_5m["volume"].astype(float)
            close = df_5m["close"].astype(float)
            vol_ma = float(vols.rolling(20).mean().iloc[-1])
            vol_now = float(vols.iloc[-1])
            vol_ratio = vol_now / max(vol_ma, 1e-10)
            vol_spike = vol_ratio >= 3.0   # 3× avg = significant

            # OBV: +volume when close > prev_close, -volume when down
            direction_5m = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            obv = (vols * direction_5m).cumsum()
            obv_ema_fast = float(obv.ewm(span=5,  adjust=False).mean().iloc[-1])
            obv_ema_slow = float(obv.ewm(span=20, adjust=False).mean().iloc[-1])
            obv_rising = obv_ema_fast > obv_ema_slow

            # 1H volume confirmation (is higher TF participating?)
            if df_1h is not None and len(df_1h) >= 20:
                vols_1h  = df_1h["volume"].astype(float)
                vol_1h_ratio = float(vols_1h.iloc[-1]) / max(
                    float(vols_1h.rolling(20).mean().iloc[-1]), 1e-10)
                # If 1H volume is below average but 5m spike = fake breakout risk
                if vol_1h_ratio < 0.6 and vol_ratio > 2.0:
                    vol_spike = False   # 1H not confirming 5m spike

            return round(vol_ratio, 2), obv_rising, vol_spike
        except Exception:
            return 1.0, True, False   # neutral defaults on error

    def _cci(self, df, period=20) -> float:
        """Commodity Channel Index — measures deviation from typical price mean."""
        try:
            tp  = (df["high"].astype(float) + df["low"].astype(float) +
                   df["close"].astype(float)) / 3.0
            sma = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            return float(((tp - sma) / (0.015 * mad.replace(0, 1e-10))).iloc[-1])
        except Exception:
            return 0.0

    def _displacement(self, df):
        try:
            row  = df.iloc[-1]
            body = abs(float(row["close"]) - float(row["open"]))
            rng  = float(row["high"]) - float(row["low"])
            disp = body / rng if rng > 0 else 0.0
            vols   = df["volume"].astype(float)
            vol_ma = float(vols.rolling(20).mean().iloc[-1])
            vol_ok = vol_ma > 0 and float(vols.iloc[-1]) / vol_ma > 1.5
            return disp, (disp > 0.65 and vol_ok)
        except Exception:
            return 0.0, False

    def _sniper_1m(self, df_1m, is_buy: bool) -> float:
        if df_1m is None or len(df_1m) < 20:
            return 0.6
        try:
            c    = df_1m["close"].astype(float)
            e9   = float(c.ewm(span=9,  adjust=False).mean().iloc[-1])
            e21  = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
            sc   = 0.3 if ((is_buy and e9 > e21) or (not is_buy and e9 < e21)) else 0.0
            fast = c.ewm(span=8,  adjust=False).mean()
            slow = c.ewm(span=17, adjust=False).mean()
            hist = float((fast - slow - (fast - slow).ewm(span=9, adjust=False).mean()).iloc[-1])
            if (is_buy and hist > 0) or (not is_buy and hist < 0):
                sc += 0.4
            return min(1.0, sc + 0.3)
        except Exception:
            return 0.6


_signal_engine: Optional[SignalEngine] = None

def get_signal_engine(binance_client=None) -> SignalEngine:
    global _signal_engine
    if _signal_engine is None:
        if binance_client is None:
            from src.data.binance_client import BinanceClient
            from config import AppConfig
            cfg = AppConfig()
            binance_client = BinanceClient(cfg.binance, cfg.scan)
        _signal_engine = SignalEngine(binance_client)
    return _signal_engine