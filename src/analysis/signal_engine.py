"""
src/analysis/signal_engine.py — Signal Engine v5.1 (Production)
Structure‑First + Sniper Pullback + Adaptive Scoring
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any

from src.utils.logger import get_logger

log = get_logger(__name__)

SignalType = Literal["BUY", "SELL"]


@dataclass
class Signal:
    symbol: str
    signal: SignalType
    grade: str
    confidence: int
    action: str
    price: float
    gain_24h: float = 0.0
    rsi_1h: float = 0.0
    rsi_4h: float = 0.0
    rsi_daily: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    sl: float = 0.0
    atr: float = 0.0
    factors: list = field(default_factory=list)
    strategies_hit: list = field(default_factory=list)
    btc_score: int = 50
    btc_trend: str = ""
    confluence: float = 0.0
    hold_time: str = ""
    news_sentiment: str = "neutral"
    regime: str = "Trending"
    sniper_conf: float = 1.0
    tqi: float = 0.5
    displacement_ratio: float = 0.0
    position_size: float = 1.0
    tag: str = "STANDARD"
    asset_category: str = "NEUTRAL"
    trigger_type: str = ""
    score_breakdown: Dict[str, float] = field(default_factory=dict)


class SignalEngine:
    """
    Signal Engine v5.1 – Structure‑First + Sniper Pullback

    Pipeline:
      1. Fetch candles (5m, 15m, 1h, 4h, 1d)
      2. Detect structure: Sweep + Reclaim, or Breakout
      3. Wait for pullback to 20 EMA / swing level (sniper entry)
      4. Confirm with engulfing/pinbar candle
      5. Base score + confirmations (HTF, Volume, RSI, BTC, Gainer)
      6. Apply minimal penalties
      7. Final score must exceed adaptive threshold
    """

    def __init__(self, binance_client, initial_threshold: int = 52):
        self._b = binance_client
        self.threshold = initial_threshold  # will be updated by scanner
        self.signals_this_cycle = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, ticker: dict, btc) -> Signal | None:
        sym = ticker.get("symbol")
        if not sym:
            return None

        try:
            # Fetch data
            df_5m = self._get_df(sym, "5m", 150)
            if df_5m is None or len(df_5m) < 100:
                return None

            df_15m = self._get_df(sym, "15m", 80)
            df_1h  = self._get_df(sym, "1h",  80)
            df_4h  = self._get_df(sym, "4h",  60)
            df_1d  = self._get_df(sym, "1d",  60)

            price = float(df_5m["close"].astype(float).iloc[-1])
            gain_24h = float(ticker.get("priceChangePercent", 0) or 0)

            # Asset category
            if gain_24h >= 5.0:
                asset_category = "GAINER"
            elif gain_24h <= -5.0:
                asset_category = "LOSER"
            else:
                asset_category = "NEUTRAL"

            # ------------------------------------------------------------------
            # STEP 1: STRUCTURE TRIGGER
            # ------------------------------------------------------------------
            sweep_low, sweep_high = self._detect_sweep(df_5m)
            reclaim_buy, reclaim_sell = self._reclaim_confirm(df_5m, sweep_low, sweep_high)
            bos_up, bos_down = self._break_of_structure(df_5m)

            structure_event = None
            trigger_type = ""
            base_score = 0.0

            if sweep_low and reclaim_buy:
                structure_event = "BUY"
                trigger_type = "Sweep"
                base_score = 55.0
            elif sweep_high and reclaim_sell:
                structure_event = "SELL"
                trigger_type = "Sweep"
                base_score = 55.0
            elif bos_up:
                structure_event = "BUY"
                trigger_type = "Breakout"
                base_score = 48.0
            elif bos_down:
                structure_event = "SELL"
                trigger_type = "Breakout"
                base_score = 48.0
            else:
                return None

            # ------------------------------------------------------------------
            # STEP 2: SNIPER PULLBACK (refine entry)
            # ------------------------------------------------------------------
            direction = None
            is_pullback = False
            retrace_pct = 0.0

            if structure_event == "BUY":
                is_pullback, retrace_pct = self._detect_pullback(df_5m, "BUY")
                if is_pullback and self._confirm_pullback_entry(df_5m, "BUY"):
                    direction = "BUY"
                    base_score += 10   # sniper precision bonus
                    trigger_type = f"Sniper Pullback ({retrace_pct:.1f}%)"
                    log.debug("  🎯 %s pullback entry at %.1f%% retrace", sym, retrace_pct)
                else:
                    # Still allow entry but with lower base score (no pullback penalty)
                    direction = "BUY"
                    base_score -= 5
                    trigger_type = f"{trigger_type} (no pullback)"
            else:  # SELL
                is_pullback, retrace_pct = self._detect_pullback(df_5m, "SELL")
                if is_pullback and self._confirm_pullback_entry(df_5m, "SELL"):
                    direction = "SELL"
                    base_score += 10
                    trigger_type = f"Sniper Pullback ({retrace_pct:.1f}%)"
                    log.debug("  🎯 %s pullback entry at %.1f%% retrace", sym, retrace_pct)
                else:
                    direction = "SELL"
                    base_score -= 5
                    trigger_type = f"{trigger_type} (no pullback)"

            is_buy = direction == "BUY"
            score = base_score
            breakdown = {"base": base_score}

            # ------------------------------------------------------------------
            # STEP 3: CONFIRMATIONS (BOOSTS)
            # ------------------------------------------------------------------
            # HTF Cascade
            htf_score, htf_labels, is_major_bull, is_major_bear, adx_4h = self._top_down(
                df_1d, df_4h, df_1h, df_15m
            )
            if is_buy:
                htf_bonus = min(25, htf_score * 3)
            else:
                htf_bonus = min(25, -htf_score * 3)
            score += htf_bonus
            breakdown["htf"] = htf_bonus

            # Volume
            vol_ratio, obv_rising, vol_spike = self._volume_analysis(df_5m, df_1h)
            if vol_spike:
                score += 15
                breakdown["vol_spike"] = 15
            elif vol_ratio >= 1.5:
                score += 12
                breakdown["vol_strong"] = 12
            elif vol_ratio < 0.7:
                score -= 2
                breakdown["vol_weak"] = -2

            # OBV alignment
            if is_buy and obv_rising:
                score += 10
                breakdown["obv"] = 10
            elif not is_buy and not obv_rising:
                score += 10
                breakdown["obv"] = 10
            elif is_buy and not obv_rising:
                score -= 1
                breakdown["obv_against"] = -1
            elif not is_buy and obv_rising:
                score -= 1
                breakdown["obv_against"] = -1

            # RSI (5m)
            rsi_5m = self._rsi(df_5m, 14)
            if is_buy and 30 < rsi_5m < 65:
                score += 12
                breakdown["rsi"] = 12
            elif not is_buy and 35 < rsi_5m < 70:
                score += 12
                breakdown["rsi"] = 12

            # BTC Bias
            btc_score_val = int(getattr(btc, "score", 50))
            if is_buy and btc_score_val >= 60:
                score += 15
                breakdown["btc"] = 15
            elif not is_buy and btc_score_val <= 40:
                score += 15
                breakdown["btc"] = 15

            # Gainer/Loser Boost
            if asset_category == "GAINER" and is_buy:
                if gain_24h < 15.0:
                    score += 10
                    breakdown["gainer"] = 10
                else:
                    score -= 3
                    breakdown["gainer_overextended"] = -3
            elif asset_category == "LOSER" and not is_buy:
                if gain_24h > -15.0:
                    score += 10
                    breakdown["loser"] = 10
                else:
                    score -= 3
                    breakdown["loser_overextended"] = -3
            elif asset_category == "LOSER" and is_buy:
                score += 6
                breakdown["capitulation"] = 6

            # ------------------------------------------------------------------
            # STEP 4: PENALTIES (minimal)
            # ------------------------------------------------------------------
            # Reversal detection for gainers
            is_reversing, _ = self._detect_gainer_reversal(df_5m, df_1h)
            if asset_category == "GAINER" and is_buy and is_reversing:
                score -= 2
                breakdown["gainer_reversing"] = -2

            # RSI extreme
            if is_buy and rsi_5m > 80:
                score -= 3
                breakdown["rsi_overbought"] = -3
            elif not is_buy and rsi_5m < 20:
                score -= 3
                breakdown["rsi_oversold"] = -3

            cci_val = self._cci(df_5m, 20)
            if (is_buy and cci_val > 200) or (not is_buy and cci_val < -200):
                score -= 2
                breakdown["cci_extreme"] = -2

            tqi_val = self._tqi(df_5m)
            if tqi_val < 0.25:
                score -= 2
                breakdown["tqi_low"] = -2

            # Hard block ONLY for extreme RSI
            if (is_buy and rsi_5m > 92) or (not is_buy and rsi_5m < 8):
                return None

            # ------------------------------------------------------------------
            # STEP 5: THRESHOLD CHECK
            # ------------------------------------------------------------------
            if score < self.threshold:
                log.info("  ❌ %s %s | score %.0f < threshold %d", sym, direction, score, self.threshold)
                return None

            # ------------------------------------------------------------------
            # STEP 6: TP/SL & Signal Creation
            # ------------------------------------------------------------------

            atr_val = float((df_5m["high"].astype(float) - df_5m["low"].astype(float)).rolling(14).mean().iloc[-1])

            sl_mult = 0.05
            sl_dist = atr_val * sl_mult
            min_sl_dist = price * 0.008
            sl_dist = max(sl_dist, min_sl_dist)

            if is_buy:
                sl  = price - sl_dist
                tp1 = price + sl_dist * 1.5
                tp2 = price + sl_dist * 2.5
                tp3 = price + sl_dist * 4.0
            else:
                sl  = price + sl_dist
                tp1 = price - sl_dist * 1.5
                tp2 = price - sl_dist * 2.5
                tp3 = price - sl_dist * 4.0

            # Grade
            if score >= 90:
                grade = "ULTRA"
                position_size = 1.0
            elif score >= 75:
                grade = "STRONG"
                position_size = 0.75
            else:
                grade = "STANDARD"
                position_size = 0.50

            confidence = int(min(100, score))

            self.signals_this_cycle += 1
            return Signal(
                symbol=sym,
                signal=direction,
                grade=grade,
                confidence=confidence,
                action="ENTER_LONG" if is_buy else "ENTER_SHORT",
                price=price,
                gain_24h=gain_24h,
                rsi_1h=self._rsi(df_1h, 14) if df_1h is not None else 50,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                sl=sl,
                atr=atr_val,
                btc_score=btc_score_val,
                btc_trend=getattr(btc, "trend", ""),
                confluence=score,
                regime="Trending",
                tqi=tqi_val,
                position_size=position_size,
                asset_category=asset_category,
                trigger_type=trigger_type,
                score_breakdown=breakdown,
                factors=[f"v5.1 trigger: {trigger_type}"],
            )

        except Exception as e:
            log.warning("SignalEngine error %s: %s", sym, e)
            return None

    # ------------------------------------------------------------------
    # Structure Detection
    # ------------------------------------------------------------------
    def _detect_sweep(self, df: pd.DataFrame) -> tuple[bool, bool]:
        high = df["high"].astype(float).values
        low  = df["low"].astype(float).values
        close = df["close"].astype(float).values
        if len(high) < 21:
            return False, False
        prev_high_20 = np.max(high[-21:-1])
        prev_low_20  = np.min(low[-21:-1])
        last_high = high[-1]
        last_low  = low[-1]
        last_close = close[-1]
        sweep_high = last_high > prev_high_20 and last_close < prev_high_20
        sweep_low  = last_low  < prev_low_20  and last_close > prev_low_20
        return sweep_low, sweep_high

    def _reclaim_confirm(self, df: pd.DataFrame, sweep_low: bool, sweep_high: bool) -> tuple[bool, bool]:
        close = df["close"].astype(float).values
        if len(close) < 3:
            return False, False
        reclaim_buy = False
        reclaim_sell = False
        if sweep_low:
            support = np.min(df["low"].astype(float).values[-21:-1])
            reclaim_buy = close[-1] > support and close[-2] <= support
        if sweep_high:
            resistance = np.max(df["high"].astype(float).values[-21:-1])
            reclaim_sell = close[-1] < resistance and close[-2] >= resistance
        return reclaim_buy, reclaim_sell

    def _break_of_structure(self, df: pd.DataFrame) -> tuple[bool, bool]:
        high = df["high"].astype(float).values
        low  = df["low"].astype(float).values
        if len(high) < 3:
            return False, False
        bos_up = high[-1] > high[-3]
        bos_down = low[-1] < low[-3]
        return bos_up, bos_down

    def _detect_pullback(self, df: pd.DataFrame, direction: str) -> tuple[bool, float]:
        if df is None or len(df) < 30:
            return False, 0.0
        close = df["close"].astype(float).values
        high = df["high"].astype(float).values
        low  = df["low"].astype(float).values
        ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
        current_close = close[-1]
        current_ema = ema20[-1]
        recent_high = np.max(high[-21:-1])
        recent_low  = np.min(low[-21:-1])
        if direction == "BUY":
            near_ema = abs(current_close - current_ema) / current_ema <= 0.005
            near_support = abs(current_close - recent_low) / recent_low <= 0.008
            if not (near_ema or near_support):
                return False, 0.0
            pullback_depth = (recent_high - current_close) / recent_high * 100
            if 0.5 <= pullback_depth <= 3.0:
                return True, pullback_depth
        else:
            near_ema = abs(current_close - current_ema) / current_ema <= 0.005
            near_resistance = abs(current_close - recent_high) / recent_high <= 0.008
            if not (near_ema or near_resistance):
                return False, 0.0
            pullback_depth = (current_close - recent_low) / recent_low * 100
            if 0.5 <= pullback_depth <= 3.0:
                return True, pullback_depth
        return False, 0.0

    def _confirm_pullback_entry(self, df: pd.DataFrame, direction: str) -> bool:
        if len(df) < 3:
            return False
        o = df["open"].astype(float).values
        h = df["high"].astype(float).values
        l = df["low"].astype(float).values
        c = df["close"].astype(float).values
        body = abs(c[-1] - o[-1])
        lower_wick = min(o[-1], c[-1]) - l[-1] if direction == "BUY" else h[-1] - max(o[-1], c[-1])
        upper_wick = h[-1] - max(o[-1], c[-1]) if direction == "BUY" else min(o[-1], c[-1]) - l[-1]
        if direction == "BUY":
            is_engulfing = c[-1] > o[-1] and c[-2] < o[-2] and c[-1] > o[-2] and o[-1] < c[-2]
            is_pinbar = lower_wick >= body * 1.5 and upper_wick <= body * 0.5
            return is_engulfing or is_pinbar
        else:
            is_engulfing = c[-1] < o[-1] and c[-2] > o[-2] and c[-1] < o[-2] and o[-1] > c[-2]
            is_pinbar = upper_wick >= body * 1.5 and lower_wick <= body * 0.5
            return is_engulfing or is_pinbar

    def _detect_gainer_reversal(self, df_5m: pd.DataFrame, df_1h: pd.DataFrame | None = None) -> tuple[bool, str]:
        if df_5m is None or len(df_5m) < 20:
            return False, ""
        close = df_5m["close"].astype(float).values
        volume = df_5m["volume"].astype(float).values
        recent_low = np.min(df_5m["low"].astype(float).values[-11:-1])
        if close[-1] < recent_low:
            return True, "support_break"
        vol_ma = np.mean(volume[-21:-1]) if len(volume) >= 21 else np.mean(volume)
        if volume[-1] > vol_ma * 2.0 and close[-1] < close[-2]:
            return True, "volume_spike_down"
        return False, ""

    # ------------------------------------------------------------------
    # Indicator Helpers (unchanged except where noted)
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

    def _top_down(self, df_1d, df_4h, df_1h, df_15m):
        def vote(df, label, w):
            try:
                if df is None or len(df) < 52:
                    return 0, f"{label}~"
                c = df["close"].astype(float)
                e21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
                e50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1])
                pr = float(c.iloc[-1])
                if pr > e50 and e21 > e50:
                    return +w, f"{label}↑"
                elif pr < e50 and e21 < e50:
                    return -w, f"{label}↓"
                else:
                    return 0, f"{label}~"
            except Exception:
                return 0, f"{label}~"

        s1d,  l1d  = vote(df_1d,  "1D",  3)
        s4h,  l4h  = vote(df_4h,  "4H",  3)
        s1h,  l1h  = vote(df_1h,  "1H",  2)
        s15m, l15m = vote(df_15m, "15m", 1)

        htf_score = s1d + s4h + s1h + s15m
        is_major_bull = (s1d > 0 and s4h > 0)
        is_major_bear = (s1d < 0 and s4h < 0)

        adx_4h = 20.0
        try:
            if df_4h is not None and len(df_4h) >= 28:
                h = df_4h["high"].astype(float)
                l = df_4h["low"].astype(float)
                c4 = df_4h["close"].astype(float)
                tr = ((h - l).combine((h - c4.shift(1)).abs(), max)
                             .combine((l - c4.shift(1)).abs(), max))
                atr14 = tr.rolling(14).mean()
                dm_up = (h - h.shift(1)).clip(lower=0)
                dm_dn = (l.shift(1) - l).clip(lower=0)
                di_up = (dm_up.rolling(14).mean() / atr14.replace(0, 1e-10)) * 100
                di_dn = (dm_dn.rolling(14).mean() / atr14.replace(0, 1e-10)) * 100
                dx = (abs(di_up - di_dn) / (di_up + di_dn + 1e-10)) * 100
                adx_4h = float(dx.rolling(14).mean().iloc[-1])
        except Exception:
            pass

        return htf_score, [l1d, l4h, l1h, l15m], is_major_bull, is_major_bear, round(adx_4h, 1)

    def _tqi(self, df, er_len=20, struct_len=20, mom_len=10) -> float:
        try:
            cl = df["close"].astype(float)
            hi = df["high"].astype(float)
            lo = df["low"].astype(float)
            er = (cl.diff(er_len).abs() /
                  cl.diff().abs().rolling(er_len).sum().replace(0, 1e-10)).clip(0, 1)
            hi_n = hi.rolling(struct_len).max()
            lo_n = lo.rolling(struct_len).min()
            struct = (((cl - lo_n) / (hi_n - lo_n + 1e-10) - 0.5).abs() * 2).clip(0, 1)
            win_dir = cl.diff(mom_len)
            bar_dir = cl.diff()
            aligned = sum(
                (((win_dir > 0) & (bar_dir.shift(i) > 0)) |
                 ((win_dir < 0) & (bar_dir.shift(i) < 0))).astype(float)
                for i in range(mom_len)
            )
            mom = (aligned / mom_len).clip(0, 1)
            atr_v = (hi - lo).rolling(14).mean()
            vol_f = ((atr_v / atr_v.rolling(100).mean().replace(0, 1e-10) - 0.6) / 1.2).clip(0, 1)
            return float((er * 0.35 + struct * 0.25 + mom * 0.20 + vol_f * 0.20).clip(0, 1).iloc[-1])
        except Exception:
            return 0.5

    def _rsi(self, df, period=14) -> float:
        try:
            c = df["close"].astype(float)
            d = c.diff()
            gain = d.clip(lower=0).rolling(period).mean()
            loss = (-d.clip(upper=0)).rolling(period).mean()
            return float((100 - 100 / (1 + gain / loss.replace(0, 1e-10))).iloc[-1])
        except Exception:
            return 50.0

    def _volume_analysis(self, df_5m: pd.DataFrame, df_1h: pd.DataFrame | None = None):
        try:
            vols = df_5m["volume"].astype(float)
            close = df_5m["close"].astype(float)
            vol_ma = float(vols.rolling(20).mean().iloc[-1])
            vol_now = float(vols.iloc[-1])
            vol_ratio = vol_now / max(vol_ma, 1e-10)
            vol_spike = vol_ratio >= 3.0

            direction_5m = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            obv = (vols * direction_5m).cumsum()
            obv_ema_fast = float(obv.ewm(span=5, adjust=False).mean().iloc[-1])
            obv_ema_slow = float(obv.ewm(span=20, adjust=False).mean().iloc[-1])
            obv_rising = obv_ema_fast > obv_ema_slow

            if df_1h is not None and len(df_1h) >= 20:
                vols_1h = df_1h["volume"].astype(float)
                vol_1h_ratio = float(vols_1h.iloc[-1]) / max(float(vols_1h.rolling(20).mean().iloc[-1]), 1e-10)
                if vol_1h_ratio < 0.6 and vol_ratio > 2.0:
                    vol_spike = False

            return round(vol_ratio, 2), obv_rising, vol_spike
        except Exception:
            return 1.0, True, False

    def _cci(self, df, period=20) -> float:
        try:
            tp = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
            sma = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            return float(((tp - sma) / (0.015 * mad.replace(0, 1e-10))).iloc[-1])
        except Exception:
            return 0.0


_signal_engine: Optional[SignalEngine] = None


_signal_engine: Optional[SignalEngine] = None

def get_signal_engine(binance_client=None) -> SignalEngine | None:
    global _signal_engine
    if _signal_engine is None:
        # Always load config to get the starting threshold
        from config import AppConfig
        cfg = AppConfig()

        if binance_client is None:
            from src.data.binance_client import BinanceClient
            binance_client = BinanceClient(cfg.binance, cfg.scan)

        _signal_engine = SignalEngine(binance_client, cfg.signal.adaptive_threshold_start)

    return _signal_engine