"""
src/analysis/tradfi_signal_engine.py
─────────────────────────────────────
Signal engine for Binance USDⓈ-M TradFi pairs:
  Metals : XAU, XAG, XPT, XPD  (Gold, Silver, Platinum, Palladium)
  Stocks : TSLA, INTC, HOOD, MSTR, AMZN, etc.

Differences vs SignalEngine (crypto):
  • No BTC strength gate    — TradFi tracks rates / equities, not BTC
  • No CryptoPanic news     — wrong universe (no TSLA earnings coverage)
  • No BTC correlation      — actively misleading for these assets
  • Wider ATR tolerance     — metals 0.2-1.5%, stocks 0.8-5%
  • US session bonus        — favours stock trades during NYSE hours

Returns the SAME `Signal` dataclass as SignalEngine, so the entire
execution layer (binance_trader, scanner auto-trade, dashboard, DB
records) treats TradFi signals identically to crypto signals.
"""

from datetime import datetime, timezone
import pandas as pd

from config import RiskConfig, SignalConfig
from src.analysis.btc_strength import BtcStrength
from src.analysis.indicators import (
    atr, bollinger_bands, ema, ema_value, macd,
    obv_trend, rsi, stochastic_rsi, volume_ratio,
)
from src.analysis.signal_engine import Signal   # reuse the dataclass
from src.data.binance_client import BinanceClient
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Symbol classification — Tier 1 only ───────────────────────────
# Highest liquidity, cleanest technicals, deepest order books.
# Tier 2/3 (XPT, XPD, INTC, COIN, HOOD, AMD, PLTR) intentionally excluded
METALS = {
    "XAUUSDT",   # Gold     — 76M volume, slowest/cleanest trends
    "XAGUSDT",   # Silver   — 31M volume, more volatile than gold
}
STOCKS = {
    "MSTRUSDT",  # Strategy (institutional BTC proxy)
    "TSLAUSDT",  # Tesla
    "AAPLUSDT",  # Apple
    "NVDAUSDT",  # Nvidia
    "AMZNUSDT",  # Amazon
    "GOOGLUSDT", # Alphabet
    "METAUSDT",  # Meta
    "MSFTUSDT",  # Microsoft
}
TRADFI_ALL = METALS | STOCKS


# ── Thresholds ─────────────────────────────────────────────────────
ULTRA_THRESHOLD  = 6.0      # lower than crypto (7.0) — fewer bonuses available
STRONG_THRESHOLD = 4.5
MIN_SLOPE        = 0.0008   # gentler than crypto (0.0015) — slower assets

RSI_MAX_BUY  = 65
RSI_MIN_BUY  = 35
RSI_MIN_SELL = 45
RSI_MAX_SELL = 72

# US equity market hours (UTC) — NYSE 13:30-20:00 UTC
US_MARKET_OPEN_UTC  = 13
US_MARKET_CLOSE_UTC = 20


class TradFiSignalEngine:
    """
    Drop-in companion to SignalEngine. Same `analyze(ticker, btc)` signature
    so the scanner can route symbols without special-casing return types.
    """

    def __init__(self, binance: BinanceClient,
                 sig_cfg: SignalConfig, risk_cfg: RiskConfig):
        self._b = binance
        self._s = sig_cfg
        self._r = risk_cfg

    # ───────────────────────────────────────────────────────────────
    # Public — same signature as SignalEngine.analyze
    # ───────────────────────────────────────────────────────────────

    def analyze(self, ticker: dict, btc: BtcStrength | None = None) -> Signal | None:
        """
        `btc` is accepted for interface compatibility but intentionally ignored.
        TradFi assets do not track BTC strength.
        """
        sym   = ticker["symbol"]
        gain  = float(ticker.get("priceChangePercent", 0) or 0)
        price = float(ticker.get("lastPrice", 0) or 0)
        if price <= 0:
            return None

        df_1d = self._b.get_klines(sym, "1d", 60)
        df_4h = self._b.get_klines(sym, "4h", 100)
        df_1h = self._b.get_klines(sym, "1h", 48)

        if df_4h.empty or len(df_4h) < 30:
            return None
        if df_1d.empty or len(df_1d) < 20:
            return None

        close_1d = df_1d["close"]
        close_4h = df_4h["close"]
        close_1h = df_1h["close"] if not df_1h.empty and len(df_1h) >= 10 else close_4h
        price_now = float(close_4h.iloc[-1])

        rsi_d  = rsi(close_1d)
        rsi_4h = rsi(close_4h)
        rsi_1h = rsi(close_1h)
        e20_d  = ema_value(close_1d, 20)
        e50_d  = ema_value(close_1d, 50)
        e20_4h = ema_value(close_4h, 20)
        e50_4h = ema_value(close_4h, 50)
        e200_d = ema_value(close_1d, 200) if len(close_1d) >= 200 else e50_d
        slope_d  = self._slope(ema(close_1d, 20), 5)
        slope_4h = self._slope(ema(close_4h, 20), 6)
        vr_4h = volume_ratio(df_4h["volume"], 20)
        vr_1d = volume_ratio(df_1d["volume"], 20)
        atr_4h = atr(df_4h, 14)
        bb_4h  = bollinger_bands(close_4h)
        srsi_4h = stochastic_rsi(close_4h)
        obv_4h = obv_trend(df_4h)
        ml, ms = macd(close_4h)
        swing_low  = float(df_4h["low"].rolling(20).min().iloc[-1])
        swing_high = float(df_4h["high"].rolling(20).max().iloc[-1])
        is_stock = sym in STOCKS

        buy_score, buy_factors = self._buy_score(
            price_now, rsi_d, rsi_4h, rsi_1h,
            e20_d, e50_d, e200_d, e20_4h, e50_4h,
            slope_d, slope_4h, vr_4h, vr_1d, atr_4h,
            obv_4h, bb_4h, srsi_4h, ml, ms, is_stock,
        )
        sell_score, sell_factors = self._sell_score(
            price_now, rsi_d, rsi_4h, rsi_1h,
            e20_d, e50_d, e20_4h, e50_4h,
            slope_d, slope_4h, vr_4h, vr_1d, atr_4h,
            obv_4h, bb_4h, srsi_4h, ml, ms, is_stock,
        )

        if buy_score >= STRONG_THRESHOLD and buy_score > sell_score:
            return self._build("BUY", sym, buy_score, buy_factors,
                               price_now, gain, rsi_1h, rsi_4h, rsi_d,
                               srsi_4h, bb_4h["pct_b"], atr_4h, swing_low,
                               is_stock)
        if sell_score >= STRONG_THRESHOLD and sell_score > buy_score:
            return self._build("SELL", sym, sell_score, sell_factors,
                               price_now, gain, rsi_1h, rsi_4h, rsi_d,
                               srsi_4h, bb_4h["pct_b"], atr_4h, swing_high,
                               is_stock)
        return None

    # ───────────────────────────────────────────────────────────────
    # BUY scoring
    # ───────────────────────────────────────────────────────────────

    def _buy_score(self, price, rsi_d, rsi_4h, rsi_1h,
                   e20_d, e50_d, e200_d, e20_4h, e50_4h,
                   slope_d, slope_4h, vr_4h, vr_1d, atr_4h,
                   obv, bb, srsi, ml, ms, is_stock):
        score, factors = 0.0, []

        # ── HARD GATES ── (no BTC gate, just price action)
        if price < e50_d:           return 0.0, []
        if slope_d < MIN_SLOPE:     return 0.0, []
        if rsi_4h > RSI_MAX_BUY:    return 0.0, []
        if rsi_4h < RSI_MIN_BUY:    return 0.0, []

        # 1. Daily trend structure (0-2.5)
        if price > e20_d > e50_d > e200_d:
            score += 2.5; factors.append("✅ DAILY: full uptrend stack")
        elif price > e20_d > e50_d:
            score += 1.8; factors.append("✅ DAILY: above EMA20/50")
        elif price > e50_d:
            score += 1.0; factors.append("⚠️ DAILY: above EMA50 only")

        # 2. Daily RSI (0-1.2)
        if 45 <= rsi_d <= 62:
            score += 1.2; factors.append(f"✅ Daily RSI {rsi_d:.0f} — healthy")
        elif 35 <= rsi_d < 45:
            score += 0.7; factors.append(f"✅ Daily RSI {rsi_d:.0f} — pullback zone")
        elif rsi_d > 70:
            score -= 0.5

        # 3. Daily slope (0-1.0)
        if slope_d > 0.004:
            score += 1.0; factors.append("✅ Strong daily upslope")
        elif slope_d > MIN_SLOPE:
            score += 0.5

        # 4. 4H trend confirmation (0-1.5)
        if price > e20_4h > e50_4h and slope_4h > MIN_SLOPE:
            score += 1.5; factors.append("✅ 4H uptrend confirmed")
        elif price > e20_4h:
            score += 0.8

        # 5. 4H RSI pullback (0-1.2)
        if 40 <= rsi_4h <= 55:
            score += 1.2; factors.append(f"✅ 4H RSI {rsi_4h:.0f} — pullback zone")
        elif rsi_4h <= RSI_MAX_BUY:
            score += 0.6

        # 6. Volume on pullback (0-1.0)
        if vr_4h < 0.8:
            score += 1.0; factors.append(f"✅ Low pullback volume ({vr_4h:.2f}x)")
        elif vr_4h < 1.0:
            score += 0.5

        # 7. MACD (0-1.0)
        if ml > ms and ml > 0:
            score += 1.0; factors.append("✅ MACD bullish above zero")
        elif ml > ms:
            score += 0.5

        # 8. OBV (0-0.8)
        if obv == "rising":
            score += 0.8; factors.append("✅ OBV rising — accumulation")

        # 9. StochRSI (0-0.7)
        if srsi["k"] < 40 and srsi["k"] > srsi["d"]:
            score += 0.7
            factors.append(f"✅ StochRSI turning up ({srsi['k']:.0f})")

        # 10. 1H timing (0-0.6)
        if 38 <= rsi_1h <= 58:
            score += 0.6; factors.append(f"✅ 1H RSI {rsi_1h:.0f} — entry timing")

        # 11. Bollinger position (0-0.5)
        if bb["pct_b"] < 0.45:
            score += 0.5; factors.append("✅ Lower half of Bollinger")

        # 12. Daily volume (0-0.5)
        if vr_1d >= 1.2:
            score += 0.5; factors.append(f"✅ Daily volume strong ({vr_1d:.1f}x)")

        # 13. ATR quality — adjusted per asset class
        s, f = self._atr_quality(atr_4h, price, is_stock)
        score += s; factors += f

        # 14. US market session bonus (stocks only)
        if is_stock:
            s, f = self._us_session_bonus()
            score += s; factors += f

        return round(score, 2), factors

    # ───────────────────────────────────────────────────────────────
    # SELL scoring
    # ───────────────────────────────────────────────────────────────

    def _sell_score(self, price, rsi_d, rsi_4h, rsi_1h,
                    e20_d, e50_d, e20_4h, e50_4h,
                    slope_d, slope_4h, vr_4h, vr_1d, atr_4h,
                    obv, bb, srsi, ml, ms, is_stock):
        score, factors = 0.0, []

        if price > e50_d:                                   return 0.0, []
        if slope_d > -MIN_SLOPE:                            return 0.0, []
        if rsi_4h < RSI_MIN_SELL or rsi_4h > RSI_MAX_SELL:  return 0.0, []

        # 1. Daily downtrend (0-2.5)
        if price < e20_d < e50_d:
            score += 2.5; factors.append("✅ DAILY: full downtrend stack")
        elif price < e50_d:
            score += 1.0; factors.append("⚠️ DAILY: below EMA50")

        # 2. Daily RSI (0-1.2)
        if 50 <= rsi_d <= 62:
            score += 1.2
            factors.append(f"✅ Daily RSI {rsi_d:.0f} — bounce into resistance")
        elif rsi_d > 62:
            score += 0.5
        elif rsi_d < 38:
            score -= 0.5

        # 3. Downslope (0-1.0)
        if slope_d < -0.004:
            score += 1.0; factors.append("✅ Strong daily downslope")
        elif slope_d < -MIN_SLOPE:
            score += 0.5

        # 4. 4H downtrend (0-1.5)
        if price < e20_4h < e50_4h and slope_4h < -MIN_SLOPE:
            score += 1.5; factors.append("✅ 4H downtrend confirmed")
        elif price < e20_4h:
            score += 0.8

        # 5. 4H RSI bounce (0-1.2)
        if 52 <= rsi_4h <= 65:
            score += 1.2; factors.append(f"✅ 4H RSI {rsi_4h:.0f} — bounce")

        # 6. Volume spike on bounce (0-1.0)
        if vr_4h >= 1.5:
            score += 1.0; factors.append(f"✅ Volume spike ({vr_4h:.1f}x) — distribution")
        elif vr_4h >= 1.1:
            score += 0.5

        # 7. MACD bearish (0-1.0)
        if ml < ms and ml < 0:
            score += 1.0; factors.append("✅ MACD bearish below zero")
        elif ml < ms:
            score += 0.5

        # 8. OBV (0-0.8)
        if obv == "falling":
            score += 0.8; factors.append("✅ OBV falling")

        # 9. StochRSI (0-0.7)
        if srsi["k"] > 60 and srsi["k"] < srsi["d"]:
            score += 0.7
            factors.append(f"✅ StochRSI turning down ({srsi['k']:.0f})")

        # 10. 1H trigger (0-0.6)
        if rsi_1h >= 60:
            score += 0.6; factors.append(f"✅ 1H RSI {rsi_1h:.0f} — local OB")

        # 11. Bollinger position (0-0.5)
        if bb["pct_b"] > 0.55:
            score += 0.5; factors.append("✅ Upper half of Bollinger")

        # 12. Daily volume (0-0.5)
        if vr_1d >= 1.2:
            score += 0.5; factors.append(f"✅ Daily volume strong ({vr_1d:.1f}x)")

        s, f = self._atr_quality(atr_4h, price, is_stock)
        score += s; factors += f
        if is_stock:
            s, f = self._us_session_bonus()
            score += s; factors += f

        return round(score, 2), factors

    # ───────────────────────────────────────────────────────────────
    # Asset-aware filters
    # ───────────────────────────────────────────────────────────────

    def _atr_quality(self, atr_val, price, is_stock):
        """Wider tolerance than crypto — stocks 0.8-5%, metals 0.2-1.5%."""
        if price <= 0 or atr_val <= 0:
            return 0.0, []
        atr_pct = atr_val / price * 100
        if is_stock:
            if 0.8 <= atr_pct <= 5.0:
                return 0.4, [f"✅ ATR: stock volatility ideal ({atr_pct:.1f}%)"]
            if atr_pct > 7.0:
                return -0.3, [f"⚠️ ATR: stock too volatile ({atr_pct:.1f}%) — earnings risk"]
        else:
            if 0.2 <= atr_pct <= 1.5:
                return 0.4, [f"✅ ATR: metal volatility ideal ({atr_pct:.1f}%)"]
            if atr_pct > 3.0:
                return -0.3, [f"⚠️ ATR: metal too volatile ({atr_pct:.1f}%)"]
        return 0.0, []

    def _us_session_bonus(self):
        """+0.5 during NYSE hours — that's when stock perpetuals actually trade."""
        try:
            h = datetime.now(timezone.utc).hour
            if US_MARKET_OPEN_UTC <= h < US_MARKET_CLOSE_UTC:
                return 0.5, ["✅ US market open — active equity liquidity"]
        except Exception:
            pass
        return 0.0, []

    # ───────────────────────────────────────────────────────────────
    # Build final Signal — same dataclass as crypto
    # ───────────────────────────────────────────────────────────────

    def _build(self, sig_type, sym, confluence, factors,
               price, gain, rsi_1h, rsi_4h, rsi_d,
               srsi, bb_pct_b, atr_4h, swing_level, is_stock):
        r = self._r
        is_buy = sig_type == "BUY"

        if confluence >= ULTRA_THRESHOLD:
            grade = f"ULTRA {'🟢🟢🟢' if is_buy else '🔴🔴🔴'}"
            hold = "4-7 days"
            base = 80
        elif confluence >= STRONG_THRESHOLD:
            grade = f"STRONG {'🟢🟢' if is_buy else '🔴🔴'}"
            hold = "2-4 days"
            base = 70
        else:
            grade = f"STANDARD {'🟢' if is_buy else '🔴'}"
            hold = "1-2 days"
            base = 60

        # Confidence boost scales with how far above STRONG threshold we are
        bonus = min(15, int((confluence - STRONG_THRESHOLD) * 4))
        confidence = min(92, max(55, base + bonus))

        # ATR-based SL — same logic as crypto engine
        sl_buf = atr_4h * 1.5
        if is_buy:
            tp1 = round(price * (1 + r.tp1_pct / 100), 8)
            tp2 = round(price * (1 + r.tp2_pct / 100), 8)
            tp3 = round(price * (1 + r.tp3_pct / 100), 8)
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
            action = "Sell / Short"

        kind = "STOCK" if is_stock else "METAL"
        return Signal(
            symbol=sym, signal=sig_type, grade=grade,
            confidence=confidence, action=action,
            price=price, gain_24h=gain,
            rsi_1h=rsi_1h, rsi_4h=rsi_4h, rsi_daily=rsi_d,
            tp1=tp1, tp2=tp2, tp3=tp3, sl=sl,
            factors=factors,
            strategies_hit=[f"TradFi-{kind}"],
            btc_score=0,                  # not applicable
            btc_trend="N/A",
            confluence=confluence, hold_days=hold,
            news_sentiment="neutral",     # no news source for TradFi
            stoch_rsi=srsi, bb_pct_b=bb_pct_b, williams=0.0,
        )

    # ───────────────────────────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────────────────────────

    @staticmethod
    def _slope(series: pd.Series, lookback: int = 5) -> float:
        if len(series) < lookback + 1:
            return 0.0
        recent = series.iloc[-lookback:]
        v0 = float(recent.iloc[0])
        return float((recent.iloc[-1] - v0) / v0 / lookback) if v0 != 0 else 0.0