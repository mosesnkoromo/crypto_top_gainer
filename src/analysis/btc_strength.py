"""src/analysis/btc_strength.py"""
from dataclasses import dataclass
import pandas as pd
from config import ScanConfig
from src.analysis.indicators import ema_value, macd, rsi, volume_ratio
from src.data.binance_client import BinanceClient
from src.utils.logger import get_logger

log = get_logger(__name__)

_LABELS = [(72,"VERY STRONG BULL 🟢🟢"),(58,"BULL 🟢"),(43,"NEUTRAL ⚪"),(28,"BEAR 🔴"),(0,"VERY WEAK BEAR 🔴🔴")]


@dataclass
class BtcStrength:
    score: int; rsi: float; trend: str; price: float
    ema20: float; ema50: float; vol_ratio: float; macd_bull: bool

    def to_dict(self):
        return {"score":self.score,"rsi":self.rsi,"trend":self.trend,"price":self.price,
                "ema20":self.ema20,"ema50":self.ema50,"vol_ratio":self.vol_ratio,"macd_bull":self.macd_bull}

    @classmethod
    def fallback(cls):
        return cls(50,50.0,"UNKNOWN ⚪",0.0,0.0,0.0,1.0,False)


def _label(score):
    for t, l in _LABELS:
        if score >= t: return l
    return "VERY WEAK BEAR 🔴🔴"


class BtcStrengthEngine:

    def __init__(self, binance: BinanceClient, scan_cfg: ScanConfig):
        self._b = binance
        self._tf = scan_cfg.timeframe

    def calculate(self) -> BtcStrength:
        df = self._b.get_klines("BTCUSDT", self._tf, 100)
        if df.empty:
            log.warning("BTC data unavailable — fallback score 50")
            return BtcStrength.fallback()

        close = df["close"]
        price = float(close.iloc[-1])
        r     = rsi(close)
        e20   = ema_value(close, 20)
        e50   = ema_value(close, 50)
        ml,ms = macd(close)
        vr    = volume_ratio(df["volume"])

        score = self._rsi_pts(r) + self._ema_pts(price,e20,e50) + self._macd_pts(ml,ms) + self._vol_pts(vr)
        result = BtcStrength(score,r,_label(score),price,round(e20,2),round(e50,2),vr,ml>ms)
        log.info("BTC Strength: %d/100 — %s (RSI %.1f)", score, result.trend, r)
        return result

    @staticmethod
    def _rsi_pts(r):
        if 45<=r<=65: return 30
        if r>65:      return 15
        if 35<=r<45:  return 20
        return 5

    @staticmethod
    def _ema_pts(p,e20,e50):
        if p>e20>e50: return 30
        if p>e20:     return 20
        if p>e50:     return 10
        return 0

    @staticmethod
    def _macd_pts(ml,ms):
        if ml>ms and ml>0: return 20
        if ml>ms:          return 12
        if ml>0:           return 8
        return 0

    @staticmethod
    def _vol_pts(vr):
        if vr>=1.5: return 20
        if vr>=1.0: return 12
        if vr>=0.7: return 6
        return 0
