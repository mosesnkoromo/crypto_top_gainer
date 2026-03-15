"""src/data/binance_client.py"""
import pandas as pd
import requests
from config import BinanceConfig, ScanConfig
from src.utils.logger import get_logger

log = get_logger(__name__)

_COLS = ["open_time","open","high","low","close","volume",
         "close_time","quote_vol","trades","buy_base","buy_quote","ignore"]
_NUM  = ["open","high","low","close","volume"]


class BinanceClient:

    def __init__(self, cfg: BinanceConfig, scan_cfg: ScanConfig):
        self._base    = cfg.base_url
        self._timeout = cfg.request_timeout
        self._scan    = scan_cfg
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    def get_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        try:
            r = self._session.get(
                f"{self._base}/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=self._timeout,
            )
            r.raise_for_status()
            df = pd.DataFrame(r.json(), columns=_COLS)
            df[_NUM] = df[_NUM].apply(pd.to_numeric)
            return df
        except Exception as e:
            log.error("Klines error [%s]: %s", symbol, e)
            return pd.DataFrame()

    def get_top_gainers(self, limit: int | None = None) -> list[dict]:
        limit = limit or self._scan.top_gainers_count
        try:
            r = self._session.get(f"{self._base}/ticker/24hr", timeout=self._timeout)
            r.raise_for_status()
            tickers = r.json()
        except Exception as e:
            log.error("Top gainers error: %s", e)
            return []
        filtered = [
            t for t in tickers
            if t["symbol"].endswith("USDT")
            and not any(s in t["symbol"] for s in self._scan.stable_coins)
            and float(t["priceChangePercent"]) >= self._scan.min_gain_percent
            and float(t["quoteVolume"]) > self._scan.min_quote_volume
        ]
        filtered.sort(key=lambda x: float(x["priceChangePercent"]), reverse=True)
        return filtered[:limit]
