"""src/data/binance_client.py
Binance REST client with automatic endpoint fallback.
Render/AWS servers are blocked by api.binance.com (HTTP 451).
Falls back through alternative endpoints automatically.
"""
import pandas as pd
import requests
from config import BinanceConfig, ScanConfig
from src.utils.logger import get_logger

log = get_logger(__name__)

# Binance serves these interchangeably — try each on 451/timeout
_ENDPOINTS = [
    "https://api.binance.com/api/v3",
    "https://api1.binance.com/api/v3",
    "https://api2.binance.com/api/v3",
    "https://api3.binance.com/api/v3",
    "https://api4.binance.com/api/v3",
]

_COLS = ["open_time","open","high","low","close","volume",
         "close_time","quote_vol","trades","buy_base","buy_quote","ignore"]
_NUM  = ["open","high","low","close","volume"]


class BinanceClient:

    def __init__(self, cfg: BinanceConfig, scan_cfg: ScanConfig):
        self._timeout  = cfg.request_timeout
        self._scan     = scan_cfg
        self._session  = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._base     = _ENDPOINTS[0]   # active endpoint, rotates on 451

    def _get(self, path: str, params: dict) -> requests.Response | None:
        """Try each endpoint until one works. Rotates on 451."""
        for base in _ENDPOINTS:
            try:
                resp = self._session.get(
                    f"{base}{path}", params=params, timeout=self._timeout
                )
                if resp.status_code == 451:
                    log.warning("Binance 451 geo-block on %s — trying next endpoint", base)
                    continue
                resp.raise_for_status()
                self._base = base   # remember working endpoint
                return resp
            except requests.HTTPError:
                continue
            except requests.RequestException as e:
                log.warning("Binance error on %s: %s — trying next", base, e)
                continue
        log.error("All Binance endpoints failed for %s", path)
        return None

    def get_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        resp = self._get("/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        if resp is None:
            return pd.DataFrame()
        try:
            df = pd.DataFrame(resp.json(), columns=_COLS)
            df[_NUM] = df[_NUM].apply(pd.to_numeric)
            return df
        except Exception as e:
            log.error("Klines parse error [%s]: %s", symbol, e)
            return pd.DataFrame()

    def get_top_gainers(self, limit: int | None = None) -> list[dict]:
        limit = limit or self._scan.top_gainers_count
        resp  = self._get("/ticker/24hr", {})
        if resp is None:
            return []
        try:
            tickers = resp.json()
        except Exception as e:
            log.error("Top gainers parse error: %s", e)
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