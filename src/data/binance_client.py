"""
src/data/binance_client.py
───────────────────────────
Market data with automatic fallback:
  Primary  — Binance API endpoints (api.binance.com → api1–api4)
  Fallback — CoinGecko free API (no geo-blocking, no API key)

Gracefully handles DNS failures, timeouts, and offline mode.
"""

import pandas as pd
import requests
from config import BinanceConfig, ScanConfig
from src.utils.logger import get_logger

log = get_logger(__name__)

_BINANCE_ENDPOINTS = [
    "https://api.binance.com/api/v3",
    "https://api1.binance.com/api/v3",
    "https://api2.binance.com/api/v3",
    "https://api3.binance.com/api/v3",
    "https://api4.binance.com/api/v3",
]
_COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# TradFi pairs only exist on the futures API, not spot
_FUT_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
_FUT_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"

# Match list in src/analysis/tradfi_signal_engine.py — keep them in sync
_TRADFI_SYMBOLS = {
    # Metals (Tier 1)
    "XAUUSDT", "XAGUSDT",
    # Stocks (Tier 1)
    "MSTRUSDT", "TSLAUSDT", "AAPLUSDT", "NVDAUSDT",
    "AMZNUSDT", "GOOGLUSDT", "METAUSDT", "MSFTUSDT",
}

_COLS = ["open_time","open","high","low","close","volume",
         "close_time","quote_vol","trades","buy_base","buy_quote","ignore"]
_NUM  = ["open","high","low","close","volume"]

_CG_IDS = {
    "BTCUSDT":"bitcoin","ETHUSDT":"ethereum","SOLUSDT":"solana",
    "BNBUSDT":"binancecoin","XRPUSDT":"ripple","ADAUSDT":"cardano",
    "DOGEUSDT":"dogecoin","AVAXUSDT":"avalanche-2","DOTUSDT":"polkadot",
    "MATICUSDT":"matic-network","LINKUSDT":"chainlink","UNIUSDT":"uniswap",
    "ATOMUSDT":"cosmos","LTCUSDT":"litecoin","NEARUSDT":"near",
    "ARBUSDT":"arbitrum","OPUSDT":"optimism","APTUSDT":"aptos",
}


class BinanceClient:

    def __init__(self, cfg: BinanceConfig, scan_cfg: ScanConfig):
        self._timeout    = cfg.request_timeout
        self._scan       = scan_cfg
        self._session    = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._cg_session = requests.Session()
        self._cg_session.headers.update({"Accept": "application/json", "User-Agent": "BTC-Bot/3"})
        self._binance_ok = True

    # ── Public ────────────────────────────────────────────────

    def get_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        # TradFi (gold, silver, stocks) only exist on futures, not spot
        if symbol in _TRADFI_SYMBOLS:
            return self._futures_klines(symbol, interval, limit)
        if self._binance_ok:
            df = self._binance_klines(symbol, interval, limit)
            if df is not None:
                return df
        return self._cg_klines(symbol, interval, limit)

    def _futures_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Direct futures klines — used for TradFi pairs only."""
        try:
            r = self._session.get(
                _FUT_KLINES_URL,
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=self._timeout,
            )
            if not r.ok:
                log.debug("Futures klines %s %s: HTTP %d", symbol, interval, r.status_code)
                return pd.DataFrame()
            df = pd.DataFrame(r.json(), columns=_COLS)
            df[_NUM] = df[_NUM].apply(pd.to_numeric)
            return df
        except Exception as e:
            log.debug("Futures klines error %s: %s", symbol, e)
            return pd.DataFrame()

    def get_tradfi_pairs(self) -> list[dict]:
        """
        Fetch 24hr ticker for TradFi futures only.
        Returns same shape as get_top_gainers so the scanner can iterate uniformly.
        """
        try:
            r = self._session.get(_FUT_TICKER_URL, timeout=self._timeout)
            if not r.ok:
                log.warning("TradFi ticker fetch: HTTP %d", r.status_code)
                return []
            tickers = r.json()
            return [t for t in tickers if t.get("symbol") in _TRADFI_SYMBOLS]
        except Exception as e:
            log.warning("TradFi ticker error: %s", e)
            return []

    def get_top_gainers(self, limit: int | None = None) -> list[dict]:
        limit = limit or self._scan.top_gainers_count
        if self._binance_ok:
            result = self._binance_tickers(limit, mode="gainers")
            if result is not None:
                return result
        return self._cg_gainers(limit)

    def get_trending_pairs(self, limit: int = 40) -> list[dict]:
        """
        Top liquid USDT pairs by volume — broader universe for trend-pullback strategy.
        Falls back to top gainers if all sources fail.
        """
        if self._binance_ok:
            result = self._binance_tickers(limit, mode="volume")
            if result is not None:
                return result
        # CoinGecko fallback — sort by volume
        cg = self._cg_gainers(limit * 2)
        if cg:
            cg.sort(key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)
            return cg[:limit]
        return []

    # ── Binance ───────────────────────────────────────────────

    def _binance_request(self, path: str, params: dict):
        """Try all Binance endpoints, return first successful response."""
        for base in _BINANCE_ENDPOINTS:
            try:
                r = self._session.get(f"{base}{path}", params=params, timeout=self._timeout)
                if r.status_code == 451:
                    continue
                r.raise_for_status()
                return r
            except requests.HTTPError:
                continue
            except Exception as e:
                log.debug("Binance error on %s: %s", base, e)
                continue
        self._binance_ok = False
        log.warning("All Binance endpoints geo-blocked — switching to CoinGecko fallback")
        return None

    def _binance_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
        r = self._binance_request("/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        if r is None:
            return None
        try:
            df = pd.DataFrame(r.json(), columns=_COLS)
            df[_NUM] = df[_NUM].apply(pd.to_numeric)
            return df
        except Exception:
            return None

    def _binance_tickers(self, limit: int, mode: str = "gainers") -> list[dict] | None:
        r = self._binance_request("/ticker/24hr", {})
        if r is None:
            return None
        try:
            tickers = r.json()
        except Exception:
            return None

        filtered = [
            t for t in tickers
            if t["symbol"].endswith("USDT")
            and not any(s in t["symbol"] for s in self._scan.stable_coins)
            and float(t["quoteVolume"]) > self._scan.min_quote_volume
        ]

        if mode == "gainers":
            filtered = [t for t in filtered
                        if float(t["priceChangePercent"]) >= self._scan.min_gain_percent]
            filtered.sort(key=lambda x: float(x["priceChangePercent"]), reverse=True)
        else:
            # volume mode — exclude extreme pumps/dumps, sort by volume
            filtered = [t for t in filtered
                        if abs(float(t["priceChangePercent"])) <= 20
                        and float(t["quoteVolume"]) > 2_000_000]
            filtered.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)

        return filtered[:limit]

    # ── CoinGecko fallback ────────────────────────────────────

    def _cg_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        coin_id = _CG_IDS.get(symbol)
        if not coin_id:
            return pd.DataFrame()
        days_map = {"1m":1,"5m":1,"15m":2,"1h":7,"4h":30,"1d":90}
        days = days_map.get(interval, 7)
        try:
            r = self._cg_session.get(
                f"{_COINGECKO_BASE}/coins/{coin_id}/ohlc",
                params={"vs_currency":"usd","days":days},
                timeout=15,
            )
            if r.status_code == 429:
                return pd.DataFrame()
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("CoinGecko OHLC error [%s]: %s", symbol, e)
            return pd.DataFrame()
        if not data:
            return pd.DataFrame()
        rows = [{"open_time":c[0],"open":float(c[1]),"high":float(c[2]),
                 "low":float(c[3]),"close":float(c[4]),"volume":1000.0,
                 "close_time":c[0]+3600000,"quote_vol":0,"trades":0,
                 "buy_base":0,"buy_quote":0,"ignore":0}
                for c in data[-limit:]]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _cg_gainers(self, limit: int) -> list[dict]:
        try:
            r = self._cg_session.get(
                f"{_COINGECKO_BASE}/coins/markets",
                params={"vs_currency":"usd","order":"market_cap_desc",
                        "per_page":250,"page":1,"sparkline":False,
                        "price_change_percentage":"24h"},
                timeout=15,
            )
            if r.status_code == 429:
                return []
            r.raise_for_status()
            coins = r.json()
        except Exception as e:
            log.error("CoinGecko markets error: %s", e)
            return []
        result = []
        for c in coins:
            pct = c.get("price_change_percentage_24h") or 0
            vol = c.get("total_volume") or 0
            price = c.get("current_price") or 0
            sym = c.get("symbol","").upper() + "USDT"
            if any(s in sym for s in self._scan.stable_coins):
                continue
            if vol < self._scan.min_quote_volume:
                continue
            result.append({
                "symbol": sym,
                "lastPrice": str(price),
                "priceChangePercent": str(pct),
                "quoteVolume": str(vol),
                "current_price": price,
            })
        return result[:limit]