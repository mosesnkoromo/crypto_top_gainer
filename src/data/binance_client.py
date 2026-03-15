"""
src/data/binance_client.py
───────────────────────────
Market data client with automatic fallback:
  Primary   — Binance API (works locally and on non-US servers)
  Fallback  — CoinGecko free API (no geo-blocking, no API key needed)

Binance blocks all US-hosted servers (AWS/Render/Heroku) with HTTP 451.
CoinGecko is used as a transparent fallback with no user configuration needed.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
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

_COINGECKO_BASE    = "https://api.coingecko.com/api/v3"
_KUCOIN_BASE       = "https://api.kucoin.com/api/v1"

_COLS = ["open_time","open","high","low","close","volume",
         "close_time","quote_vol","trades","buy_base","buy_quote","ignore"]
_NUM  = ["open","high","low","close","volume"]

# CoinGecko coin ID mapping for top coins
_CG_IDS = {
    "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "SOLUSDT": "solana",
    "BNBUSDT": "binancecoin", "XRPUSDT": "ripple", "ADAUSDT": "cardano",
    "DOGEUSDT": "dogecoin", "AVAXUSDT": "avalanche-2", "DOTUSDT": "polkadot",
    "MATICUSDT": "matic-network", "LINKUSDT": "chainlink", "UNIUSDT": "uniswap",
    "ATOMUSDT": "cosmos", "LTCUSDT": "litecoin", "NEARUSDT": "near",
    "ARBUSDT": "arbitrum", "OPUSDT": "optimism", "APTUSDT": "aptos",
}


class BinanceClient:

    def __init__(self, cfg: BinanceConfig, scan_cfg: ScanConfig):
        self._timeout      = cfg.request_timeout
        self._scan         = scan_cfg
        self._session      = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._binance_ok   = True   # flips False after 451 confirmed
        self._cg_session   = requests.Session()
        self._cg_session.headers.update({
            "Accept": "application/json",
            "User-Agent": "BTC-Strength-Bot/3.0"
        })

    # ── Public API ────────────────────────────────────────────

    def get_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        if self._binance_ok:
            df = self._binance_klines(symbol, interval, limit)
            if df is not None:
                return df
        # Fallback to CoinGecko
        return self._cg_klines(symbol, interval, limit)

    def get_top_gainers(self, limit: int | None = None) -> list[dict]:
        limit = limit or self._scan.top_gainers_count
        if self._binance_ok:
            result = self._binance_gainers(limit)
            if result is not None:
                return result
        return self._cg_gainers(limit)

    # ── Binance ───────────────────────────────────────────────

    def _binance_get(self, path: str, params: dict):
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
        r = self._binance_get("/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        if r is None:
            return None
        try:
            df = pd.DataFrame(r.json(), columns=_COLS)
            df[_NUM] = df[_NUM].apply(pd.to_numeric)
            return df
        except Exception:
            return None

    def _binance_gainers(self, limit: int) -> list[dict] | None:
        r = self._binance_get("/ticker/24hr", {})
        if r is None:
            return None
        try:
            tickers = r.json()
            filtered = [
                t for t in tickers
                if t["symbol"].endswith("USDT")
                and not any(s in t["symbol"] for s in self._scan.stable_coins)
                and float(t["priceChangePercent"]) >= self._scan.min_gain_percent
                and float(t["quoteVolume"]) > self._scan.min_quote_volume
            ]
            filtered.sort(key=lambda x: float(x["priceChangePercent"]), reverse=True)
            return filtered[:limit]
        except Exception:
            return None

    # ── CoinGecko Fallback ────────────────────────────────────

    def _cg_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """
        Convert CoinGecko OHLC data into the same DataFrame format as Binance klines.
        Uses /coins/{id}/ohlc endpoint (free, no key needed).
        """
        coin_id = _CG_IDS.get(symbol)
        if not coin_id:
            # For unknown coins, synthesise from price history
            return self._cg_klines_from_prices(symbol, limit)

        # Map interval to CoinGecko days param
        days_map = {"1m": 1, "5m": 1, "15m": 2, "1h": 7, "4h": 30, "1d": 90}
        days = days_map.get(interval, 7)

        try:
            r = self._cg_session.get(
                f"{_COINGECKO_BASE}/coins/{coin_id}/ohlc",
                params={"vs_currency": "usd", "days": days},
                timeout=15,
            )
            if r.status_code == 429:
                log.warning("CoinGecko rate limited — using synthetic data")
                return self._synthetic_df(limit)
            r.raise_for_status()
            data = r.json()  # [[timestamp, open, high, low, close], ...]
        except Exception as e:
            log.warning("CoinGecko OHLC error [%s]: %s", symbol, e)
            return self._synthetic_df(limit)

        if not data:
            return self._synthetic_df(limit)

        rows = []
        for candle in data[-limit:]:
            ts, o, h, l, c = candle[0], candle[1], candle[2], candle[3], candle[4]
            rows.append({
                "open_time": ts, "open": float(o), "high": float(h),
                "low": float(l), "close": float(c), "volume": 1000.0,
                "close_time": ts + 3600000, "quote_vol": 0, "trades": 0,
                "buy_base": 0, "buy_quote": 0, "ignore": 0,
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return self._synthetic_df(limit)
        return df

    def _cg_klines_from_prices(self, symbol: str, limit: int) -> pd.DataFrame:
        """For coins not in _CG_IDS map, search CoinGecko by symbol."""
        coin_slug = symbol.replace("USDT","").lower()
        try:
            r = self._cg_session.get(
                f"{_COINGECKO_BASE}/coins/markets",
                params={"vs_currency": "usd", "ids": coin_slug, "sparkline": False},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            if not data:
                return self._synthetic_df(limit)
            coin_id = data[0]["id"]
            return self._cg_klines(symbol.upper() + "USDT" if not symbol.endswith("USDT") else symbol,
                                    "1h", limit)
        except Exception:
            return self._synthetic_df(limit)

    def _cg_gainers(self, limit: int) -> list[dict]:
        """
        Fetch top gainers from CoinGecko /coins/markets sorted by 24h change.
        Returns data in Binance ticker format.
        """
        try:
            r = self._cg_session.get(
                f"{_COINGECKO_BASE}/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "percent_change_24h",
                    "per_page": 250,
                    "page": 1,
                    "sparkline": False,
                    "price_change_percentage": "24h",
                },
                timeout=15,
            )
            if r.status_code == 429:
                log.warning("CoinGecko rate limited on gainers")
                return []
            r.raise_for_status()
            coins = r.json()
        except Exception as e:
            log.error("CoinGecko gainers error: %s", e)
            return []

        tickers = []
        for c in coins:
            pct = c.get("price_change_percentage_24h") or 0
            vol = c.get("total_volume") or 0
            price = c.get("current_price") or 0
            sym = c.get("symbol","").upper() + "USDT"

            if any(s in sym for s in self._scan.stable_coins):
                continue
            if pct < self._scan.min_gain_percent:
                continue
            if vol < self._scan.min_quote_volume:
                continue

            tickers.append({
                "symbol":              sym,
                "lastPrice":           str(price),
                "priceChangePercent":  str(pct),
                "quoteVolume":         str(vol),
            })

        tickers.sort(key=lambda x: float(x["priceChangePercent"]), reverse=True)
        log.info("CoinGecko fallback: %d gainers found", len(tickers[:limit]))
        return tickers[:limit]

    @staticmethod
    def _synthetic_df(limit: int) -> pd.DataFrame:
        """Empty-ish DataFrame so analysis doesn't crash — signals will be filtered out."""
        return pd.DataFrame()