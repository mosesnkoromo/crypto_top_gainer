"""
src/analysis/news_engine.py
────────────────────────────
Crypto news from cryptocurrency.cv — completely free, no API key needed.
Used internally to boost/reduce signal confluence scores only.

API docs: https://cryptocurrency.cv
Endpoint: GET https://cryptocurrency.cv/api/archive?ticker=BTC&limit=50
"""

import requests
from datetime import datetime, timezone
from src.utils.logger import get_logger

log = get_logger(__name__)

_BASE_URL  = "https://cryptocurrency.cv/api/archive"
_FALLBACK  = "https://api.coingecko.com/api/v3/news"   # secondary fallback

# Bearish keywords to detect negative sentiment
_BEARISH = {"hack","exploit","crash","ban","sued","scam","fraud","collapse",
            "plunge","dump","bear","sell-off","liquidat","bankrupt","rug","exit scam"}
_BULLISH = {"rally","surge","bull","breakout","adoption","partnership","launch",
            "upgrade","etf","approval","institutional","record","high","milestone"}


class NewsEngine:

    def __init__(self, api_key: str = ""):
        self._api_key    = api_key.strip()  # kept for backward compat, not used
        self._cache:     list[dict] = []
        self._cached_at: datetime | None = None
        self._ttl_min    = 30

    def get_news(self, limit: int = 50) -> list[dict]:
        """Return cached news. Fetches once per 30 min."""
        if self._is_valid():
            return self._cache[:limit]
        fetched = self._fetch_cv(limit) or self._fetch_coingecko(limit)
        self._cache     = fetched
        self._cached_at = datetime.now(timezone.utc)
        log.info("News cache refreshed: %d articles", len(self._cache))
        return self._cache[:limit]

    def get_sentiment_for(self, symbol: str) -> dict:
        """
        Score sentiment for a specific coin from cached news.
        Reads from cache — never triggers a new fetch.
        """
        if not self._cache:
            return {"label": "neutral", "articles": 0, "score": 0}

        coin = symbol.replace("USDT", "").replace("BTC", "").upper()
        relevant = [
            n for n in self._cache
            if coin in n.get("currencies", "").upper()
            or coin.lower() in n.get("title", "").lower()
        ]
        if not relevant:
            return {"label": "neutral", "articles": 0, "score": 0}

        pos   = sum(1 for n in relevant if n["sentiment"] == "positive")
        neg   = sum(1 for n in relevant if n["sentiment"] == "negative")
        total = len(relevant)
        score = round((pos - neg) / total * 100) if total else 0
        label = "positive" if score >= 25 else ("negative" if score <= -25 else "neutral")
        return {"label": label, "score": score, "articles": total}

    # ── Internal ──────────────────────────────────────────────

    def _is_valid(self) -> bool:
        if not self._cached_at or not self._cache:
            return False
        age = (datetime.now(timezone.utc) - self._cached_at).total_seconds() / 60
        return age < self._ttl_min

    def _fetch_cv(self, limit: int) -> list[dict]:
        """cryptocurrency.cv — free, no key. Falls back gracefully on 403/429."""
        try:
            resp = requests.get(
                _BASE_URL,
                params={"limit": min(limit, 100)},
                timeout=12,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            if resp.status_code == 403:
                log.debug("cryptocurrency.cv 403 — trying CoinGecko fallback")
                return []   # silently fall through to CoinGecko
            if resp.status_code == 429:
                log.debug("cryptocurrency.cv rate limited — using cache or CoinGecko")
                return []
            if resp.status_code != 200:
                log.debug("cryptocurrency.cv HTTP %d — using fallback", resp.status_code)
                return []

            text = resp.text.strip()
            if not text or not text.startswith(("[", "{")):
                log.warning("cryptocurrency.cv returned non-JSON")
                return []

            data = resp.json()
            # Handle both list and {"data": [...]} formats
            items = data if isinstance(data, list) else data.get("data", data.get("articles", []))

            parsed = []
            for item in items:
                title = (item.get("title") or item.get("name") or "").lower()
                url   = item.get("url") or item.get("link") or ""
                src   = item.get("source") or item.get("publisher") or "Unknown"
                tickers = item.get("tickers") or item.get("currencies") or item.get("symbols") or []
                currencies = ",".join(
                    t if isinstance(t, str) else t.get("symbol", "")
                    for t in tickers
                ).upper()

                # Sentiment from keywords
                b_hits = sum(1 for w in _BULLISH if w in title)
                bear_hits = sum(1 for w in _BEARISH if w in title)
                if b_hits > bear_hits:   sent = "positive"
                elif bear_hits > b_hits: sent = "negative"
                else:                    sent = "neutral"

                parsed.append({
                    "title":      item.get("title") or item.get("name") or "",
                    "url":        url,
                    "source":     src if isinstance(src, str) else src.get("name","Unknown"),
                    "sentiment":  sent,
                    "published":  item.get("published_at") or item.get("date") or "",
                    "currencies": currencies,
                })
            log.info("cryptocurrency.cv: %d articles loaded", len(parsed))
            return parsed
        except ValueError:
            log.warning("cryptocurrency.cv returned invalid JSON")
            return []
        except Exception as e:
            log.warning("cryptocurrency.cv error: %s", e)
            return []

    def _fetch_coingecko(self, limit: int) -> list[dict]:
        """CoinGecko news as secondary fallback — no key needed."""
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/news",
                timeout=10,
                headers={"User-Agent": "BTC-Strength-Bot/3.0"},
            )
            if resp.status_code != 200:
                return []
            items = resp.json().get("data", [])
            parsed = []
            for item in items[:limit]:
                title = (item.get("title") or "").lower()
                b_hits    = sum(1 for w in _BULLISH if w in title)
                bear_hits = sum(1 for w in _BEARISH if w in title)
                sent = "positive" if b_hits > bear_hits else ("negative" if bear_hits > b_hits else "neutral")
                parsed.append({
                    "title":      item.get("title",""),
                    "url":        item.get("url",""),
                    "source":     item.get("author","CoinGecko"),
                    "sentiment":  sent,
                    "published":  item.get("updated_at",""),
                    "currencies": "",
                })
            log.info("CoinGecko news fallback: %d articles", len(parsed))
            return parsed
        except Exception as e:
            log.warning("CoinGecko news error: %s", e)
            return []