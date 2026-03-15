"""
src/analysis/news_engine.py
────────────────────────────
Fetches crypto news from CryptoPanic for internal signal analysis.
News is NOT displayed on dashboard — used only to boost/reduce signal confidence.

CryptoPanic v1 API: https://cryptopanic.com/api/v1/posts/?auth_token=KEY
"""

import requests
from datetime import datetime, timezone
from src.utils.logger import get_logger

log = get_logger(__name__)

# CryptoPanic v1 API (works with auth token)
# CryptoPanic API — v1 requires auth_token query param
# Free tier: https://cryptopanic.com/developers/api/
_API_URL = "https://cryptopanic.com/api/v1/posts/"


class NewsEngine:

    def __init__(self, api_key: str = ""):
        self._api_key   = api_key.strip()
        self._cache:    list[dict] = []
        self._cached_at: datetime | None = None
        self._ttl_min   = 30

    def get_news(self, limit: int = 50) -> list[dict]:
        if self._is_valid():
            return self._cache[:limit]
        self._cache     = self._fetch()
        self._cached_at = datetime.now(timezone.utc)
        return self._cache[:limit]

    def get_sentiment_for(self, symbol: str) -> dict:
        """Return sentiment context for a coin symbol e.g. 'SOLUSDT' → 'SOL'."""
        coin = symbol.replace("USDT","").replace("BTC","").upper()
        news = self.get_news(100)
        relevant = [n for n in news if coin in n.get("currencies","").upper()
                    or coin.lower() in n.get("title","").lower()]

        if not relevant:
            return {"label": "neutral", "articles": 0, "score": 0}

        pos   = sum(1 for n in relevant if n["sentiment"] == "positive")
        neg   = sum(1 for n in relevant if n["sentiment"] == "negative")
        total = len(relevant)
        score = round((pos - neg) / total * 100) if total else 0

        label = "positive" if score >= 30 else ("negative" if score <= -30 else "neutral")
        return {"label": label, "score": score, "articles": total}

    # ── Internal ──────────────────────────────────────────────

    def _is_valid(self) -> bool:
        if not self._cached_at or not self._cache:
            return False
        return (datetime.now(timezone.utc) - self._cached_at).total_seconds() / 60 < self._ttl_min

    def _fetch(self) -> list[dict]:
        if not self._api_key:
            log.debug("No CRYPTOPANIC_API_KEY — news disabled")
            return []
        try:
            resp = requests.get(
                _API_URL,
                params={
                    "auth_token": self._api_key,
                    "public": "true",
                    "filter": "hot",
                    "kind": "news",
                    "regions": "en",
                },
                timeout=12,
            )
            if resp.status_code == 404:
                log.warning("CryptoPanic 404 — check API key at cryptopanic.com/developers/api/")
                return []
            if resp.status_code == 403:
                log.warning("CryptoPanic 403 — API key invalid or expired")
                return []
            if resp.status_code != 200:
                log.warning("CryptoPanic HTTP %d", resp.status_code)
                return []

            items  = resp.json().get("results", [])
            parsed = []
            for item in items:
                currencies = ",".join(c.get("code","") for c in (item.get("currencies") or []))
                votes      = item.get("votes") or {}
                pos  = votes.get("positive",0) + votes.get("liked",0)
                neg  = votes.get("negative",0) + votes.get("disliked",0)
                sent = "positive" if pos > neg * 1.5 else ("negative" if neg > pos * 1.5 else "neutral")
                parsed.append({
                    "title":      item.get("title",""),
                    "url":        item.get("url",""),
                    "source":     (item.get("source") or {}).get("title","Unknown"),
                    "sentiment":  sent,
                    "published":  item.get("published_at",""),
                    "currencies": currencies,
                })
            log.info("Fetched %d news items from CryptoPanic", len(parsed))
            return parsed
        except Exception as e:
            log.warning("News fetch error: %s", e)
            return []