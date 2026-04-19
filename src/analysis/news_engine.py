"""
src/analysis/news_engine.py
────────────────────────────
Production-grade news engine with multiple providers, circuit breakers,
persistent cache (diskcache / JSON fallback), rate limiting, and health monitoring.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.utils.logger import get_logger

load_dotenv()
log = get_logger(__name__)

# ------------------------------------------------------------------
# Configuration from environment
# ------------------------------------------------------------------
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY", "")

CACHE_TTL_MIN = int(os.getenv("NEWS_CACHE_TTL_MIN", "30"))
CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "3"))
CIRCUIT_BREAKER_TIMEOUT_SEC = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "300"))  # 5 min
RATE_LIMIT_SLEEP_SEC = float(os.getenv("RATE_LIMIT_SLEEP", "1.0"))

# ------------------------------------------------------------------
# Keyword-based sentiment (fallback when provider doesn't give sentiment)
# ------------------------------------------------------------------
_BEARISH = {
    "hack", "exploit", "crash", "ban", "sued", "scam", "fraud", "collapse",
    "plunge", "dump", "bear", "sell-off", "liquidat", "bankrupt", "rug", "exit scam"
}
_BULLISH = {
    "rally", "surge", "bull", "breakout", "adoption", "partnership", "launch",
    "upgrade", "etf", "approval", "institutional", "record", "high", "milestone"
}

# ------------------------------------------------------------------
# Persistent cache (diskcache preferred, fallback to JSON file)
# ------------------------------------------------------------------
class PersistentCache:
    """Disk-based cache using `diskcache` if available, otherwise JSON file."""
    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self._cache = None
        self._cache_dir = Path(".cache")
        self._cache_dir.mkdir(exist_ok=True)

        # Try to use diskcache (recommended)
        try:
            import diskcache
            self._cache = diskcache.Cache(str(self._cache_dir / "diskcache"))
            log.info("Using diskcache (embedded, high performance)")
        except ImportError:
            log.warning("diskcache not installed, falling back to JSON file cache. Install with: pip install diskcache")
            self._cache = None
            self._json_file = self._cache_dir / "news_cache.json"

    def get(self, key: str) -> Optional[Any]:
        if self._cache is not None:
            # diskcache handles TTL automatically if we set expire on set
            value = self._cache.get(key)
            return value

        # JSON fallback
        if not self._json_file.exists():
            return None
        try:
            with open(self._json_file, "r") as f:
                cache = json.load(f)
            entry = cache.get(key)
            if entry and entry["expires"] > time.time():
                return entry["data"]
            return None
        except Exception:
            return None

    def set(self, key: str, value: Any):
        if self._cache is not None:
            self._cache.set(key, value, expire=self.ttl)
            return

        # JSON fallback
        try:
            if self._json_file.exists():
                with open(self._json_file, "r") as f:
                    cache = json.load(f)
            else:
                cache = {}
            cache[key] = {
                "expires": time.time() + self.ttl,
                "data": value
            }
            with open(self._json_file, "w") as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            log.error(f"Failed to write JSON cache: {e}")

    @property
    def backend_name(self) -> str:
        return "diskcache" if self._cache is not None else "json_file"

# ------------------------------------------------------------------
# Circuit Breaker (unchanged)
# ------------------------------------------------------------------
class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int, timeout_sec: int):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout_sec = timeout_sec
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout_sec:
                self.state = "HALF_OPEN"
                log.info(f"Circuit breaker '{self.name}' -> HALF_OPEN")
            else:
                log.warning(f"Circuit breaker '{self.name}' is OPEN, skipping call")
                raise Exception(f"CircuitBreakerOpen: {self.name}")

        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.reset()
                log.info(f"Circuit breaker '{self.name}' -> CLOSED (success in half-open)")
            return result
        except Exception as e:
            self._record_failure()
            raise e

    def _record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.state == "CLOSED" and self.failures >= self.failure_threshold:
            self.state = "OPEN"
            log.error(f"Circuit breaker '{self.name}' -> OPEN after {self.failures} failures")

    def reset(self):
        self.failures = 0
        self.state = "CLOSED"
        self.last_failure_time = 0.0

    def status(self) -> dict:
        return {
            "state": self.state,
            "failures": self.failures,
            "last_failure": self.last_failure_time,
            "cooldown_remaining": max(0, self.timeout_sec - (time.time() - self.last_failure_time)) if self.state == "OPEN" else 0
        }

# ------------------------------------------------------------------
# Provider implementations (GNews, CryptoCompare) – unchanged
# ------------------------------------------------------------------
class GNewsProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "gnews"
        self.base_url = "https://gnews.io/api/v4/search"
        self.last_request_time = 0

    def fetch(self, limit: int) -> List[dict]:
        if not self.api_key:
            raise ValueError("GNEWS_API_KEY missing")

        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_SLEEP_SEC:
            time.sleep(RATE_LIMIT_SLEEP_SEC - elapsed)
        self.last_request_time = time.time()

        params = {
            "q": "cryptocurrency OR bitcoin OR ethereum OR crypto",
            "token": self.api_key,
            "lang": "en",
            "max": min(limit, 50),
            "sortby": "publishedAt"
        }
        resp = requests.get(self.base_url, params=params, timeout=12)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            raise Exception(f"Rate limited by GNews, retry after {retry_after}s")
        if resp.status_code != 200:
            raise Exception(f"GNews HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        articles = data.get("articles", [])
        parsed = []
        for art in articles:
            title = art.get("title", "")
            if not title:
                continue
            sentiment = self._sentiment_from_title(title)
            parsed.append({
                "title": title,
                "url": art.get("url", ""),
                "source": art.get("source", {}).get("name", "GNews"),
                "sentiment": sentiment,
                "published": art.get("publishedAt", ""),
                "currencies": self._extract_coins(title)
            })
        return parsed

    @staticmethod
    def _sentiment_from_title(title: str) -> str:
        lower = title.lower()
        bullish = sum(1 for w in _BULLISH if w in lower)
        bearish = sum(1 for w in _BEARISH if w in lower)
        if bullish > bearish:
            return "positive"
        if bearish > bullish:
            return "negative"
        return "neutral"

    @staticmethod
    def _extract_coins(title: str) -> str:
        common = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "DOT", "LINK", "MATIC"]
        found = [c for c in common if c.lower() in title.lower()]
        return ",".join(found)


class CryptoCompareProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "cryptocompare"
        self.base_url = "https://min-api.cryptocompare.com/data/v2/news/"
        self.last_request_time = 0

    def fetch(self, limit: int) -> List[dict]:
        if not self.api_key:
            raise ValueError("CRYPTOCOMPARE_API_KEY missing")

        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_SLEEP_SEC:
            time.sleep(RATE_LIMIT_SLEEP_SEC - elapsed)
        self.last_request_time = time.time()

        params = {
            "api_key": self.api_key,
            "lang": "EN",
            "limit": min(limit, 50),
            "feeds": "cointelegraph,coindesk,decrypt,newsbtc,beincrypto"
        }
        resp = requests.get(self.base_url, params=params, timeout=12)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            raise Exception(f"Rate limited by CryptoCompare, retry after {retry_after}s")
        if resp.status_code != 200:
            raise Exception(f"CryptoCompare HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        articles = data.get("Data", [])
        parsed = []
        for art in articles:
            title = art.get("title", "")
            if not title:
                continue
            sentiment = art.get("sentiment", "neutral")
            if sentiment not in ("positive", "neutral", "negative"):
                sentiment = "neutral"
            parsed.append({
                "title": title,
                "url": art.get("url", ""),
                "source": art.get("source", "CryptoCompare"),
                "sentiment": sentiment,
                "published": datetime.fromtimestamp(art.get("published_on", 0)).isoformat(),
                "currencies": art.get("categories", "").upper()
            })
        return parsed


# ------------------------------------------------------------------
# Main NewsEngine class
# ------------------------------------------------------------------
class NewsEngine:
    def __init__(self, api_key: str = ""):
        """
        api_key parameter kept for backward compatibility, but real keys
        are read from environment variables.
        """
        self._cache = PersistentCache(ttl_seconds=CACHE_TTL_MIN * 60)
        self._cache_key = "news_articles"

        # Initialize providers only if API keys exist
        self._providers = []
        if GNEWS_API_KEY:
            self._providers.append(GNewsProvider(GNEWS_API_KEY))
        if CRYPTOCOMPARE_API_KEY:
            self._providers.append(CryptoCompareProvider(CRYPTOCOMPARE_API_KEY))

        if not self._providers:
            log.error("No news API keys configured. Please set GNEWS_API_KEY or CRYPTOCOMPARE_API_KEY in .env")

        # Circuit breakers per provider
        self._breakers = {}
        for p in self._providers:
            self._breakers[p.name] = CircuitBreaker(
                p.name,
                CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                CIRCUIT_BREAKER_TIMEOUT_SEC
            )

    def get_news(self, limit: int = 50) -> List[dict]:
        """Return cached news articles. Fetches fresh data if cache is stale."""
        cached = self._cache.get(self._cache_key)
        if cached is not None:
            log.debug(f"Cache hit: {len(cached)} articles")
            return cached[:limit]

        articles = self._fetch_from_providers(limit)
        if articles:
            self._cache.set(self._cache_key, articles)
            log.info(f"News cache refreshed: {len(articles)} articles")
        else:
            log.warning("No news fetched from any provider; cache remains empty")
        return articles[:limit] if articles else []

    def get_sentiment_for(self, symbol: str) -> dict:
        """Calculate sentiment for a specific coin from cached news."""
        news = self._cache.get(self._cache_key)
        if not news:
            return {"label": "neutral", "articles": 0, "score": 0}

        coin = symbol.replace("USDT", "").upper()
        relevant = []
        for article in news:
            currencies = article.get("currencies", "").upper()
            title = article.get("title", "").upper()
            if coin in currencies or coin in title:
                relevant.append(article)

        if not relevant:
            return {"label": "neutral", "articles": 0, "score": 0}

        positive = sum(1 for a in relevant if a["sentiment"] == "positive")
        negative = sum(1 for a in relevant if a["sentiment"] == "negative")
        total = len(relevant)

        score = round((positive - negative) / total * 100) if total else 0
        if score >= 25:
            label = "positive"
        elif score <= -25:
            label = "negative"
        else:
            label = "neutral"

        return {"label": label, "score": score, "articles": total}

    def health(self) -> dict:
        """Return health status of all providers and cache."""
        status = {
            "cache": {
                "type": self._cache.backend_name,
                "has_data": self._cache.get(self._cache_key) is not None,
                "ttl_minutes": CACHE_TTL_MIN
            },
            "providers": {}
        }
        for provider in self._providers:
            cb = self._breakers[provider.name]
            status["providers"][provider.name] = {
                "api_key_configured": True,
                "circuit_breaker": cb.status()
            }
        # Add missing providers as not configured
        if "gnews" not in status["providers"]:
            status["providers"]["gnews"] = {"api_key_configured": False}
        if "cryptocompare" not in status["providers"]:
            status["providers"]["cryptocompare"] = {"api_key_configured": False}
        return status

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    def _fetch_from_providers(self, limit: int) -> List[dict]:
        """Try each provider in order, return first successful result."""
        for provider in self._providers:
            cb = self._breakers[provider.name]
            try:
                articles = cb.call(provider.fetch, limit)
                if articles:
                    log.info(f"Successfully fetched {len(articles)} articles from {provider.name}")
                    return articles
            except Exception as e:
                log.warning(f"Provider {provider.name} failed: {e}")
                continue
        log.error("All news providers failed")
        return []