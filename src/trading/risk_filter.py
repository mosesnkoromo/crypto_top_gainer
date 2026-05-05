"""
src/trading/risk_filter.py
───────────────────────────
Pre-trade risk filter — sits between SignalEngine and BinanceTrader.

Returns (allowed: bool, reason: str). Designed to be additive: if the
filter is removed, current behaviour is unchanged. Failures here NEVER
raise — they always return a decision.

Layers:
  1. Pair blacklist     — symbols that have only ever lost money
  2. Liquidity gate     — reject sub-$10M 24hr volume pairs (gap risk)
  3. Correlation guard  — max 2 open positions per correlated group
  4. STOP_MARKET hint   — flag low-liquidity pairs so trader uses
                          STOP_MARKET-only (no LIMIT fallback)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from src.utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────
# LAYER 1 — Pair blacklist
# ─────────────────────────────────────────────────────────────────
# Hard blacklist: 0% historical win rate AND ≥3% net loss.
# Re-evaluate every 30 days. Reset in the dashboard if a pair recovers.
HARD_BLACKLIST: set[str] = {
    "GIGGLEUSDT",  # 1 trade, -18.22%, microcap meme
    "BANANAS31USDT",  # 1 trade, -16.59%, microcap meme
    "FILUSDT",  # 1 trade, -11.00%
    "CHZUSDT",  # 1 trade, -10.98%
    "SEIUSDT",  # 1 trade, -9.29%
    "HBARUSDT",  # 1 trade, -4.63%
    "AAVEUSDT",  # 0/8, -$5.40 — structural failure
    "WLFIUSDT",  # 0/1, -$5.45 — microcap pattern
    "ZAMAUSDT",  # 1 trade, -4.63%
    "ASTERUSDT",  # 1 trade, -4.63%
}

# Watch blacklist: mixed record, 50% WR but net negative.
# These pairs trade only at STANDARD grade size (or skip entirely).
WATCH_BLACKLIST: set[str] = {
    "ZECUSDT",  # 0/2, -10.98%
    "ENAUSDT",  # 2/2, -11.70%
    "TAOUSDT",
    "MASKUSDT",  # 2 trade 0% WR, -$1.73
    "METAUSDT",  # 2 trade 0% WR, -$1.73
    "NFPUSDT",  # 2 trade 0% WR, -$1.73
    "OPENUSDT",  # 2 trade 0% WR, -$1.73
    "SUIUSDT",  # 2 trade 0% WR, -$1.73
}

# ─────────────────────────────────────────────────────────────────
# LAYER 2 — Liquidity gate
# ─────────────────────────────────────────────────────────────────
MIN_QUOTE_VOLUME_24H_USD = 10_000_000  # below this = reject

# Sub-$50M = use STOP_MARKET only (no LIMIT fallback). LIMIT stops
# don't fire on illiquid books — the position bleeds through them.
STOP_MARKET_ONLY_THRESHOLD_USD = 50_000_000

# ─────────────────────────────────────────────────────────────────
# LAYER 3 — Correlation guard
# ─────────────────────────────────────────────────────────────────
# Pairs that historically dump together. Cap concurrent exposure.
CORRELATION_GROUPS: list[set[str]] = [
    {"BTCUSDT", "ETHUSDT"},
    {"XRPUSDT", "XLMUSDT"},
    {"LINKUSDT", "DOTUSDT", "ATOMUSDT"},
    {"SOLUSDT", "AVAXUSDT", "NEARUSDT", "SEIUSDT"},
    {"DOGEUSDT", "PEPEUSDT", "WIFUSDT", "1000PEPEUSDT", "FLOKIUSDT", "1000FLOKIUSDT"},
    {"ARBUSDT", "OPUSDT"},
    {"INJUSDT", "TIAUSDT"},
]

MAX_CORRELATED_OPEN = 2


# ─────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────
@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""
    layer: str = ""
    use_stop_market_only: bool = False  # Layer 5 / P1 hint
    quote_volume_usd: float = 0.0


# ─────────────────────────────────────────────────────────────────
# Volume cache — avoid re-fetching ticker on every signal
# ─────────────────────────────────────────────────────────────────
class _VolumeCache:
    def __init__(self, ttl_sec: int = 300):
        self._ttl = ttl_sec
        self._data: dict[str, tuple[float, float]] = {}  # sym -> (vol_usd, ts)
        self._all_loaded_at: float = 0.0
        self._all_data: dict[str, float] = {}  # bulk snapshot

    def get(self, symbol: str) -> Optional[float]:
        now = time.time()
        # Try bulk cache first
        if (now - self._all_loaded_at) < self._ttl and symbol in self._all_data:
            return self._all_data[symbol]
        # Try per-symbol cache
        item = self._data.get(symbol)
        if item and (now - item[1]) < self._ttl:
            return item[0]
        return None

    def set_all(self, data: dict[str, float]) -> None:
        self._all_data = dict(data)
        self._all_loaded_at = time.time()

    def set_one(self, symbol: str, vol: float) -> None:
        self._data[symbol] = (vol, time.time())


# ─────────────────────────────────────────────────────────────────
# RiskFilter
# ─────────────────────────────────────────────────────────────────
class RiskFilter:
    """
    Stateless filter; safe to instantiate once and reuse.
    Network calls are cached (volume snapshot lives 5 min).
    """

    def __init__(self,
                 min_volume_usd: float = MIN_QUOTE_VOLUME_24H_USD,
                 stop_market_only_threshold_usd: float = STOP_MARKET_ONLY_THRESHOLD_USD,
                 max_correlated_open: int = MAX_CORRELATED_OPEN):
        self._min_vol = min_volume_usd
        self._stop_only_v = stop_market_only_threshold_usd
        self._max_corr = max_correlated_open
        self._vol_cache = _VolumeCache(ttl_sec=300)

    # ── Layer 1 ────────────────────────────────────────────────

    def is_blacklisted(self, symbol: str) -> tuple[bool, str]:
        sym = symbol.upper()
        if sym in HARD_BLACKLIST:
            return True, f"L1 hard-blacklist: {sym} (0% WR history)"
        if sym in WATCH_BLACKLIST:
            return True, f"L1 watch-blacklist: {sym} (mixed record, paused)"
        return False, ""

    # ── Layer 2 ────────────────────────────────────────────────

    def get_quote_volume(self, symbol: str) -> Optional[float]:
        """24hr quote volume in USDT. Cached. Returns None on lookup failure."""
        cached = self._vol_cache.get(symbol)
        if cached is not None:
            return cached

        # Try bulk fetch first — one HTTP call for all symbols
        try:
            r = requests.get(
                "https://fapi.binance.com/fapi/v1/ticker/24hr",
                timeout=6,
            )
            if r.ok:
                data = r.json()
                snap = {}
                for t in data:
                    sym = t.get("symbol", "")
                    qv = t.get("quoteVolume")
                    if sym and qv is not None:
                        try:
                            snap[sym] = float(qv)
                        except (TypeError, ValueError):
                            pass
                if snap:
                    self._vol_cache.set_all(snap)
                    return snap.get(symbol)
        except Exception as e:
            log.debug("Bulk ticker fetch failed: %s", e)

        # Fallback: single-symbol lookup (rate-limited path)
        try:
            r = requests.get(
                "https://fapi.binance.com/fapi/v1/ticker/24hr",
                params={"symbol": symbol},
                timeout=6,
            )
            if r.ok:
                vol = float(r.json().get("quoteVolume", 0))
                self._vol_cache.set_one(symbol, vol)
                return vol
        except Exception as e:
            log.debug("Single ticker fetch %s failed: %s", symbol, e)
        return None

    def liquidity_check(self, symbol: str) -> tuple[bool, float, bool]:
        """
        Returns (passes_min, vol_usd, requires_stop_market_only).
        If volume is unknown (fetch failed), we pass rather than block —
        avoids breaking the bot when Binance ticker endpoint is flaky.
        """
        vol = self.get_quote_volume(symbol)
        if vol is None:
            return True, 0.0, False  # fail-open
        passes_min = vol >= self._min_vol
        require_sm = vol < self._stop_only_v
        return passes_min, vol, require_sm

    # ── Layer 3 ────────────────────────────────────────────────

    def correlation_check(self, symbol: str, open_symbols: list[str]
                          ) -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        `open_symbols` = currently-open futures positions on Binance.
        """
        sym = symbol.upper()
        open_set = {s.upper() for s in open_symbols}

        for group in CORRELATION_GROUPS:
            if sym not in group:
                continue
            in_group = open_set & group
            # Don't double-count if the symbol is already open (re-entry case)
            in_group_excl_self = in_group - {sym}
            if len(in_group_excl_self) >= self._max_corr:
                others = ",".join(sorted(in_group_excl_self))
                return False, f"L3 correlation: {self._max_corr} already open in group ({others})"
        return True, ""

    # ── Master gate ────────────────────────────────────────────

    def evaluate(self, symbol: str, open_symbols: list[str]) -> RiskDecision:
        """
        Single entry point. Calls all layers in order; first failure short-circuits.
        Sets `use_stop_market_only=True` when liquidity is between min and STOP_ONLY threshold.
        """
        sym = (symbol or "").upper()

        # Layer 1
        bl, reason = self.is_blacklisted(sym)
        if bl:
            return RiskDecision(False, reason, "L1")

        # Layer 2
        ok, vol, require_sm = self.liquidity_check(sym)
        if not ok:
            return RiskDecision(
                False,
                f"L2 liquidity: ${vol / 1e6:.1f}M < ${self._min_vol / 1e6:.0f}M minimum",
                "L2",
                quote_volume_usd=vol,
            )

        # Layer 3
        ok, reason = self.correlation_check(sym, open_symbols)
        if not ok:
            return RiskDecision(False, reason, "L3", quote_volume_usd=vol)

        # All passed
        return RiskDecision(
            True, "", "",
            use_stop_market_only=require_sm,
            quote_volume_usd=vol,
        )
