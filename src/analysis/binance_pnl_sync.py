"""
src/analysis/binance_pnl_sync.py — v3
─────────────────────────────────────
Real Binance PnL synchronisation.

Fixes vs v2:
  • sync_recent() now logs EACH symbol updated (with PnL amount + event count)
    and EACH symbol skipped (with the reason), so you can diagnose exactly what
    happened without digging into the DB.
  • Skipped reasons are grouped and printed together at the end for readability.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone as _tz
from typing import Optional
from urllib.parse import urlencode

import requests

from src.utils.logger import get_logger

log = get_logger(__name__)

_FUT_BASE        = "https://fapi.binance.com"
_FUT_INCOME_PATH = "/fapi/v1/income"

# Binance docs: max window for /fapi/v1/income is 200 days.
_MAX_WINDOW_MS = 200 * 24 * 3600 * 1000

# In-memory cache of symbols that have failed THIS process lifecycle —
# avoids calling Binance again until the bot restarts.
_FAILED_SYMBOLS_CACHE: set[str] = set()


class BinancePnLSync:
    """
    Pull realised-PnL events from Binance and write them onto SignalRecords.

    Usage:
        sync = BinancePnLSync(api_key, api_secret)
        sync.sync_recent(days=7)
    """

    def __init__(self, api_key: str, api_secret: str):
        self._key    = api_key
        self._secret = api_secret

    # ── Binance helpers ──────────────────────────────────────────

    def _signed_get(self, path: str, params: dict) -> Optional[list]:
        if not self._key or not self._secret:
            log.debug("PnL sync: no API keys configured")
            return None

        params = dict(params)
        params["timestamp"]  = int(time.time() * 1000)
        params["recvWindow"] = 10000
        query = urlencode(params)
        sig   = hmac.new(self._secret.encode(),
                         query.encode(),
                         hashlib.sha256).hexdigest()
        url   = f"{_FUT_BASE}{path}?{query}&signature={sig}"
        try:
            r = requests.get(url, headers={"X-MBX-APIKEY": self._key}, timeout=12)
            if r.status_code != 200:
                body = (r.text or "<empty>")[:300]
                log.warning("Binance income %s: HTTP %d symbol=%s body=%s",
                            path, r.status_code,
                            params.get("symbol", "?"), body)
                return None
            return r.json()
        except Exception as e:
            log.warning("Binance income fetch error (%s): %s",
                        params.get("symbol", "?"), e)
            return None

    def fetch_realized_pnl(self,
                            symbol: str,
                            start_ms: int,
                            end_ms: Optional[int] = None) -> Optional[list[dict]]:
        """
        Fetch REALIZED_PNL income events for a symbol in [start_ms, end_ms].
        Returns list[dict] on success (possibly empty), None on API error.
        """
        try:
            symbol.encode("ascii")
        except (UnicodeEncodeError, AttributeError):
            log.debug("PnL sync: skipping non-ASCII symbol %r", symbol)
            return []

        if not start_ms or start_ms <= 0:
            return []
        now_ms = int(time.time() * 1000)
        if start_ms > now_ms:
            log.debug("PnL sync %s: start_ms in future, skipping", symbol)
            return []
        if end_ms is not None and end_ms < start_ms:
            log.debug("PnL sync %s: end < start, skipping", symbol)
            return []
        if end_ms is not None and (end_ms - start_ms) > _MAX_WINDOW_MS:
            log.debug("PnL sync %s: window > 200d, clamping", symbol)
            end_ms = start_ms + _MAX_WINDOW_MS
        if end_ms is not None and end_ms > now_ms:
            end_ms = now_ms

        params = {
            "symbol":     symbol,
            "incomeType": "REALIZED_PNL",
            "startTime":  start_ms,
            "limit":      1000,
        }
        if end_ms:
            params["endTime"] = end_ms

        data = self._signed_get(_FUT_INCOME_PATH, params)
        if data is None:
            return None
        if not isinstance(data, list):
            return []

        out = []
        for ev in data:
            try:
                out.append({
                    "income": float(ev.get("income", 0)),
                    "time":   int(ev.get("time", 0)),
                    "info":   ev.get("info", ""),
                })
            except (TypeError, ValueError):
                continue
        return out

    # ── Per-signal sync ──────────────────────────────────────────

    @staticmethod
    def _load_breakdown(sig_record) -> dict:
        try:
            existing = sig_record.score_breakdown
            if not existing:
                return {}
            return json.loads(existing) if isinstance(existing, str) else dict(existing)
        except Exception:
            return {}

    @staticmethod
    def _save_breakdown(sig_record, breakdown: dict) -> None:
        try:
            sig_record.score_breakdown = json.dumps(breakdown)
            sig_record.save(update_fields=["score_breakdown"])
        except Exception as e:
            log.debug("PnL sync save_breakdown error: %s", e)

    def sync_signal(self, sig_record) -> Optional[float]:
        """
        Sync one SignalRecord. Returns realised PnL USDT, 0.0 if no events,
        or None on API error (record is marked failed so we skip next time).
        """
        if sig_record.outcome == "PENDING":
            return None
        if not sig_record.entry_price or sig_record.entry_price <= 0:
            return None

        if sig_record.symbol in _FAILED_SYMBOLS_CACHE:
            return None

        try:
            start_ms = int(sig_record.created_at.timestamp() * 1000)
        except Exception:
            return None

        if sig_record.closed_at:
            try:
                end_ms = int(sig_record.closed_at.timestamp() * 1000) + 6 * 3600 * 1000
            except Exception:
                end_ms = start_ms + 7 * 24 * 3600 * 1000
        else:
            end_ms = start_ms + 7 * 24 * 3600 * 1000

        events = self.fetch_realized_pnl(sig_record.symbol, start_ms, end_ms)

        if events is None:
            _FAILED_SYMBOLS_CACHE.add(sig_record.symbol)
            breakdown = self._load_breakdown(sig_record)
            breakdown["binance_pnl_sync_failed"] = True
            breakdown["binance_pnl_failed_at"]   = datetime.now(_tz.utc).isoformat()
            self._save_breakdown(sig_record, breakdown)
            return None

        if not events:
            breakdown = self._load_breakdown(sig_record)
            breakdown["binance_realized_pnl_usdt"] = 0.0
            breakdown["binance_pnl_event_count"]  = 0
            breakdown["binance_pnl_synced_at"]    = datetime.now(_tz.utc).isoformat()
            self._save_breakdown(sig_record, breakdown)
            return 0.0

        total_usdt = sum(e["income"] for e in events)
        breakdown = self._load_breakdown(sig_record)
        breakdown["binance_realized_pnl_usdt"] = round(total_usdt, 4)
        breakdown["binance_pnl_synced_at"]    = datetime.now(_tz.utc).isoformat()
        breakdown["binance_pnl_event_count"]  = len(events)
        self._save_breakdown(sig_record, breakdown)
        return total_usdt

    # ── Batch sync ───────────────────────────────────────────────

    def sync_recent(self, days: int = 7) -> dict:
        """
        Bulk-fetch all REALIZED_PNL events for the window in ONE call,
        then bucket them onto matching SignalRecords.

        Logs every symbol updated (with PnL $ and event count) and every
        symbol skipped (with the reason), grouped by reason for readability.
        """
        from dashboard.models import SignalRecord
        from django.utils import timezone

        since = timezone.now() - timedelta(days=days)
        start_ms = int(since.timestamp() * 1000)
        end_ms   = int(time.time() * 1000)

        # ONE bulk call — no symbol filter, server returns all symbols
        events = self._signed_get(_FUT_INCOME_PATH, {
            "incomeType": "REALIZED_PNL",
            "startTime":  start_ms,
            "endTime":    end_ms,
            "limit":      1000,
        })
        if events is None:
            log.warning("Binance PnL bulk fetch failed — body in prior log line")
            return {"updated": 0, "skipped": 0, "errors": 1}
        if not isinstance(events, list):
            return {"updated": 0, "skipped": 0, "errors": 0}

        # Bucket events by symbol → list of (income_usdt, time_ms)
        by_symbol: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for ev in events:
            try:
                sym  = ev.get("symbol", "")
                inc  = float(ev.get("income", 0))
                t_ms = int(ev.get("time", 0))
                if sym:
                    by_symbol[sym].append((inc, t_ms))
            except (TypeError, ValueError):
                continue

        log.info("Binance PnL bulk: %d events across %d symbols",
                 len(events), len(by_symbol))

        # Match against SignalRecords
        qs = SignalRecord.objects.filter(
            created_at__gte=since,
        ).exclude(outcome="PENDING")

        updated   = 0
        skipped   = 0
        # Track what happened to each record for detailed logging
        updated_lines: list[str] = []          # e.g. "SOLUSDT TP2 → $1.2340 (3 evts)"
        skipped_by_reason: dict[str, list[str]] = defaultdict(list)  # reason → [symbols]

        for rec in qs:
            bd = self._load_breakdown(rec)

            # ── Already synced ──────────────────────────────────
            if "binance_realized_pnl_usdt" in bd:
                skipped += 1
                skipped_by_reason["already_synced"].append(
                    f"{rec.symbol}({rec.outcome})"
                )
                continue

            # ── Bad timestamp ───────────────────────────────────
            try:
                rec_start = int(rec.created_at.timestamp() * 1000)
                rec_end   = (
                    int(rec.closed_at.timestamp() * 1000) + 6 * 3600 * 1000
                    if rec.closed_at
                    else rec_start + 7 * 24 * 3600 * 1000
                )
            except Exception:
                skipped += 1
                skipped_by_reason["bad_timestamp"].append(
                    f"{rec.symbol}({rec.outcome})"
                )
                continue

            # ── Match events in window ──────────────────────────
            matched = [
                inc
                for inc, t in by_symbol.get(rec.symbol, [])
                if rec_start <= t <= rec_end
            ]

            pnl_val = round(sum(matched), 4)
            bd["binance_realized_pnl_usdt"] = pnl_val
            bd["binance_pnl_event_count"]   = len(matched)
            bd["binance_pnl_synced_at"]     = datetime.now(_tz.utc).isoformat()
            self._save_breakdown(rec, bd)
            updated += 1

            if matched:
                sign = "+" if pnl_val >= 0 else ""
                updated_lines.append(
                    f"{rec.symbol} {rec.outcome} → {sign}${pnl_val:.4f} ({len(matched)} evts)"
                )
                log.info("PnL sync: %s %s → $%.4f (%d events)",
                         rec.symbol, rec.outcome, pnl_val, len(matched))
            else:
                # Closed signal found but no matching Binance income event
                # (trade may have been manual, or events outside the window)
                updated_lines.append(
                    f"{rec.symbol} {rec.outcome} → $0.0000 (no Binance events in window)"
                )
                skipped_by_reason["no_binance_events_in_window"].append(
                    f"{rec.symbol}({rec.outcome})"
                )

        # ── Print updated symbols ───────────────────────────────
        if updated_lines:
            log.info("PnL sync — %d updated:\n  %s",
                     len(updated_lines), "\n  ".join(updated_lines))

        # ── Print skipped symbols grouped by reason ─────────────
        if skipped_by_reason:
            for reason, syms in skipped_by_reason.items():
                # Show up to 15 per reason to keep logs clean
                preview = syms[:15]
                tail    = f" … +{len(syms) - 15} more" if len(syms) > 15 else ""
                log.info("PnL sync skipped [%s] x%d: %s%s",
                         reason, len(syms), ", ".join(preview), tail)

        log.info("Binance PnL bulk sync done: %d updated, %d skipped",
                 updated, skipped)
        return {"updated": updated, "skipped": skipped, "errors": 0}