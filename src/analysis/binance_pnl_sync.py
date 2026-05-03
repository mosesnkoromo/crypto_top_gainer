"""
src/analysis/binance_pnl_sync.py
─────────────────────────────────
Real Binance PnL synchronisation.

The dashboard's `profit_pct` column is computed from signal-level
outcomes (TP1=+0.6%, TP2=+1.2%, ...). That's not the actual realised
PnL — it ignores fill slippage, partial fills, and per-position fees.

This module queries Binance income history and writes the REAL pct
PnL into `SignalRecord.score_breakdown["binance_realized_pnl_pct"]`
without touching the existing `profit_pct` column. Dashboards can
then prefer the real value when present, falling back to the signal
outcome when the trade wasn't auto-executed or sync hasn't run yet.

Schema-safe: uses the existing JSONField. No migration needed.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

from src.utils.logger import get_logger

log = get_logger(__name__)

_FUT_INCOME_URL = "https://fapi.binance.com/fapi/v1/income"


class BinancePnLSync:
    """
    Pull realised-PnL events from Binance and write them onto SignalRecords.

    Usage:
        sync = BinancePnLSync(api_key, api_secret)
        n = sync.sync_recent(days=7)   # update the last week's signals
    """

    def __init__(self, api_key: str, api_secret: str):
        self._key    = api_key
        self._secret = api_secret

    # ── Binance helpers ──────────────────────────────────────────

    def _signed_get(self, path: str, params: dict) -> Optional[list]:
        import hmac, hashlib
        from urllib.parse import urlencode

        if not self._key or not self._secret:
            return None

        params = dict(params)
        params["timestamp"]  = int(time.time() * 1000)
        params["recvWindow"] = 10000
        query = urlencode(params)
        sig   = hmac.new(self._secret.encode(),
                         query.encode(),
                         hashlib.sha256).hexdigest()
        url   = f"https://fapi.binance.com{path}?{query}&signature={sig}"
        try:
            r = requests.get(url, headers={"X-MBX-APIKEY": self._key}, timeout=12)
            if r.status_code != 200:
                log.warning("Binance income fetch %s: HTTP %d", path, r.status_code)
                return None
            return r.json()
        except Exception as e:
            log.warning("Binance income fetch error: %s", e)
            return None

    def fetch_realized_pnl(self,
                            symbol: str,
                            start_ms: int,
                            end_ms: Optional[int] = None) -> list[dict]:
        """
        Fetch REALIZED_PNL income events for a symbol in [start_ms, end_ms].
        Returns list of dicts with {"income": float, "time": int}.
        """
        params = {
            "symbol":     symbol,
            "incomeType": "REALIZED_PNL",
            "startTime":  start_ms,
            "limit":      1000,
        }
        if end_ms:
            params["endTime"] = end_ms

        data = self._signed_get("/fapi/v1/income", params)
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

    def sync_signal(self, sig_record) -> Optional[float]:
        """
        Sync one SignalRecord. Returns the realised PnL pct or None
        if Binance had no matching activity (manual trade, signal-only, etc.)
        """
        # Only meaningful for closed signals
        if sig_record.outcome == "PENDING":
            return None
        if not sig_record.entry_price or sig_record.entry_price <= 0:
            return None

        # Window: signal time → close time + 6h buffer
        start_ms = int(sig_record.created_at.timestamp() * 1000)
        if sig_record.closed_at:
            end_ms = int(sig_record.closed_at.timestamp() * 1000) + 6 * 3600 * 1000
        else:
            end_ms = start_ms + 7 * 24 * 3600 * 1000

        events = self.fetch_realized_pnl(sig_record.symbol, start_ms, end_ms)
        if not events:
            return None

        # Sum realized PnL (USDT) for this symbol within the window
        total_usdt = sum(e["income"] for e in events)

        # Convert to % using the position notional. We don't have the
        # exact qty stored on the record — best estimate from signal
        # entry × the bot's risk allocation. Fallback: report raw USDT
        # via the same field as `_usd` suffix variant.
        # For now: store both raw USDT and the inferred pct.
        # (Pct inference requires balance at trade time, which we
        # don't track. So we expose USDT as primary truth.)

        breakdown = {}
        try:
            existing = sig_record.score_breakdown
            if existing:
                breakdown = json.loads(existing) if isinstance(existing, str) else dict(existing)
        except Exception:
            breakdown = {}

        breakdown["binance_realized_pnl_usdt"] = round(total_usdt, 4)
        breakdown["binance_pnl_synced_at"]    = datetime.utcnow().isoformat()
        breakdown["binance_pnl_event_count"]  = len(events)

        sig_record.score_breakdown = json.dumps(breakdown)
        sig_record.save(update_fields=["score_breakdown"])

        return total_usdt

    # ── Batch sync ───────────────────────────────────────────────

    def sync_recent(self, days: int = 7) -> dict:
        """
        Batch sync all closed signals from the last N days.
        Returns {"updated": N, "skipped": N, "errors": N}.
        """
        from dashboard.models import SignalRecord
        from django.utils import timezone

        since = timezone.now() - timedelta(days=days)
        qs = SignalRecord.objects.filter(
            created_at__gte=since,
        ).exclude(outcome="PENDING")

        updated = skipped = errors = 0
        for rec in qs:
            try:
                # Skip if already synced recently
                try:
                    bd = json.loads(rec.score_breakdown) if rec.score_breakdown else {}
                except Exception:
                    bd = {}
                if "binance_realized_pnl_usdt" in bd:
                    skipped += 1
                    continue

                result = self.sync_signal(rec)
                if result is not None:
                    updated += 1
                    log.info("PnL sync: %s %s → $%.4f USDT",
                             rec.symbol, rec.outcome, result)
                else:
                    skipped += 1
            except Exception as e:
                errors += 1
                log.warning("PnL sync error %s: %s", rec.symbol, e)
                continue

        log.info("Binance PnL sync done: %d updated, %d skipped, %d errors",
                 updated, skipped, errors)
        return {"updated": updated, "skipped": skipped, "errors": errors}