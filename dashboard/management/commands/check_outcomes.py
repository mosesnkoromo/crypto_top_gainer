"""
dashboard/management/commands/check_outcomes.py  — v2 SCALP
─────────────────────────────────────────────────────────────
Determines real trade outcomes by querying Binance order fill history.
Uses 5m candles (not 1H) and compares against actual fill prices.

Priority order:
  1. Query Binance /fapi/v1/userTrades for actual fill prices
  2. Fall back to 5m candle simulation if no API keys
  3. Use actual entry price (not signal entry) for P&L calc

Run: python manage.py check_outcomes
Or auto-runs every scan cycle in scanner.py
"""
from __future__ import annotations

import os
import hmac
import hashlib
import time
import requests
import pandas as pd
from datetime import timedelta
from urllib.parse import urlencode
from django.core.management.base import BaseCommand
from django.utils import timezone
from dashboard.models import SignalRecord
from src.utils.logger import get_logger

log = get_logger(__name__)

FAPI_BASE = "https://fapi.binance.com"
SAPI_BASE = "https://api.binance.com"


def _signed_req(base: str, path: str, params: dict,
                api_key: str, api_secret: str) -> dict | list | None:
    """Make a signed Binance request."""
    p = {k: v for k, v in params.items() if v is not None}
    p["timestamp"]  = int(time.time() * 1000)
    p["recvWindow"] = 10000
    query = urlencode(p)
    sig   = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    url   = f"{base}{path}?{query}&signature={sig}"
    try:
        resp = requests.get(url, headers={"X-MBX-APIKEY": api_key}, timeout=12)
        if resp.ok:
            return resp.json()
    except Exception as e:
        log.debug("Signed req error: %s", e)
    return None


def get_candles_5m(symbol: str, since_ms: int, limit: int = 60) -> pd.DataFrame:
    """Fetch 5m candles from signal creation time (public endpoint)."""
    try:
        # Try futures first, fall back to spot
        for base, path in [
            ("https://fapi.binance.com", "/fapi/v1/klines"),
            ("https://api.binance.com",  "/api/v3/klines"),
        ]:
            resp = requests.get(f"{base}{path}", params={
                "symbol": symbol, "interval": "5m",
                "startTime": since_ms, "limit": limit,
            }, timeout=12)
            if resp.ok and resp.json():
                df = pd.DataFrame(resp.json(), columns=[
                    "open_time","open","high","low","close","volume",
                    "close_time","quote_vol","trades","buy_base","buy_quote","ignore"
                ])
                for col in ["open","high","low","close"]:
                    df[col] = pd.to_numeric(df[col])
                df["open_time"] = pd.to_numeric(df["open_time"])
                return df
    except Exception as e:
        log.debug("5m candles error %s: %s", symbol, e)
    return pd.DataFrame()


def get_actual_trades(symbol: str, since_ms: int,
                      api_key: str, api_secret: str) -> list:
    """Get actual trade fills from Binance futures history."""
    if not api_key or not api_secret:
        return []
    result = _signed_req(FAPI_BASE, "/fapi/v1/userTrades", {
        "symbol": symbol, "startTime": since_ms, "limit": 50,
    }, api_key, api_secret)
    return result if isinstance(result, list) else []


def determine_outcome_from_trades(signal: SignalRecord,
                                   trades: list) -> tuple[str, float, float]:
    """
    Use actual Binance trade fills to determine outcome.
    Returns (outcome_code, close_price, actual_pnl_pct).
    """
    if not trades:
        return "PENDING", 0.0, 0.0

    is_buy = signal.signal == "BUY"

    # Find entry fill: first trade in the right direction
    entry_side = "BUY" if is_buy else "SELL"
    close_side = "SELL" if is_buy else "BUY"

    entry_fills = [t for t in trades if t.get("side") == entry_side]
    close_fills = [t for t in trades if t.get("side") == close_side]

    if not entry_fills:
        return "PENDING", 0.0, 0.0

    # Calculate actual average entry price
    total_qty = sum(float(t.get("qty", 0)) for t in entry_fills)
    if total_qty == 0:
        return "PENDING", 0.0, 0.0
    actual_entry = sum(float(t.get("price", 0)) * float(t.get("qty", 0))
                       for t in entry_fills) / total_qty

    if not close_fills:
        return "PENDING", 0.0, 0.0

    # Most recent close fill
    close_fills_sorted = sorted(close_fills, key=lambda t: int(t.get("time", 0)))
    close_price = float(close_fills_sorted[-1].get("price", 0))

    if close_price == 0:
        return "PENDING", 0.0, 0.0

    # Determine outcome from actual prices vs signal targets
    if is_buy:
        pnl_pct = (close_price - actual_entry) / actual_entry * 100
        if close_price >= signal.tp3:   outcome = "TP3"
        elif close_price >= signal.tp2: outcome = "TP2"
        elif close_price >= signal.tp1: outcome = "TP1"
        elif pnl_pct > 0.1:             outcome = "TP1"  # profitable but below TP1 signal level
        else:                            outcome = "SL"
    else:
        pnl_pct = (actual_entry - close_price) / actual_entry * 100
        if close_price <= signal.tp3:   outcome = "TP3"
        elif close_price <= signal.tp2: outcome = "TP2"
        elif close_price <= signal.tp1: outcome = "TP1"
        elif pnl_pct > 0.1:             outcome = "TP1"  # profitable but below signal TP1
        else:                            outcome = "SL"

    return outcome, close_price, round(pnl_pct, 2)


def determine_outcome_from_candles(signal: SignalRecord,
                                    df: pd.DataFrame) -> tuple[str, float]:
    """
    Fallback: walk 5m candles forward and check which level was hit FIRST.
    Uses minute-level open_time to simulate sequence correctly.
    """
    if df.empty:
        return "PENDING", 0.0

    is_sell = signal.signal == "SELL"
    tp1, tp2, tp3, sl = signal.tp1, signal.tp2, signal.tp3, signal.sl

    for _, row in df.iterrows():
        high = float(row["high"])
        low  = float(row["low"])
        open_p = float(row["open"])

        if is_sell:
            # For SELL (short): price falls to TP, rises to SL
            # Use open price direction to determine order within candle
            if open_p >= high * 0.998:  # opened near high → likely fell first
                if low  <= tp1: return "TP1", tp1
                if high >= sl:  return "SL",  sl
            else:              # opened near low → likely rose first
                if high >= sl:  return "SL",  sl
                if low  <= tp1: return "TP1", tp1
        else:
            # For BUY (long): price rises to TP, falls to SL
            if open_p <= low * 1.002:   # opened near low → likely rose first
                if high >= tp1: return "TP1", tp1
                if low  <= sl:  return "SL",  sl
            else:              # opened near high → likely fell first
                if low  <= sl:  return "SL",  sl
                if high >= tp1: return "TP1", tp1

    return "PENDING", 0.0


class Command(BaseCommand):
    help = "Auto-updates signal outcomes from Binance trade history (5m scalp aware)"

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true",
                            help="Print outcomes without saving")
        parser.add_argument("--hours", type=int, default=4,
                            help="Look back N hours for pending signals (default 4)")

    def handle(self, *args, **options):
        dry_run   = options.get("dry_run", False)
        lookback  = options.get("hours", 4)
        api_key    = os.environ.get("BINANCE_API_KEY", "")
        api_secret = os.environ.get("BINANCE_API_SECRET", "")

        cutoff = timezone.now() - timedelta(hours=lookback)
        pending = SignalRecord.objects.filter(
            outcome="PENDING",
            created_at__gte=cutoff,
        ).order_by("created_at")

        if not pending.exists():
            self.stdout.write("No pending signals to check.")
            return

        updated = 0
        for sig in pending:
            since_ms = int(sig.created_at.timestamp() * 1000)

            # ── Try actual trade history first ────────────────────────
            actual_trades = get_actual_trades(sig.symbol, since_ms, api_key, api_secret)
            if actual_trades:
                outcome, close_price, pnl_pct = determine_outcome_from_trades(
                    sig, actual_trades)
            else:
                # ── Fall back to 5m candle simulation ─────────────────
                df = get_candles_5m(sig.symbol, since_ms, limit=60)
                outcome, close_price = determine_outcome_from_candles(sig, df)
                if close_price and sig.entry_price:
                    if sig.signal == "BUY":
                        pnl_pct = round((close_price - sig.entry_price) / sig.entry_price * 100, 2)
                    else:
                        pnl_pct = round((sig.entry_price - close_price) / sig.entry_price * 100, 2)
                else:
                    pnl_pct = 0.0

            if outcome == "PENDING":
                continue

            if dry_run:
                self.stdout.write(
                    f"[DRY] {sig.symbol} {sig.signal} → {outcome} "
                    f"@ {close_price:.6g} ({pnl_pct:+.2f}%)"
                )
            else:
                sig.outcome     = outcome
                sig.profit_pct  = pnl_pct
                sig.save(update_fields=["outcome", "profit_pct"])
                updated += 1
                log.info("Outcome updated: %s %s → %s (%+.2f%%)",
                         sig.symbol, sig.signal, outcome, pnl_pct)

        self.stdout.write(
            f"✅ Checked {pending.count()} signals — {updated} outcomes updated"
        )