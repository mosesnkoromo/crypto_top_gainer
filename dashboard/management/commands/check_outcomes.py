"""
dashboard/management/commands/check_outcomes.py
────────────────────────────────────────────────
Auto-checks PENDING signals against Binance price history
and updates outcomes (TP1/TP2/TP3/SL/BE) automatically.

Run once manually: python manage.py check_outcomes
Or scheduled every 15 min alongside the bot.
"""

import requests
import pandas as pd
from datetime import timedelta
from django.core.management.base import BaseCommand
from django.utils import timezone
from dashboard.models import SignalRecord


BINANCE_URL = "https://api.binance.com/api/v3/klines"


def get_candles_since(symbol: str, since_ts: int) -> pd.DataFrame:
    """Fetch 1h candles from signal creation time."""
    try:
        resp = requests.get(BINANCE_URL, params={
            "symbol": symbol, "interval": "1h",
            "startTime": since_ts, "limit": 200,
        }, timeout=12)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json(), columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades","buy_base","buy_quote","ignore"
        ])
        for c in ["high","low","close"]:
            df[c] = pd.to_numeric(df[c])
        return df
    except Exception:
        return pd.DataFrame()


def determine_outcome(signal: SignalRecord, df: pd.DataFrame) -> tuple[str, float]:
    """
    Walk candles forward from entry and check which level was hit first.
    Returns (outcome_code, close_price).
    """
    if df.empty:
        return "PENDING", None

    is_sell = signal.signal == "SELL"

    for _, row in df.iterrows():
        high = row["high"]
        low  = row["low"]

        if is_sell:
            # For SELL: TPs go DOWN, SL goes UP
            if low <= signal.tp3:   return "TP3", signal.tp3
            if low <= signal.tp2:   return "TP2", signal.tp2
            if low <= signal.tp1:   return "TP1", signal.tp1
            if high >= signal.sl:   return "SL",  signal.sl
        else:
            # For BUY: TPs go UP, SL goes DOWN
            if high >= signal.tp3:  return "TP3", signal.tp3
            if high >= signal.tp2:  return "TP2", signal.tp2
            if high >= signal.tp1:  return "TP1", signal.tp1
            if low  <= signal.sl:   return "SL",  signal.sl

    return "PENDING", None


def calc_profit(signal: SignalRecord, close_price: float) -> float:
    if not close_price or not signal.entry_price:
        return 0.0
    if signal.signal == "SELL":
        return round((signal.entry_price - close_price) / signal.entry_price * 100, 2)
    else:
        return round((close_price - signal.entry_price) / signal.entry_price * 100, 2)


class Command(BaseCommand):
    help = "Auto-check pending signal outcomes via Binance price history"

    def handle(self, *args, **options):
        # Only check signals < 7 days old
        cutoff   = timezone.now() - timedelta(days=7)
        pending  = SignalRecord.objects.filter(outcome="PENDING", created_at__gte=cutoff)
        total    = pending.count()
        updated  = 0

        self.stdout.write(f"Checking {total} pending signals...")

        for sig in pending:
            since_ms = int(sig.created_at.timestamp() * 1000)
            df = get_candles_since(sig.symbol, since_ms)

            outcome, close_price = determine_outcome(sig, df)

            if outcome != "PENDING":
                sig.outcome     = outcome
                sig.close_price = close_price
                sig.profit_pct  = calc_profit(sig, close_price)
                sig.auto_checked = True
                sig.closed_at   = timezone.now()
                sig.save()
                updated += 1
                self.stdout.write(
                    f"  ✅ {sig.symbol} {sig.signal} → {outcome} "
                    f"({sig.profit_pct:+.2f}%)"
                )

        self.stdout.write(
            self.style.SUCCESS(
                f"Done — {updated}/{total} signals updated"
            )
        )
