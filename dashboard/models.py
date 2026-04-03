"""
dashboard/models.py — v2
All database models.

Fixes vs v1:
  - SignalRecord.grade: max_length 10→20, no choices constraint (engine stores "ULTRA 🟢🟢🟢")
  - SignalRecord.notes: added (used for SPOT flags and auto-trade order IDs)
  - AutoTradeState.capital_usdt: default 0→100
  - AutoTradeState.get(): uses get_or_create(pk=1) — consistent with scanner
"""

import json
from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models
from django.utils import timezone


class SignalRecord(models.Model):
    SIGNAL_CHOICES  = [("BUY", "Buy"), ("SELL", "Sell")]
    OUTCOME_CHOICES = [
        ("PENDING", "Pending"), ("TP1", "TP1 Hit"), ("TP2", "TP2 Hit"),
        ("TP3", "TP3 Hit"), ("SL", "Stop Loss"), ("BE", "Breakeven"), ("MANUAL", "Manual"),
    ]

    symbol     = models.CharField(max_length=20, db_index=True)
    signal     = models.CharField(max_length=4,  choices=SIGNAL_CHOICES, db_index=True)

    # grade stores the full string from signal engine e.g. "ULTRA 🟢🟢🟢" or "STRONG 🟢🟢"
    # No choices constraint — engine appends emoji. max_length=30 handles full string.
    grade      = models.CharField(max_length=30, db_index=True)

    confidence = models.IntegerField()
    confluence = models.FloatField()

    entry_price = models.FloatField()
    tp1         = models.FloatField()
    tp2         = models.FloatField()
    tp3         = models.FloatField()
    sl          = models.FloatField()

    gain_24h    = models.FloatField()
    rsi         = models.FloatField()
    btc_score   = models.IntegerField()
    btc_trend   = models.CharField(max_length=40)
    factors     = models.TextField(default="[]")

    # Notes — used for SPOT flags, auto-trade order IDs, etc.
    notes       = models.TextField(blank=True, default="")

    # Outcome
    outcome     = models.CharField(max_length=10, choices=OUTCOME_CHOICES,
                                   default="PENDING", db_index=True)
    close_price = models.FloatField(null=True, blank=True)
    profit_pct  = models.FloatField(null=True, blank=True)
    auto_checked= models.BooleanField(default=False)

    created_at  = models.DateTimeField(default=timezone.now, db_index=True)
    closed_at   = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes  = [
            models.Index(fields=["created_at", "signal"]),
            models.Index(fields=["symbol", "created_at"]),
            models.Index(fields=["outcome", "created_at"]),
        ]

    def __str__(self):
        return f"{self.signal} {self.symbol} [{self.grade}] {self.outcome}"

    @property
    def is_win(self):  return self.outcome in ("TP1", "TP2", "TP3")

    @property
    def is_loss(self): return self.outcome == "SL"

    @property
    def grade_key(self) -> str:
        """Returns just the grade word without emoji: 'ULTRA', 'STRONG', 'STANDARD'."""
        return self.grade.split()[0] if self.grade else ""

    def get_factors_list(self):
        try:    return json.loads(self.factors)
        except: return []


class ScanRecord(models.Model):
    scanned_at    = models.DateTimeField(default=timezone.now, db_index=True)
    pairs_scanned = models.IntegerField(default=0)
    signals_found = models.IntegerField(default=0)
    signals_sent  = models.IntegerField(default=0)
    btc_score     = models.IntegerField(default=0)
    btc_price     = models.FloatField(default=0)
    btc_trend     = models.CharField(max_length=40, default="")

    class Meta:
        ordering = ["-scanned_at"]

    def __str__(self):
        return f"Scan {self.scanned_at:%Y-%m-%d %H:%M} — {self.signals_found} signals"


class CapitalRecord(models.Model):
    """Track capital growth over time."""
    date        = models.DateField(default=timezone.now, db_index=True)
    capital_usd = models.FloatField()
    notes       = models.CharField(max_length=200, blank=True)
    created_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-date"]

    def __str__(self):
        return f"${self.capital_usd:,.2f} on {self.date}"


class NewsItem(models.Model):
    """Crypto news cached from news APIs — used for sentiment display."""
    title      = models.CharField(max_length=300)
    url        = models.URLField(max_length=500)
    source     = models.CharField(max_length=100)
    sentiment  = models.CharField(max_length=10, default="neutral")
    published  = models.DateTimeField()
    fetched_at = models.DateTimeField(default=timezone.now)
    currencies = models.CharField(max_length=200, default="")

    class Meta:
        ordering        = ["-published"]
        unique_together = [("title", "source")]

    def __str__(self):
        return f"[{self.sentiment}] {self.title[:60]}"


class AutoTradeState(models.Model):
    """
    Singleton — independent on/off for Spot and Futures auto-trading.
    Each has its own risk%, max trades, and daily counter.
    """
    # ── Spot ─────────────────────────────────────────────────
    spot_enabled        = models.BooleanField(default=False)
    spot_risk           = models.FloatField(default=2.0,
                                            validators=[MinValueValidator(0.1), MaxValueValidator(5.0)])
    spot_max_trades     = models.PositiveIntegerField(default=5, validators=[MinValueValidator(1)])
    spot_trades_today   = models.PositiveIntegerField(default=0)
    spot_total          = models.PositiveIntegerField(default=0)

    # ── Futures ───────────────────────────────────────────────
    futures_enabled     = models.BooleanField(default=False)
    futures_risk        = models.FloatField(default=2.0,
                                            validators=[MinValueValidator(0.1), MaxValueValidator(5.0)])
    futures_max_trades  = models.PositiveIntegerField(default=5, validators=[MinValueValidator(1)])
    futures_trades_today= models.PositiveIntegerField(default=0)
    futures_total       = models.PositiveIntegerField(default=0)

    # ── Legacy / shared ───────────────────────────────────────
    # Kept for backward-compat; use spot_enabled / futures_enabled instead
    enabled         = models.BooleanField(default=False)
    mode            = models.CharField(max_length=10, default="spot")
    capital_usdt    = models.FloatField(default=100.0)
    risk_per_trade  = models.FloatField(default=2.0)
    max_trades_day  = models.PositiveIntegerField(default=5)
    trades_today    = models.PositiveIntegerField(default=0)
    total_auto_trades= models.PositiveIntegerField(default=0)
    updated_at      = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name        = "Auto Trade State"
        verbose_name_plural = "Auto Trade State"

    def __str__(self):
        parts = []
        if self.spot_enabled:    parts.append("Spot ON")
        if self.futures_enabled: parts.append("Futures ON")
        return "AutoTrade: " + (", ".join(parts) or "OFF")

    @classmethod
    def get(cls) -> "AutoTradeState":
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj

    def save(self, *args, **kwargs):
        for f in ("spot_risk","futures_risk","risk_per_trade"):
            v = getattr(self, f, None)
            if v is not None: setattr(self, f, max(0.1, min(float(v), 5.0)))
        for f in ("spot_max_trades","futures_max_trades","max_trades_day"):
            v = getattr(self, f, None)
            if v is not None and v < 1: setattr(self, f, 1)
        super().save(*args, **kwargs)


class ScalpPosition(models.Model):
    """Tracks when each futures position was opened for time-exit logic."""
    symbol    = models.CharField(max_length=20)
    side      = models.CharField(max_length=10)   # LONG or SHORT
    qty       = models.FloatField(default=0)
    entry     = models.FloatField(default=0)
    opened_at = models.DateTimeField(auto_now_add=True)
    closed    = models.BooleanField(default=False)
    close_reason = models.CharField(max_length=20, default="")  # TP1/TP2/SL/TIME/MANUAL

    class Meta:
        ordering = ["-opened_at"]

    def __str__(self):
        return f"{self.symbol} {self.side} @ {self.entry} ({'closed' if self.closed else 'open'})"

    @classmethod
    def open(cls, symbol, side, qty, entry):
        return cls.objects.create(symbol=symbol, side=side, qty=qty, entry=entry)

    @classmethod
    def close_sym(cls, symbol, reason=""):
        cls.objects.filter(symbol=symbol, closed=False).update(
            closed=True, close_reason=reason)