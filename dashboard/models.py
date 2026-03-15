"""
dashboard/models.py
────────────────────
All database models.
"""
import json
from django.db import models
from django.utils import timezone


class SignalRecord(models.Model):
    SIGNAL_CHOICES  = [("BUY","Buy"),("SELL","Sell")]
    GRADE_CHOICES   = [("ULTRA","Ultra"),("STRONG","Strong"),("STANDARD","Standard")]
    OUTCOME_CHOICES = [
        ("PENDING","Pending"),("TP1","TP1 Hit"),("TP2","TP2 Hit"),
        ("TP3","TP3 Hit"),("SL","Stop Loss"),("BE","Breakeven"),("MANUAL","Manual"),
    ]

    symbol       = models.CharField(max_length=20, db_index=True)
    signal       = models.CharField(max_length=4,  choices=SIGNAL_CHOICES, db_index=True)
    grade        = models.CharField(max_length=10, choices=GRADE_CHOICES,  db_index=True)
    confidence   = models.IntegerField()
    confluence   = models.FloatField()

    entry_price  = models.FloatField()
    tp1          = models.FloatField()
    tp2          = models.FloatField()
    tp3          = models.FloatField()
    sl           = models.FloatField()

    gain_24h     = models.FloatField()
    rsi          = models.FloatField()
    btc_score    = models.IntegerField()
    btc_trend    = models.CharField(max_length=40)
    factors      = models.TextField(default="[]")

    # Outcome — can be set manually or auto-detected
    outcome      = models.CharField(max_length=10, choices=OUTCOME_CHOICES, default="PENDING", db_index=True)
    close_price  = models.FloatField(null=True, blank=True)
    profit_pct   = models.FloatField(null=True, blank=True)
    auto_checked = models.BooleanField(default=False)  # True once auto-checked via Binance

    created_at   = models.DateTimeField(default=timezone.now, db_index=True)
    closed_at    = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes  = [
            models.Index(fields=["created_at","signal"]),
            models.Index(fields=["symbol","created_at"]),
        ]

    def __str__(self):
        return f"{self.signal} {self.symbol} [{self.grade}] {self.outcome}"

    @property
    def is_win(self):  return self.outcome in ("TP1","TP2","TP3")
    @property
    def is_loss(self): return self.outcome == "SL"

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
    date         = models.DateField(default=timezone.now, db_index=True)
    capital_usd  = models.FloatField()
    notes        = models.CharField(max_length=200, blank=True)
    created_at   = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-date"]

    def __str__(self):
        return f"${self.capital_usd:,.2f} on {self.date}"


class NewsItem(models.Model):
    """Crypto news cached from CryptoPanic — used for sentiment display."""
    title      = models.CharField(max_length=300)
    url        = models.URLField(max_length=500)
    source     = models.CharField(max_length=100)
    sentiment  = models.CharField(max_length=10, default="neutral")
    published  = models.DateTimeField()
    fetched_at = models.DateTimeField(default=timezone.now)
    currencies = models.CharField(max_length=200, default="")

    class Meta:
        ordering       = ["-published"]
        unique_together = [("title", "source")]

    def __str__(self):
        return f"[{self.sentiment}] {self.title[:60]}"