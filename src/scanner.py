"""
src/scanner.py — v4
Full automated scan loop:
  1. Auto-check outcomes of previous pending signals (Binance price history)
  2. Fetch news & score sentiment per coin
  3. Calculate BTC Strength Score
  4. Scan top gainers with 7-factor confluence + news modifier
  5. Save signals to DB & send single WhatsApp digest
  6. Learn from past wins/losses to adjust confidence weighting

All automated — no manual commands needed.
"""

import json
import os
import time
from datetime import datetime, timedelta
from django.utils import timezone
from collections import defaultdict

import django
import requests
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "btc_project.settings")
try:
    django.setup()
except RuntimeError:
    pass

from config import AppConfig
from src.alerts.whatsapp import WhatsAppSender
from src.analysis.btc_strength import BtcStrengthEngine
from src.analysis.news_engine import NewsEngine
from src.analysis.signal_engine import Signal, SignalEngine
from src.data.binance_client import BinanceClient
from src.utils.formatter import fmt_btc_update, fmt_digest, fmt_no_signals
from src.utils.logger import get_logger

log = get_logger(__name__)

_GRADE_ORDER    = {"ULTRA": 0, "STRONG": 1, "STANDARD": 2}
_BINANCE_KLINES = "https://api.binance.com/api/v3/klines"


# ─────────────────────────────────────────────────────────────
#  Pattern Learner — adjusts confidence based on past outcomes
# ─────────────────────────────────────────────────────────────

class PatternLearner:
    """
    Tracks which confluence factors correlate with wins/losses.
    Returns a confidence adjustment (-10 to +10) for each new signal
    based on historical win rates for that grade + BTC score range.
    """

    def __init__(self):
        self._cache: dict = {}
        self._refreshed_at = None

    def get_confidence_boost(self, grade: str, btc_score: int, symbol: str) -> int:
        """
        Returns confidence adjustment based on:
        - Historical win rate for this grade
        - BTC score range performance
        - Symbol-specific win rate
        """
        self._refresh_if_stale()
        boost = 0

        # Grade performance boost
        grade_wr = self._cache.get(f"grade_{grade}", 50)
        if grade_wr >= 75:   boost += 5
        elif grade_wr >= 65: boost += 3
        elif grade_wr <= 40: boost -= 5
        elif grade_wr <= 50: boost -= 3

        # BTC range performance
        btc_range = self._btc_range(btc_score)
        range_wr  = self._cache.get(f"btc_{btc_range}", 50)
        if range_wr >= 70:   boost += 3
        elif range_wr <= 40: boost -= 3

        # Symbol performance
        sym_wr = self._cache.get(f"sym_{symbol}", None)
        if sym_wr is not None:
            if sym_wr >= 80:   boost += 4
            elif sym_wr <= 35: boost -= 4

        return max(-10, min(10, boost))

    def _refresh_if_stale(self):
        if (self._refreshed_at and
                (timezone.now() - self._refreshed_at).total_seconds() < 3600):
            return
        try:
            from dashboard.models import SignalRecord
            from django.db.models import Count, Q

            cache = {}
            since = timezone.now() - timedelta(days=60)

            # Grade win rates
            for grade in ("ULTRA", "STRONG", "STANDARD"):
                qs    = SignalRecord.objects.filter(grade=grade, created_at__gte=since).exclude(outcome="PENDING")
                total = qs.count()
                wins  = qs.filter(outcome__in=["TP1","TP2","TP3"]).count()
                if total >= 5:
                    cache[f"grade_{grade}"] = round(wins / total * 100)

            # BTC range win rates
            for rng in ("high","mid","low"):
                if rng == "high":   qs = SignalRecord.objects.filter(btc_score__gte=60, created_at__gte=since)
                elif rng == "mid":  qs = SignalRecord.objects.filter(btc_score__gte=40, btc_score__lt=60, created_at__gte=since)
                else:               qs = SignalRecord.objects.filter(btc_score__lt=40, created_at__gte=since)
                qs    = qs.exclude(outcome="PENDING")
                total = qs.count()
                wins  = qs.filter(outcome__in=["TP1","TP2","TP3"]).count()
                if total >= 5:
                    cache[f"btc_{rng}"] = round(wins / total * 100)

            # Top symbol win rates
            pairs = (SignalRecord.objects.filter(created_at__gte=since)
                     .exclude(outcome="PENDING")
                     .values("symbol")
                     .annotate(total=Count("id"),
                               wins=Count("id", filter=Q(outcome__in=["TP1","TP2","TP3"])))
                     .filter(total__gte=3))
            for p in pairs:
                cache[f"sym_{p['symbol']}"] = round(p["wins"] / p["total"] * 100)

            self._cache        = cache
            self._refreshed_at = timezone.now()
            log.info("PatternLearner refreshed — %d entries", len(cache))
        except Exception as e:
            log.warning("PatternLearner refresh failed: %s", e)

    @staticmethod
    def _btc_range(score: int) -> str:
        if score >= 60: return "high"
        if score >= 40: return "mid"
        return "low"


# ─────────────────────────────────────────────────────────────
#  Auto Outcome Checker
# ─────────────────────────────────────────────────────────────

class OutcomeChecker:
    """
    Automatically resolves PENDING signals by checking Binance
    candle history to see which level (TP1/TP2/TP3/SL) was hit first.
    Runs at the start of each scan cycle.
    """

    def check_pending(self) -> int:
        """Returns number of signals resolved."""
        try:
            from dashboard.models import SignalRecord

            cutoff  = timezone.now() - timedelta(days=7)
            pending = SignalRecord.objects.filter(outcome="PENDING", created_at__gte=cutoff)
            count   = 0

            for sig in pending:
                since_ms = int(sig.created_at.timestamp() * 1000)
                outcome, close_price = self._determine(sig, since_ms)
                if outcome != "PENDING":
                    sig.outcome      = outcome
                    sig.close_price  = close_price
                    sig.profit_pct   = self._calc_pnl(sig, close_price)
                    sig.auto_checked = True
                    sig.closed_at    = timezone.now()
                    sig.save()
                    count += 1
                    log.info(
                        "Auto-outcome: %s %s → %s (%+.2f%%)",
                        sig.signal, sig.symbol, outcome, sig.profit_pct or 0
                    )
            return count
        except Exception as e:
            log.warning("OutcomeChecker error: %s", e)
            return 0

    def _determine(self, sig, since_ms: int) -> tuple[str, float | None]:
        try:
            resp = requests.get(_BINANCE_KLINES, params={
                "symbol": sig.symbol, "interval": "1h",
                "startTime": since_ms, "limit": 168,  # 7 days
            }, timeout=12)
            resp.raise_for_status()
            df = pd.DataFrame(resp.json(), columns=[
                "open_time","open","high","low","close","volume",
                "close_time","quote_vol","trades","buy_base","buy_quote","ignore"
            ])
            for c in ["high","low"]:
                df[c] = pd.to_numeric(df[c])
        except Exception:
            return "PENDING", None

        is_sell = sig.signal == "SELL"
        for _, row in df.iterrows():
            h, l = row["high"], row["low"]
            if is_sell:
                if l <= sig.tp3:  return "TP3", sig.tp3
                if l <= sig.tp2:  return "TP2", sig.tp2
                if l <= sig.tp1:  return "TP1", sig.tp1
                if h >= sig.sl:   return "SL",  sig.sl
            else:
                if h >= sig.tp3:  return "TP3", sig.tp3
                if h >= sig.tp2:  return "TP2", sig.tp2
                if h >= sig.tp1:  return "TP1", sig.tp1
                if l  <= sig.sl:  return "SL",  sig.sl
        return "PENDING", None

    @staticmethod
    def _calc_pnl(sig, close_price) -> float | None:
        if not close_price or not sig.entry_price:
            return None
        if sig.signal == "SELL":
            return round((sig.entry_price - close_price) / sig.entry_price * 100, 2)
        return round((close_price - sig.entry_price) / sig.entry_price * 100, 2)


# ─────────────────────────────────────────────────────────────
#  Main Scanner
# ─────────────────────────────────────────────────────────────

class Scanner:
    """
    Full automated scan cycle:
    1. Auto-resolve pending signal outcomes
    2. Fetch & cache news sentiment
    3. Score BTC Strength
    4. Analyse top gainers with 7-factor confluence + news modifier
    5. Apply pattern-learning confidence boosts
    6. Save to DB + send single WhatsApp digest
    """

    def __init__(self, cfg: AppConfig):
        self._cfg     = cfg
        binance       = BinanceClient(cfg.binance, cfg.scan)
        self._btc     = BtcStrengthEngine(binance, cfg.scan)
        self._news    = NewsEngine(os.environ.get("CRYPTOPANIC_API_KEY", ""))
        self._sig     = SignalEngine(binance, cfg.signal, cfg.risk, cfg.scan, self._news)
        self._wa      = WhatsAppSender(cfg.whatsapp)
        self._bin     = binance
        self._checker = OutcomeChecker()
        self._learner = PatternLearner()

        self._cooldowns: dict[str, datetime] = {}
        self._last_btc_update: datetime | None = None
        # Track symbols with existing pending signal (no duplicates)
        self._pending_symbols: set[str] = set()

    def run_cycle(self) -> None:
        from dashboard.models import ScanRecord, SignalRecord, NewsItem

        now = timezone.now()
        log.info("=" * 65)
        log.info("Scan cycle started at %s UTC", now.strftime("%H:%M:%S"))

        # ── Step 1: Auto-resolve pending outcomes ────────────
        resolved = self._checker.check_pending()
        if resolved:
            log.info("Auto-resolved %d pending signal outcomes", resolved)

        # ── Step 2: Refresh pending symbol set (no duplicates) ──
        try:
            self._pending_symbols = set(
                SignalRecord.objects.filter(outcome="PENDING")
                .values_list("symbol", flat=True)
            )
        except Exception:
            self._pending_symbols = set()

        # ── Step 3: Fetch news (cached 30 min) ───────────────
        news_items = self._news.get_news(50)
        log.info("News: %d items available for sentiment scoring", len(news_items))

        # Save new news to DB
        for n in news_items:
            try:
                from django.utils.dateparse import parse_datetime
                NewsItem.objects.get_or_create(
                    title=n["title"][:298], source=n["source"],
                    defaults={
                        "url": n.get("url",""), "sentiment": n.get("sentiment","neutral"),
                        "published": timezone.now(), "currencies": n.get("currencies",""),
                    }
                )
            except Exception:
                pass

        # ── Step 4: BTC Strength ──────────────────────────────
        btc     = self._btc.calculate()
        gainers = self._bin.get_top_gainers()
        log.info("Top gainers eligible: %d | BTC Score: %d/100", len(gainers), btc.score)

        # ── Step 5: Analyse pairs ─────────────────────────────
        collected: list[Signal] = []
        for ticker in gainers:
            sym = ticker["symbol"]

            # Skip if in cooldown
            if self._is_in_cooldown(sym, now):
                log.debug("Cooldown: %s", sym)
                continue

            # Skip if already has a pending signal for this pair
            if sym in self._pending_symbols:
                log.debug("Skipping %s — pending signal exists", sym)
                continue

            signal = self._sig.analyze(ticker, btc)
            if signal:
                # Apply pattern-learning confidence boost
                boost = self._learner.get_confidence_boost(
                    signal.grade.split()[0], signal.btc_score, sym
                )
                if boost != 0:
                    signal = self._apply_boost(signal, boost)
                    log.info("Pattern boost %+d%% applied to %s", boost, sym)

                collected.append(signal)
                log.info(
                    "Queued: %s %s — %s (%d%%) confluence=%.1f",
                    signal.signal, sym, signal.grade, signal.confidence, signal.confluence,
                )

            time.sleep(self._cfg.alert.binance_rate_limit_seconds)

        log.info("Scan done — %d pairs, %d signals queued", len(gainers), len(collected))

        # ── Step 6: Save scan record ──────────────────────────
        scan_rec = ScanRecord.objects.create(
            pairs_scanned=len(gainers), signals_found=len(collected),
            signals_sent=0, btc_score=btc.score,
            btc_price=btc.price, btc_trend=btc.trend,
        )

        # ── Step 7: Sort & send ───────────────────────────────
        collected.sort(key=lambda s: _GRADE_ORDER.get(s.grade.split()[0], 9))

        # BTC update only when no signals and timer elapsed
        if self._should_send_btc_update(now) and not collected:
            self._wa.send(fmt_btc_update(btc.to_dict()))
            self._last_btc_update = now
            log.info("BTC update sent (no signals)")

        if collected:
            # Save to DB
            for s in collected:
                try:
                    SignalRecord.objects.create(
                        symbol=s.symbol, signal=s.signal,
                        grade=s.grade.split()[0], confidence=s.confidence,
                        confluence=s.confluence, entry_price=s.price,
                        tp1=s.tp1, tp2=s.tp2, tp3=s.tp3, sl=s.sl,
                        gain_24h=s.gain_24h, rsi=s.rsi,
                        btc_score=s.btc_score, btc_trend=s.btc_trend,
                        factors=json.dumps(s.factors),
                    )
                except Exception as e:
                    log.warning("DB save failed %s: %s", s.symbol, e)
                self._cooldowns[s.symbol] = now

            msg       = fmt_digest(collected, btc.to_dict(), self._cfg.risk, news_items)
            delivered = self._wa.send(msg)

            if delivered:
                self._last_btc_update = now
                scan_rec.signals_sent = len(collected)
                scan_rec.save()
                log.info("Digest delivered (%d signals)", len(collected))
            else:
                # Roll back cooldowns on failure so signals retry
                for s in collected:
                    self._cooldowns.pop(s.symbol, None)
                log.error("Digest failed — will retry next cycle")
        else:
            log.info("No signals this cycle")

        log.info("Cycle complete")

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _apply_boost(signal: Signal, boost: int) -> Signal:
        """Return a new Signal with adjusted confidence."""
        from dataclasses import replace
        new_conf = max(50, min(95, signal.confidence + boost))
        tag = f"✅ Pattern boost +{boost}%" if boost > 0 else f"⚠️ Pattern penalty {boost}%"
        new_factors = signal.factors + [tag]
        return replace(signal, confidence=new_conf, factors=new_factors)

    def _should_send_btc_update(self, now: datetime) -> bool:
        if self._last_btc_update is None:
            return True
        return (now - self._last_btc_update).total_seconds() >= self._cfg.alert.btc_update_every_hours * 3600

    def _is_in_cooldown(self, symbol: str, now: datetime) -> bool:
        last = self._cooldowns.get(symbol)
        if last is None:
            return False
        return (now - last).total_seconds() < self._cfg.alert.cooldown_hours * 3600