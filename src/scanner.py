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
from src.analysis.spot_signal_engine import SpotSignalEngine, SpotSignal
from src.trading.binance_trader import BinanceTrader, TradeResult
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
        """
        Walk candles forward and find the HIGHEST TP reached before SL.

        Simulates realistic trade management:
          - Track highest TP hit so far
          - Once TP1 hit, move SL to breakeven (entry price)
          - Once TP2 hit, move SL to TP1
          - SL can only be hit BEFORE first TP or if price reverses past last TP
          - Final outcome = highest TP reached
        """
        try:
            resp = requests.get(_BINANCE_KLINES, params={
                "symbol": sig.symbol, "interval": "1h",
                "startTime": since_ms, "limit": 168,
            }, timeout=12)
            resp.raise_for_status()
            df = pd.DataFrame(resp.json(), columns=[
                "open_time","open","high","low","close","volume",
                "close_time","quote_vol","trades","buy_base","buy_quote","ignore"
            ])
            for col in ["high","low"]:
                df[col] = pd.to_numeric(df[col])
        except Exception:
            return "PENDING", None

        is_sell    = sig.signal == "SELL"
        best_tp    = None          # highest TP reached so far
        trail_sl   = sig.sl        # trailing SL — moves up as TPs are hit

        for _, row in df.iterrows():
            h, l = float(row["high"]), float(row["low"])

            if is_sell:
                # Check SL first (before TPs on same candle is conservative)
                if trail_sl is not None and h >= trail_sl:
                    # SL hit — exit with whatever we have
                    return best_tp or "SL", (best_tp and {
                        "TP1": sig.tp1, "TP2": sig.tp2, "TP3": sig.tp3
                    }.get(best_tp)) or sig.sl

                if l <= sig.tp3:
                    return "TP3", sig.tp3          # full target hit
                if l <= sig.tp2:
                    best_tp  = "TP2"
                    trail_sl = sig.tp1             # SL moves to TP1 level
                elif l <= sig.tp1 and best_tp is None:
                    best_tp  = "TP1"
                    trail_sl = sig.entry_price     # SL moves to breakeven
            else:
                if trail_sl is not None and l <= trail_sl:
                    return best_tp or "SL", (best_tp and {
                        "TP1": sig.tp1, "TP2": sig.tp2, "TP3": sig.tp3
                    }.get(best_tp)) or sig.sl

                if h >= sig.tp3:
                    return "TP3", sig.tp3
                if h >= sig.tp2:
                    best_tp  = "TP2"
                    trail_sl = sig.tp1
                elif h >= sig.tp1 and best_tp is None:
                    best_tp  = "TP1"
                    trail_sl = sig.entry_price

        # End of candles — return best reached
        return best_tp or "PENDING", ({
            "TP1": sig.tp1, "TP2": sig.tp2, "TP3": sig.tp3
        }.get(best_tp)) if best_tp else None

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
        self._spot   = SpotSignalEngine(binance)
        self._trader: "BinanceTrader | None" = None
        self._trader_mode: str = ""   # track current mode so we rebuild if mode changes
        self._wa      = WhatsAppSender(cfg.whatsapp)
        self._bin     = binance
        self._checker = OutcomeChecker()
        self._learner = PatternLearner()

        self._cooldowns: dict[str, datetime] = {}
        self._last_btc_update: datetime | None = None
        self._last_daily_report: datetime | None = None
        self._daily_scan_count: int = 0
        self._pending_symbols: set[str] = set()

    def run_cycle(self) -> None:
        from dashboard.models import ScanRecord, SignalRecord, NewsItem

        now = timezone.now()
        log.info("=" * 65)
        from zoneinfo import ZoneInfo
        _EAT = ZoneInfo("Africa/Dar_es_Salaam")
        now_eat = now.astimezone(_EAT)
        log.info("Scan cycle started at %s EAT", now_eat.strftime("%H:%M:%S"))

        # ── Step 1: Auto-resolve pending outcomes ────────────
        resolved = self._checker.check_pending()
        if resolved:
            log.info("Auto-resolved %d pending signal outcomes", resolved)

        # ── Step 2: Block symbols with pending OR today's signals ──
        # Prevents: same pair signaled twice in same day, or re-signaling
        # while a previous trade is still open
        try:
            from django.utils import timezone as tz
            today_start = tz.now().replace(hour=0, minute=0, second=0, microsecond=0)
            self._pending_symbols = set(
                SignalRecord.objects
                .filter(created_at__gte=today_start)  # any signal today
                .values_list("symbol", flat=True)
            ) | set(
                SignalRecord.objects
                .filter(outcome="PENDING")             # any open trade
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
        # v7: scan BOTH top gainers (momentum) AND liquid pairs (trends)
        # This gives 50-60 pairs per cycle → 5-15 signals daily
        top_gainers  = self._bin.get_top_gainers(30)
        liquid_pairs = self._bin.get_trending_pairs(30)
        # Merge, deduplicate by symbol
        seen = set()
        gainers = []
        for t in top_gainers + liquid_pairs:
            if t["symbol"] not in seen:
                seen.add(t["symbol"])
                gainers.append(t)
        log.info("Pairs to scan: %d (top gainers + liquid) | BTC Score: %d/100", len(gainers), btc.score)

        # ── Step 5: Analyse pairs ─────────────────────────────
        collected: list[Signal] = []
        spot_signals: list = []   # spot engine results (default empty)
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

        # Keep only top 5 by confluence, same direction as majority
        buy_sigs  = [s for s in collected if s.signal == "BUY"]
        sell_sigs = [s for s in collected if s.signal == "SELL"]
        if buy_sigs and sell_sigs:
            # If both directions fire, keep only the stronger side
            buy_avg  = sum(s.confluence for s in buy_sigs)  / len(buy_sigs)
            sell_avg = sum(s.confluence for s in sell_sigs) / len(sell_sigs)
            collected = buy_sigs if buy_avg >= sell_avg else sell_sigs
            log.info("Direction consensus: kept %s only (buy avg=%.1f sell avg=%.1f)",
                     "BUY" if buy_avg >= sell_avg else "SELL", buy_avg, sell_avg)

        MAX_SIGNALS = 5
        collected = sorted(collected, key=lambda s: s.confluence, reverse=True)[:MAX_SIGNALS]
        if len(collected) < len(buy_sigs) + len(sell_sigs):
            log.info("Signal cap: trimmed to top %d by confluence", MAX_SIGNALS)

        # Always send combined report every scan — weekly stats + open positions + new signals
        if not collected:
            import inspect as _ins
            _fmt_p = _ins.signature(fmt_digest).parameters
            _extra = {"closed_today": [], "open_positions": [], "scan_number": self._daily_scan_count} if "closed_today" in _fmt_p else {}
            if "spot_signals" in _fmt_p:
                _extra["spot_signals"] = spot_signals
            msg = fmt_digest([], btc.to_dict(), self._cfg.risk, news_items, **_extra)
            self._wa.send(msg)
            log.info("Combined report sent (no new signals)")

        if collected:
            # Save to DB
            for s in collected:
                try:
                    SignalRecord.objects.create(
                        symbol=s.symbol, signal=s.signal,
                        grade=s.grade.split()[0], confidence=s.confidence,
                        confluence=s.confluence, entry_price=s.price,
                        tp1=s.tp1, tp2=s.tp2, tp3=s.tp3, sl=s.sl,
                        gain_24h=s.gain_24h, rsi=getattr(s, 'rsi_1h', getattr(s, 'rsi', 0)),
                        btc_score=s.btc_score, btc_trend=s.btc_trend,
                        factors=json.dumps(s.factors),
                    )
                except Exception as e:
                    log.warning("DB save failed %s: %s", s.symbol, e)
                self._cooldowns[s.symbol] = now

            # Pull today's closed + open signals for daily summary
            try:
                from dashboard.models import SignalRecord
                from django.utils import timezone as dtz
                from zoneinfo import ZoneInfo
                _EAT_z = ZoneInfo("Africa/Dar_es_Salaam")
                today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

                closed_today = []
                for sig_rec in SignalRecord.objects.filter(
                    created_at__gte=today_start
                ).exclude(outcome="PENDING").order_by("-closed_at")[:10]:
                    closed_today.append({
                        "symbol":   sig_rec.symbol,
                        "signal":   sig_rec.signal,
                        "grade":    sig_rec.grade,
                        "outcome":  sig_rec.outcome,
                        "pnl":      sig_rec.profit_pct,
                        "entry_price": sig_rec.entry_price,
                        "tp1": sig_rec.tp1, "tp2": sig_rec.tp2, "tp3": sig_rec.tp3,
                        "sl": sig_rec.sl,
                        "closed_at": sig_rec.closed_at.astimezone(_EAT_z).strftime("%H:%M") if sig_rec.closed_at else "",
                    })

                open_positions = []
                for sig_rec in SignalRecord.objects.filter(
                    created_at__gte=today_start, outcome="PENDING"
                ).order_by("-created_at")[:5]:
                    open_positions.append({
                        "symbol":   sig_rec.symbol,
                        "signal":   sig_rec.signal,
                        "grade":    sig_rec.grade,
                        "entry_price": sig_rec.entry_price,
                        "tp1": sig_rec.tp1, "tp2": sig_rec.tp2, "tp3": sig_rec.tp3,
                        "sl": sig_rec.sl,
                    })
            except Exception as _e:
                log.debug("Daily data fetch error: %s", _e)
                closed_today, open_positions = [], []

            # Build extra kwargs only if formatter supports them
            import inspect as _inspect
            _fmt_params = _inspect.signature(fmt_digest).parameters
            _extra = {}
            if "closed_today" in _fmt_params:
                _extra["closed_today"]   = closed_today
                _extra["open_positions"] = open_positions
                _extra["scan_number"]    = self._daily_scan_count
            msg = fmt_digest(collected, btc.to_dict(), self._cfg.risk, news_items, **_extra)
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

        # Auto-trade runs every cycle — retries today pending signals too
        self._run_auto_trade(collected, now)

        log.info("Cycle complete")
        self._daily_scan_count += 1
        # Reset auto-trade daily count at midnight EAT
        try:
            from dashboard.models import AutoTradeState
            from zoneinfo import ZoneInfo
            eat_hour = now.astimezone(ZoneInfo("Africa/Dar_es_Salaam")).hour
            eat_min  = now.astimezone(ZoneInfo("Africa/Dar_es_Salaam")).minute
            if eat_hour == 0 and eat_min < 16:
                state = AutoTradeState.get()
                if state.trades_today > 0:
                    state.trades_today = 0
                    state.save(update_fields=["trades_today"])
        except Exception: pass
        # Daily report is now embedded in every signal digest (not time-gated)

    # ── Helpers ───────────────────────────────────────────────

    def _run_auto_trade(self, new_signals: list, now):
        """
        Execute auto-trades for:
          1. New signals from this cycle
          2. Today PENDING DB signals not yet auto-traded (retry)
        Runs every scan cycle regardless of whether new signals fired.
        """
        try:
            from dashboard.models import AutoTradeState, SignalRecord
            state = AutoTradeState.get()
            if not state.enabled:
                return
            trader = self._get_trader(state)
            if not trader:
                log.debug("Auto-trade ON but trader not ready")
                return

            balance = trader.get_balance()
            log.info("Auto-trade [%s] balance=$%.2f new=%d",
                     state.mode.upper(), balance, len(new_signals))

            # Build candidates: new signals + pending DB not yet traded
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            pending_recs = list(
                SignalRecord.objects.filter(
                    created_at__gte=today_start,
                    outcome="PENDING",
                ).exclude(notes__icontains="AUTO:")
            )

            seen = set()

            # --- new signals first ---
            for sig in new_signals:
                grade_key = sig.grade.split()[0]
                if grade_key not in ("ULTRA", "STRONG"):
                    continue
                if state.trades_today >= state.max_trades_day:
                    break
                if sig.signal == "SELL" and state.mode == "spot":
                    continue
                if sig.symbol in seen:
                    continue
                seen.add(sig.symbol)
                result = trader.execute_signal(sig, balance)
                self._save_trade_result(sig, result)
                if result.success:
                    balance -= result.qty * result.entry_price
                    state.trades_today += 1
                    state.total_auto_trades += 1
                    state.save(update_fields=["trades_today", "total_auto_trades"])
                    log.info("AUTO TRADE OK: %s %s @ %.6g | oco=%s sl=%s",
                             result.side, sig.symbol, result.entry_price,
                             result.oco_id, result.sl_order_id)
                else:
                    log.warning("AUTO TRADE FAIL: %s %s — %s",
                                sig.signal, sig.symbol, result.error)

            # --- pending DB retries ---
            for rec in pending_recs:
                if rec.symbol in seen:
                    continue
                grade_key = rec.grade.split()[0]
                if grade_key not in ("ULTRA", "STRONG"):
                    continue
                if state.trades_today >= state.max_trades_day:
                    break
                if rec.signal == "SELL" and state.mode == "spot":
                    continue
                seen.add(rec.symbol)

                # Build minimal signal-like object from DB record
                class _S:
                    pass
                sig = _S()
                sig.symbol    = rec.symbol
                sig.signal    = rec.signal
                sig.grade     = rec.grade
                sig.price     = rec.entry_price
                sig.tp1       = rec.tp1
                sig.tp2       = rec.tp2
                sig.tp3       = rec.tp3
                sig.sl        = rec.sl
                sig.btc_score = rec.btc_score
                sig.confidence= rec.confidence

                result = trader.execute_signal(sig, balance)
                # Save result to DB record notes
                try:
                    mark = "YES" if result.success else ("FAIL:" + result.error[:40])
                    rec.notes = (rec.notes or "") + f" | AUTO:{mark}"
                    rec.save(update_fields=["notes"])
                except Exception:
                    pass

                if result.success:
                    balance -= result.qty * result.entry_price
                    state.trades_today += 1
                    state.total_auto_trades += 1
                    state.save(update_fields=["trades_today", "total_auto_trades"])
                    log.info("AUTO TRADE (retry) OK: %s %s @ %.6g",
                             result.side, rec.symbol, result.entry_price)
                else:
                    log.warning("AUTO TRADE (retry) FAIL: %s %s — %s",
                                rec.signal, rec.symbol, result.error)

        except Exception as e:
            log.error("Auto-trade error: %s", e, exc_info=True)

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
        # FIX 5: 4-hour cooldown per symbol regardless of direction.
        # Report showed GUSDT signaled 6x in one day, MEUSDT 5x — compounding losses.
        # The cooldown now blocks BOTH BUY and SELL on the same symbol for 4 hours
        # after any signal (win or loss) to prevent repeated entries on volatile pairs.
        COOLDOWN_SECONDS = max(
            self._cfg.alert.cooldown_hours * 3600,
            24 * 3600   # minimum 24 hours for swing trading — let the trade breathe
        )
        last = self._cooldowns.get(symbol)
        if last is None:
            return False
        return (now - last).total_seconds() < COOLDOWN_SECONDS