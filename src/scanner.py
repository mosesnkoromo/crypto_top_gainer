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
from src.analysis.signal_engine import Signal, SignalEngine, get_signal_engine
from src.analysis.spot_signal_engine import SpotSignalEngine, SpotSignal
from src.trading.binance_trader import BinanceTrader, TradeResult
from src.data.binance_client import BinanceClient
from src.utils.formatter import fmt_btc_update, fmt_digest, fmt_no_signals
from src.utils.logger import get_logger
from src.analysis.signal_simulator import get_simulator, SimResult

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
                "symbol": sig.symbol, "interval": "5m",
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
        self._sig     = get_signal_engine(binance)
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
        self._pair_cooldowns: dict = {}   # {symbol: datetime} — 10min cooldown after loss
        self._delayed_entries: dict = {}  # {symbol: {signal, retries, added_at}} delayed retry
        self._sim = get_simulator()       # pre-trade backtest simulation gate

    def run_cycle(self) -> None:
        from dashboard.models import ScanRecord, SignalRecord, NewsItem

        now = timezone.now()
        log.info("=" * 65)
        from zoneinfo import ZoneInfo
        _EAT = ZoneInfo("Africa/Dar_es_Salaam")
        now_eat = now.astimezone(_EAT)
        log.info("Scan cycle started at %s EAT", now_eat.strftime("%H:%M:%S"))

        # ── DAILY LOSS CIRCUIT BREAKER ────────────────────────────────
        try:
            from dashboard.models import SignalRecord
            _today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            _day_losses  = list(SignalRecord.objects.filter(
                created_at__gte=_today_start, outcome="SL"
            ).values_list("profit_pct", flat=True))
            _total_loss  = sum(abs(p or 0) for p in _day_losses)
            if _total_loss > 4.0:  # >4% total SL losses today → pause
                log.warning("🛑 CIRCUIT BREAKER: %.1f%% SL losses today — pausing new entries",
                            _total_loss)
                return
        except Exception:
            pass

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
            sym = t["symbol"]
            if sym not in seen:
                seen.add(sym)
                gainers.append(t)
        log.info("Pairs to scan: %d (top gainers + liquid) | BTC Score: %d/100", len(gainers), btc.score)

        # ── CRASH PROTECTION: BTC < 15 → close all LONG positions ────
        if btc.score < 15:
            try:
                from dashboard.models import AutoTradeState as _ATSC
                from config import load_config as _lcc
                _stc = _ATSC.get()
                _cfgc = _lcc()
                if _stc.futures_enabled and _cfgc.auto.has_keys:
                    from src.trading.binance_trader import BinanceTrader as _BTC2
                    _ftc = _BTC2(_cfgc.auto.api_key, _cfgc.auto.api_secret,
                                 mode="futures", live=not _cfgc.auto.testnet)
                    for _pc in _ftc.get_positions():
                        if float(_pc.get("positionAmt", 0)) > 0:
                            _sc2 = _pc.get("symbol", "")
                            _qc  = abs(float(_pc.get("positionAmt", 0)))
                            _ftc._req("DELETE", "/fapi/v1/allOpenOrders", {"symbol": _sc2})
                            _ftc._req("POST", "/fapi/v1/order", {
                                "symbol": _sc2, "side": "SELL", "type": "MARKET",
                                "quantity": _qc, "reduceOnly": "true"})
                            log.warning("🚨 CRASH PROTECTION: closed LONG %s (BTC=%d/100)",
                                        _sc2, btc.score)
            except Exception as _ce:
                log.debug("Crash protection: %s", _ce)

        # ── Step 4b: Retry delayed entries ───────────────────
        # Signals that passed score but sniper rejected — retry up to 3 cycles
        collected: list[Signal] = []      # define early so delayed entries can append
        spot_signals: list = []
        to_remove = []
        for _dsym, _de in list(self._delayed_entries.items()):
            _dsig    = _de["signal"]
            _retries = _de["retries"]
            _added   = _de["added_at"]
            # Expire after 3 retries (6 minutes) or 10 min max
            age_min = (now - _added).total_seconds() / 60
            if _retries >= 3 or age_min > 10:
                to_remove.append(_dsym)
                log.debug("Delayed entry %s expired (%d retries, %.0fmin)", _dsym, _retries, age_min)
                continue
            try:
                # Fetch fresh ticker
                _fresh = self._bin.get_ticker(_dsym)
                if not _fresh:
                    _de["retries"] += 1; continue
                _cur_p = float(_fresh.get("lastPrice", 0) or 0)
                _entry_p = float(_dsig.price)
                if _entry_p <= 0:
                    to_remove.append(_dsym); continue
                # Price must still be within 0.5% of original entry
                _drift = abs(_cur_p - _entry_p) / _entry_p * 100
                if _drift > 0.5:
                    log.debug("Delayed %s: price drifted %.2f%% — discarding", _dsym, _drift)
                    to_remove.append(_dsym); continue
                # Re-check sniper on fresh 1m candle
                _df1m = self._bin.get_klines(_dsym, "1m", 20)
                from src.analysis.indicators import ema_value
                _sniper_ok = True
                if not _df1m.empty and len(_df1m) >= 10:
                    _c1m = _df1m["close"].astype(float)
                    _e9  = ema_value(_c1m, 9)
                    _e21 = ema_value(_c1m, 21)
                    _is_buy = _dsig.signal == "BUY"
                    _sniper_ok = (_e9 > _e21) if _is_buy else (_e9 < _e21)
                if _sniper_ok:
                    log.info("  ⏰ %s DELAYED ENTRY fired (retry %d, drift=%.2f%%)",
                             _dsym, _retries+1, _drift)
                    collected.append(_dsig)
                    to_remove.append(_dsym)
                else:
                    _de["retries"] += 1
                    log.debug("Delayed %s: sniper still weak (retry %d)", _dsym, _retries+1)
            except Exception as _dex:
                log.debug("Delayed entry check %s: %s", _dsym, _dex)
                _de["retries"] += 1
        for _k in to_remove:
            self._delayed_entries.pop(_k, None)

        # ── Step 5: Analyse pairs ─────────────────────────────
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

            # Check if signal engine queued a delayed entry (sniper rejected good signal)
            if signal is None and getattr(self._sig, "_delayed_sym", None) == sym:
                self._sig._delayed_sym = None
                # Build lightweight delayed entry object using last ticker data
                if sym not in self._delayed_entries:
                    # Re-analyze to get the signal object (sniper check bypassed)
                    try:
                        _ticker_copy = dict(ticker)
                        _prev_check = self._sig._sniper_score_1m
                        self._sig._sniper_score_1m = lambda *a, **k: 0.6   # force pass
                        _delayed_sig = self._sig.analyze(_ticker_copy, btc)
                        self._sig._sniper_score_1m = _prev_check
                        if _delayed_sig:
                            self._delayed_entries[sym] = {
                                "signal": _delayed_sig, "retries": 0, "added_at": now}
                            log.info("  ⏰ %s queued for delayed entry (3 retries)", sym)
                    except Exception: pass

            if signal:
                # Apply pattern-learning confidence boost
                boost = self._learner.get_confidence_boost(
                    signal.grade.split()[0], signal.btc_score, sym
                )
                if boost != 0:
                    signal = self._apply_boost(signal, boost)
                    log.info("Pattern boost %+d%% applied to %s", boost, sym)

                # ── PRE-TRADE SIMULATION GATE ─────────────────────────
                # Simulate the trade on recent candle history before queuing.
                # Blocks false signals like the ones that caused -5.46% losses.
                try:
                    # Context-aware candle window: fewer candles during capitulation
                    # prevents old uptrend data killing valid reversal sim trades
                    btc_s = self._btc.calculate() if hasattr(self._btc, "calculate") else None
                    _btc_rsi = getattr(btc_s, "rsi", 50) if btc_s else 50
                    _sim_candles = 50 if _btc_rsi < 32 else 150   # 32 catches BTC RSI=28-31
                    df_5m_sim = self._bin.get_klines(sym, "5m", _sim_candles)
                    sim_result = self._sim.simulate(signal, df_5m_sim,
                                                    label="PRE-TRADE SIM")

                    if not sim_result.approved:
                        log.warning(
                            "🚫 SIM BLOCKED %s %s | WR=%.0f%% E=%+.2f%% | %s",
                            signal.signal, sym, sim_result.win_rate * 100,
                            sim_result.expectancy * 100, sim_result.reason,
                        )
                        # Store sim result on signal for dashboard visibility
                        signal.factors = signal.factors + [
                            f"🚫 Sim blocked: {sim_result.reason}",
                            f"Sim WR={sim_result.win_rate:.0%} ({sim_result.wins}/{sim_result.n_trades}) vP&L=${sim_result.virtual_pnl:+.2f}",
                        ]
                        # Queue for delayed retry — maybe next cycle is better
                        if sym not in self._delayed_entries:
                            self._delayed_entries[sym] = {
                                "signal": signal, "retries": 0, "added_at": now}
                            log.info("  ⏰ %s queued for delayed retry after sim block", sym)
                        continue   # skip this signal — do NOT add to collected

                    # Sim approved — attach result to signal for logging/display
                    signal.factors = signal.factors + [
                        f"✅ Sim: WR={sim_result.win_rate:.0%} ({sim_result.wins}/{sim_result.n_trades}) E={sim_result.expectancy:+.2%}",
                    ]
                    log.info(
                        "✅ SIM APPROVED %s %s | WR=%.0f%% (%d/%d) E=%+.2f%% vP&L=$%+.2f",
                        signal.signal, sym,
                        sim_result.win_rate * 100, sim_result.wins, sim_result.n_trades,
                        sim_result.expectancy * 100, sim_result.virtual_pnl,
                    )
                except Exception as _sim_err:
                    log.warning("Sim error %s: %s — allowing trade", sym, _sim_err)
                # ── END SIM GATE ──────────────────────────────────────

                collected.append(signal)
                log.info(
                    "Queued: %s %s — %s (%d%%) confluence=%.1f",
                    signal.signal, sym, signal.grade, signal.confidence, signal.confluence,
                )

            time.sleep(self._cfg.alert.binance_rate_limit_seconds)

        log.info("Scan done — %d pairs, %d signals queued", len(gainers), len(collected))
        if not collected:
            import time as _t_fb
            _now_ts = _t_fb.time()
            if not hasattr(self, "_last_signal_ts"):
                self._last_signal_ts = _now_ts
            _dry_min = (_now_ts - self._last_signal_ts) / 60
            if _dry_min >= 20:
                log.info("  ⚠️  Dry spell %.0f min — Grade B mode: Choppy_Range threshold relaxed by 7pts", _dry_min)
                if not hasattr(self, "_grade_b_mode"):
                    self._grade_b_mode = True
            log.info("  ℹ️  No signals this cycle — all pairs scored below threshold or rejected")
        else:
            self._last_signal_ts = __import__("time").time()
            self._grade_b_mode = False

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

        # Sort by confluence — no hard cap, allow all quality signals
        collected = sorted(collected, key=lambda s: s.confluence, reverse=True)[:10]

        # Always send combined report every scan — positions + new signals
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
        self._run_auto_trade(collected, spot_signals, now)

        # ── Protect unprotected futures positions (every cycle) ───
        try:
            from dashboard.models import AutoTradeState, SignalRecord
            _state = AutoTradeState.get()
            if _state.futures_enabled:
                from datetime import timedelta
                week_ago = now - timedelta(days=7)
                _sig_lookup = {}
                for _rec in SignalRecord.objects.filter(
                    created_at__gte=week_ago, outcome="PENDING"
                ).order_by("-created_at"):
                    if _rec.symbol not in _sig_lookup:
                        class _S: pass
                        _s = _S()
                        _s.tp1=_rec.tp1; _s.tp2=_rec.tp2
                        _s.tp3=_rec.tp3; _s.sl=_rec.sl
                        _sig_lookup[_rec.symbol] = _s
                _fut_t = self._get_trader_for_mode(_state, "futures")
                if _fut_t:
                    _protected = _fut_t.protect_open_positions(_sig_lookup)
                    if _protected:
                        log.info("Protected %d positions: %s", len(_protected), _protected)
        except Exception as _pe:
            log.warning("protect_positions error: %s", _pe)

        # ── ORPHAN ORDER CLEANUP ──────────────────────────────────
        # If orders exist for a symbol that has NO open position,
        # the position was closed (TP/SL/trailing hit) but orders remain.
        # Cancel them and auto-update the signal outcome.
        try:
            from src.trading.binance_trader import BinanceTrader as _BT2
            from config import load_config as _lc3
            from dashboard.models import AutoTradeState as _ATS2, SignalRecord
            _st2 = _ATS2.get()
            if _st2.futures_enabled:
                _cfg3 = _lc3()
                if _cfg3.auto.has_keys:
                    _ft2 = _BT2(_cfg3.auto.api_key, _cfg3.auto.api_secret,
                                 mode="futures", live=not _cfg3.auto.testnet)
                    _open_positions  = {p["symbol"] for p in _ft2.get_positions()}
                    _all_orders      = _ft2.get_open_orders()
                    _syms_with_orders = {o.get("symbol","") for o in _all_orders}
                    _orphan_syms = _syms_with_orders - _open_positions - {""}
                    for _osym in _orphan_syms:
                        # Skip symbols recently traded (grace period 90s)
                        from dashboard.models import ScalpPosition as _SPG
                        _sp_ts = _SPG.objects.filter(symbol=_osym).order_by("-opened_at").first()
                        if _sp_ts and _sp_ts.opened_at:
                            import datetime as _dtg
                            _age_s = (_dtg.datetime.now(_dtg.timezone.utc) - _sp_ts.opened_at.astimezone(_dtg.timezone.utc)).total_seconds()
                            if _age_s < 90:
                                log.debug("Orphan grace: %s opened %.0fs ago — skipping", _osym, _age_s)
                                continue
                        log.info("🧹 Orphan orders for %s (position closed) — cancelling", _osym)
                        _ft2._req("DELETE", "/fapi/v1/allOpenOrders", {"symbol": _osym})
                        try:
                            _algos2 = _ft2._req("GET", "/fapi/v1/openAlgoOrders",
                                                {"symbol": _osym}) or {}
                            for _ao2 in (_algos2.get("algoOrders",[])
                                         if isinstance(_algos2,dict) else []):
                                _aid2 = _ao2.get("algoId") or _ao2.get("orderId")
                                if _aid2:
                                    _ft2._req("DELETE","/fapi/v1/algoOrder",
                                              {"symbol":_osym,"algoId":_aid2})
                        except Exception: pass
                        # Auto-update pending signal outcome → TP1 (position closed with profit)
                        from zoneinfo import ZoneInfo
                        _EAT2 = ZoneInfo("Africa/Dar_es_Salaam")
                        _today2 = now.astimezone(_EAT2).replace(
                            hour=0, minute=0, second=0, microsecond=0)
                        _pending = SignalRecord.objects.filter(
                            symbol=_osym, outcome="PENDING",
                            created_at__gte=_today2,
                            notes__icontains="AUTO_FUT:YES"
                        ).order_by("-created_at").first()
                        if _pending:
                            # Mark as TP1 hit (most likely — position closed in profit)
                            _pending.outcome = "TP1"
                            _pending.save(update_fields=["outcome"])
                            log.info("  Signal %s outcome → TP1 (position closed)", _osym)
                        # Close ScalpPosition record
                        try:
                            from dashboard.models import ScalpPosition
                            ScalpPosition.objects.filter(
                                symbol=_osym, closed=False
                            ).update(closed=True, close_reason="AUTO")
                        except Exception: pass
        except Exception as _oe:
            log.debug("Orphan cleanup: %s", _oe)

        # ── GRACE PERIOD: skip positions opened in last 90 seconds ──
        # Prevents protect_loop and orphan_cleanup from interfering with
        # freshly placed TP/SL orders that haven't propagated to Binance yet
        import time as _time
        _grace_cutoff_ts = _time.time() - 90  # 90 second grace window

        # Time-exit removed — SL/TP handle all exits


        # ── Auto-check outcomes for pending signals ──────────────
        try:
            from dashboard.management.commands.check_outcomes import (
                get_candles_5m, determine_outcome_from_candles,
                get_actual_trades, determine_outcome_from_trades
            )
            from dashboard.models import SignalRecord as _SR
            import os as _os
            _api_key    = _os.environ.get("BINANCE_API_KEY","")
            _api_secret = _os.environ.get("BINANCE_API_SECRET","")
            from datetime import timedelta as _td3
            _EAT3 = ZoneInfo("Africa/Dar_es_Salaam")
            _cutoff3 = now - _td3(hours=2)
            _pending3 = _SR.objects.filter(
                outcome="PENDING", created_at__gte=_cutoff3)
            for _sig3 in _pending3:
                try:
                    _since3 = int(_sig3.created_at.timestamp() * 1000)
                    _trades3 = get_actual_trades(
                        _sig3.symbol, _since3, _api_key, _api_secret)
                    if _trades3:
                        _out3, _cp3, _pnl3 = determine_outcome_from_trades(
                            _sig3, _trades3)
                    else:
                        _df3 = get_candles_5m(_sig3.symbol, _since3, 60)
                        _out3, _cp3 = determine_outcome_from_candles(_sig3, _df3)
                        if _cp3 and _sig3.entry_price:
                            _pnl3 = round(
                                (_cp3 - _sig3.entry_price) / _sig3.entry_price * 100
                                * (1 if _sig3.signal == "BUY" else -1), 2)
                        else:
                            _pnl3 = 0.0
                    if _out3 != "PENDING":
                        _sig3.outcome    = _out3
                        _sig3.profit_pct = _pnl3
                        _sig3.save(update_fields=["outcome","profit_pct"])
                        log.info("Outcome: %s %s → %s (%+.2f%%)",
                                 _sig3.symbol, _sig3.signal, _out3, _pnl3)
                except Exception as _se3:
                    log.debug("Outcome check %s: %s", _sig3.symbol, _se3)
        except Exception as _oe3:
            log.debug("Auto outcome check: %s", _oe3)



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

    def _get_trader_for_mode(self, state, mode: str):
        try:
            api_key    = self._cfg.auto.api_key
            api_secret = self._cfg.auto.api_secret
            if not api_key or not api_secret:
                log.warning('Auto-trade: no API keys in .env')
                return None
            from src.trading.binance_trader import BinanceTrader
            risk  = state.spot_risk        if mode == 'spot' else state.futures_risk
            max_t = state.spot_max_trades   if mode == 'spot' else state.futures_max_trades
            return BinanceTrader(
                api_key    = api_key,
                api_secret = api_secret,
                mode       = mode,
                live       = not self._cfg.auto.testnet,
                risk_pct   = risk,
                daily_loss_limit_pct = self._cfg.auto.daily_loss_limit_pct,
                max_trades_per_day   = max_t,
            )
        except Exception as e:
            log.error('Trader init error (%s): %s', mode, e)
            return None

    def _get_trader(self, state):
        mode = getattr(state, 'mode', 'spot')
        return self._get_trader_for_mode(state, mode)

    def _run_auto_trade(self, new_signals: list, spot_signals: list, now):

        """
        Independent auto-trade for Spot and Futures.
        - spot_enabled: BUY signals only → Market BUY + OCO + TP2/TP3
        - futures_enabled: BUY (long) + SELL (short) → Futures with leverage
        - Checks both new signals this cycle AND today's untraded PENDING DB signals
        - SpotSignal records (notes starts with SPOT) also checked for spot
        """
        try:
            from dashboard.models import AutoTradeState, SignalRecord
            state = AutoTradeState.get()

            spot_on    = state.spot_enabled
            fut_on     = state.futures_enabled
            if not spot_on and not fut_on:
                return

            # Build traders
            spot_trader = self._get_trader_for_mode(state, "spot")    if spot_on    else None
            fut_trader  = self._get_trader_for_mode(state, "futures") if fut_on     else None

            # Connectivity pre-check — log clearly if API is unreachable
            def _check_api(url):
                try:
                    import requests as _rq
                    r = _rq.get(url, timeout=4)
                    return r.status_code == 200
                except Exception:
                    return False

            spot_api_ok = _check_api("https://api.binance.com/api/v3/ping")
            fut_api_ok  = _check_api("https://fapi.binance.com/fapi/v1/ping")

            if spot_on and not spot_api_ok:
                log.warning("AUTO-TRADE BLOCKED: Binance Spot API (api.binance.com) is unreachable from this network. Use VPN or deploy to a server.")
                spot_on = False  # disable for this cycle
                spot_trader = None

            if fut_on and not fut_api_ok:
                log.warning("AUTO-TRADE BLOCKED: Binance Futures API (fapi.binance.com) is unreachable from this network. Use VPN or deploy to a server.")
                fut_on = False
                fut_trader = None

            if not spot_on and not fut_on:
                log.warning("Both APIs unreachable — skipping auto-trade this cycle")
                return

            # Sync daily trade counters from DB (single source of truth)
            from zoneinfo import ZoneInfo as _ZI
            from datetime import timedelta as _td
            _eat      = now.astimezone(_ZI("Africa/Dar_es_Salaam"))
            _t_start  = _eat.replace(hour=0, minute=0, second=0, microsecond=0)
            _t_end    = _t_start + _td(days=1)
            _spot_cnt = SignalRecord.objects.filter(
                created_at__gte=_t_start, created_at__lt=_t_end,
                notes__icontains="AUTO_SPOT:YES").count()
            _fut_cnt  = SignalRecord.objects.filter(
                created_at__gte=_t_start, created_at__lt=_t_end,
                notes__icontains="AUTO_FUT:YES").count()
            if state.spot_trades_today != _spot_cnt or state.futures_trades_today != _fut_cnt:
                state.spot_trades_today    = _spot_cnt
                state.futures_trades_today = _fut_cnt
                state.save(update_fields=["spot_trades_today","futures_trades_today"])
                log.info("Counters synced → spot=%d fut=%d (today only)", _spot_cnt, _fut_cnt)

            spot_bal = spot_trader.get_available_balance()  if spot_trader else 0.0
            fut_bal  = fut_trader.get_available_balance()   if fut_trader  else 0.0
            log.info("Auto-trade | Spot=%s $%.2f (api=%s) | Futures=%s $%.2f (api=%s) | new=%d",
                     "ON" if spot_on else "OFF", spot_bal, "✅" if spot_api_ok else "❌",
                     "ON" if fut_on  else "OFF", fut_bal,  "✅" if fut_api_ok  else "❌",
                     len(new_signals))

            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

            # --- Build candidates list ---
            # Futures: ALL signals from signal_engine (BUY long + SELL short)
            fut_candidates_new = list(new_signals)
            # Spot: ONLY spot_signals from SpotSignalEngine (BUY only, no short)
            # Do NOT send futures signals to spot trader — they have different TP/SL logic
            spot_candidates_new = list(spot_signals) if spot_signals else []

            # Pending DB futures signals not yet auto-traded
            pending_fut = list(SignalRecord.objects.filter(
                created_at__gte=today_start, outcome="PENDING"
            ).exclude(notes__icontains="AUTO_FUT:YES"))

            # Pending DB spot signals — only those explicitly marked as SPOT
            pending_spot = list(SignalRecord.objects.filter(
                created_at__gte=today_start, outcome="PENDING",
                signal="BUY", notes__icontains="SPOT"
            ).exclude(notes__icontains="AUTO_SPOT:YES"))

            def _sig_from_rec(rec):
                class _S: pass
                s = _S()
                s.symbol=rec.symbol; s.signal=rec.signal; s.grade=rec.grade
                # Use ORIGINAL entry_price — TP/SL levels are relative to it
                s.price = rec.entry_price
                s.tp1=rec.tp1; s.tp2=rec.tp2
                s.tp3=rec.tp3; s.sl=rec.sl
                s.btc_score=rec.btc_score; s.confidence=rec.confidence
                # Staleness checks — skip stale pending signals
                try:
                    import requests as _rq
                    r = _rq.get("https://api.binance.com/api/v3/ticker/price",
                                params={"symbol": rec.symbol}, timeout=4)
                    if r.ok:
                        cur = float(r.json().get("price", rec.entry_price))
                        entry = rec.entry_price or cur

                        # 1. Price already past SL → trade would immediately lose
                        if rec.signal == "BUY"  and cur <= rec.sl:
                            log.info("Skip %s BUY — price $%.5g already below SL $%.5g", rec.symbol, cur, rec.sl)
                            return None
                        if rec.signal == "SELL" and cur >= rec.sl:
                            log.info("Skip %s SELL — price $%.5g already above SL $%.5g", rec.symbol, cur, rec.sl)
                            return None

                        # 2. Price already past TP1 → trade opens and closes instantly
                        if rec.signal == "BUY"  and rec.tp1 > 0 and cur >= rec.tp1:
                            log.info("Skip %s BUY — price $%.5g already past TP1 $%.5g", rec.symbol, cur, rec.tp1)
                            return None
                        if rec.signal == "SELL" and rec.tp1 > 0 and cur <= rec.tp1:
                            log.info("Skip %s SELL — price $%.5g already past TP1 $%.5g", rec.symbol, cur, rec.tp1)
                            return None

                        # 3. Entry divergence > 3% → signal is stale, conditions changed
                        if entry > 0:
                            divergence = abs(cur - entry) / entry * 100
                            if divergence > 3.0:
                                log.info("Skip %s — entry divergence %.1f%% too large (stale signal)", rec.symbol, divergence)
                                return None
                except Exception:
                    pass
                return s

            # --- Execute SPOT trades ---
            if spot_on and spot_trader:
                seen_spot = set()
                all_spot  = list(spot_candidates_new) + [_sig_from_rec(r) for r in pending_spot]
                log.info("Spot candidates: %d (new=%d pending=%d)", len(all_spot), len(spot_candidates_new), len(pending_spot))
                for sig in all_spot:
                    if sig is None: continue
                    grade_key = sig.grade.split()[0]
                    if grade_key not in ("ULTRA","STRONG","STANDARD"):
                        log.debug("Spot skip %s: grade=%s", sig.symbol, grade_key)
                        continue
                    if state.spot_trades_today >= state.spot_max_trades:
                        log.warning("Spot daily limit %d reached", state.spot_max_trades)
                        break
                    if sig.signal != "BUY":
                        log.debug("Spot skip %s: signal=%s (not BUY)", sig.symbol, sig.signal)
                        continue
                    if sig.symbol in seen_spot:
                        continue
                    seen_spot.add(sig.symbol)

                    result = spot_trader.execute_signal(sig, spot_bal)
                    # Mark DB record
                    rec_qs = SignalRecord.objects.filter(
                        symbol=sig.symbol, created_at__gte=today_start, outcome="PENDING"
                    ).first()
                    if rec_qs:
                        mark = f"AUTO_SPOT:{'YES' if result.success else 'FAIL'}"
                        if result.success:
                            mark += f" oco={result.oco_id} sl={result.sl_order_id}"
                        else:
                            mark += f" {result.error[:40]}"
                        rec_qs.notes = (rec_qs.notes or "") + " | " + mark
                        rec_qs.save(update_fields=["notes"])

                    if result.success:
                        spot_bal -= result.qty * result.entry_price
                        state.spot_trades_today += 1
                        state.spot_total += 1
                        state.save(update_fields=["spot_trades_today","spot_total"])
                        log.info("SPOT TRADE ✅ %s %s @ %.6g | oco=%s sl=%s tp1=%s tp2=%s tp3=%s",
                                 result.side, sig.symbol, result.entry_price,
                                 result.oco_id, result.sl_order_id,
                                 sig.tp1, sig.tp2, sig.tp3)
                    else:
                        log.warning("SPOT TRADE ❌ %s %s — %s",
                                    sig.signal, sig.symbol, result.error)

            # --- Execute FUTURES trades ---
            if fut_on and fut_trader:
                seen_fut = set()
                trades_this_cycle = 0          # max 6 new trades per cycle
                MAX_PER_CYCLE     = 2          # prevents 3 simultaneous losses
                # Expire pending signals older than 30 minutes
                import datetime as _dt_exp
                _now_exp = _dt_exp.datetime.now(tz=_dt_exp.timezone.utc)
                _fresh_pending_fut = []
                for _r in pending_fut:
                    _ca = _r.created_at
                    if _ca.tzinfo is None:
                        _ca = _ca.replace(tzinfo=_dt_exp.timezone.utc)
                    _age_min = (_now_exp - _ca).total_seconds() / 60
                    if _age_min > 30:
                        _r.outcome = "EXPIRED"
                        _r.save(update_fields=["outcome"])
                        log.info("  ⏰ EXPIRED stale pending %s (%s) — age=%.0f min",
                                 _r.symbol, _r.signal, _age_min)
                    else:
                        _fresh_pending_fut.append(_r)
                all_fut  = list(fut_candidates_new) + [_sig_from_rec(r) for r in _fresh_pending_fut]
                log.info("Futures candidates: %d (new=%d pending=%d bal=$%.2f)", len(all_fut), len(fut_candidates_new), len(_fresh_pending_fut), fut_bal)
                for sig in all_fut:
                    if sig is None: continue
                    grade_key = sig.grade.split()[0]
                    if grade_key not in ("ULTRA","STRONG","STANDARD"):
                        log.debug("Futures skip %s: grade=%s", sig.symbol, grade_key)
                        continue
                    if trades_this_cycle >= MAX_PER_CYCLE:
                        log.info("Cycle trade limit (%d) reached — deferring %s to next cycle",
                                 MAX_PER_CYCLE, sig.symbol)
                        break
                    try:
                        from src.trading.binance_trader import BinanceTrader as _BT
                        from config import load_config as _lc2
                        _c2 = _lc2()
                        if _c2.auto.has_keys:
                            _tmp = _BT(_c2.auto.api_key, _c2.auto.api_secret,
                                       mode="futures", live=not _c2.auto.testnet)
                            _open_count = len(_tmp.get_positions())
                            if _open_count >= 6:
                                log.info("Max 6 positions cap reached (%d) — skip %s",
                                         _open_count, sig.symbol)
                                continue
                    except Exception:
                        pass
                    if sig.symbol in seen_fut:
                        continue
                    seen_fut.add(sig.symbol)

                    # Cooldown check — skip pair for 10 min after a loss (Improvement 3.4)
                    import datetime as _dtc
                    _cool_until = self._pair_cooldowns.get(sig.symbol)
                    if _cool_until and _dtc.datetime.now() < _cool_until:
                        _rem = int((_cool_until - _dtc.datetime.now()).total_seconds() / 60)
                        log.info("  ⏸ %s in cooldown (%d min remaining) — skipping",
                                 sig.symbol, _rem)
                        continue

                    # Double-check simulation before real execution (full report)
                    try:
                        df_5m_exec = self._bin.get_klines(sig.symbol, "5m", 150)
                        _exec_sim  = self._sim.simulate(sig, df_5m_exec,
                                                        label="EXEC RE-SIM")
                        if not _exec_sim.approved:
                            log.warning(
                                "🚫 EXEC SIM BLOCKED %s | WR=%.0f%% E=%+.2f%% | %s",
                                sig.symbol, _exec_sim.win_rate*100,
                                _exec_sim.expectancy*100, _exec_sim.reason)
                            continue   # skip — don't execute
                        log.info("✅ EXEC SIM OK %s WR=%.0f%% E=%+.2f%% DD=%.1f%%",
                                 sig.symbol, _exec_sim.win_rate*100,
                                 _exec_sim.expectancy*100,
                                 _exec_sim.max_drawdown*100)
                    except Exception as _es:
                        log.warning("Exec sim error %s: %s", sig.symbol, _es)

                    result = fut_trader.execute_signal(sig, fut_bal)
                    # Mark DB record
                    rec_qs = SignalRecord.objects.filter(
                        symbol=sig.symbol, created_at__gte=today_start, outcome="PENDING"
                    ).first()
                    if rec_qs:
                        mark = f"AUTO_FUT:{'YES' if result.success else 'FAIL'}"
                        if result.success:
                            mark += f" sl={result.sl_order_id} tp1={result.tp1_order_id}"
                        else:
                            mark += f" {result.error[:40]}"
                        rec_qs.notes = (rec_qs.notes or "") + " | " + mark
                        rec_qs.save(update_fields=["notes"])

                    if result.success:
                        fut_bal -= result.qty * result.entry_price
                        state.futures_trades_today += 1
                        state.futures_total += 1
                        state.save(update_fields=["futures_trades_today","futures_total"])
                        log.info("FUTURES TRADE ✅ %s %s @ %.6g | sl=%s tp1=%s tp2=%s tp3=%s",
                                 result.side, sig.symbol, result.entry_price,
                                 result.sl_order_id, sig.tp1, sig.tp2, sig.tp3)
                        # Record open time for scalp time-exit
                        try:
                            from dashboard.models import ScalpPosition
                            _pos_side = "LONG" if result.side == "BUY" else "SHORT"
                            ScalpPosition.open(sig.symbol, _pos_side,
                                               result.qty, result.entry_price)
                        except Exception: pass
                    else:
                        log.warning("FUTURES TRADE ❌ %s %s — %s",
                                    sig.signal, sig.symbol, result.error)
                        # If failed due to balance/size — mark DB record as FAILED
                        # so it stops being re-queued every 2 minutes indefinitely
                        if result.error and ("too low" in result.error or "rounds to zero" in result.error):
                            _fail_rec = SignalRecord.objects.filter(
                                symbol=sig.symbol, outcome="PENDING"
                            ).first()
                            if _fail_rec:
                                _fail_rec.outcome = "FAIL"
                                _fail_rec.notes = (_fail_rec.notes or "") + f" | AUTO_FAIL: {result.error[:60]}"
                                _fail_rec.save(update_fields=["outcome", "notes"])
                                log.info("  📌 %s marked FAIL in DB (balance too low — removed from retry queue)", sig.symbol)

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
            5 * 60      # minimum 5 minutes for scalp trade before staleness check
        )
        last = self._cooldowns.get(symbol)
        if last is None:
            return False
        return (now - last).total_seconds() < COOLDOWN_SECONDS