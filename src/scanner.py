"""
src/scanner.py — v5
Full automated scan loop with structure‑first signal engine integration,
candidate ranking, adaptive threshold, and auto‑trade execution.
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
from src.analysis.signal_engine import Signal, get_signal_engine
from src.analysis.spot_signal_engine import SpotSignalEngine
from src.trading.binance_trader import BinanceTrader
from src.data.binance_client import BinanceClient
from src.utils.formatter import  fmt_digest
from src.utils.logger import get_logger
from src.analysis.signal_simulator import get_simulator
from src.analysis.scalping_engine import ScalpingEngine

log = get_logger(__name__)

_GRADE_ORDER    = {"ULTRA": 0, "STRONG": 1, "STANDARD": 2}
_BINANCE_KLINES = "https://api.binance.com/api/v3/klines"


# -----------------------------------------------------------------------------
# PatternLearner (unchanged)
# -----------------------------------------------------------------------------
class PatternLearner:
    def __init__(self):
        self._cache: dict = {}
        self._refreshed_at = None

    def get_confidence_boost(self, grade: str, btc_score: int, symbol: str) -> int:
        self._refresh_if_stale()
        boost = 0
        grade_wr = self._cache.get(f"grade_{grade}", 50)
        if grade_wr >= 75:   boost += 5
        elif grade_wr >= 65: boost += 3
        elif grade_wr <= 40: boost -= 5
        elif grade_wr <= 50: boost -= 3
        btc_range = self._btc_range(btc_score)
        range_wr  = self._cache.get(f"btc_{btc_range}", 50)
        if range_wr >= 70:   boost += 3
        elif range_wr <= 40: boost -= 3
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
            for grade in ("ULTRA", "STRONG", "STANDARD"):
                qs    = SignalRecord.objects.filter(grade=grade, created_at__gte=since).exclude(outcome="PENDING")
                total = qs.count()
                wins  = qs.filter(outcome__in=["TP1","TP2","TP3"]).count()
                if total >= 5:
                    cache[f"grade_{grade}"] = round(wins / total * 100)
            for rng in ("high","mid","low"):
                if rng == "high":   qs = SignalRecord.objects.filter(btc_score__gte=60, created_at__gte=since)
                elif rng == "mid":  qs = SignalRecord.objects.filter(btc_score__gte=40, btc_score__lt=60, created_at__gte=since)
                else:               qs = SignalRecord.objects.filter(btc_score__lt=40, created_at__gte=since)
                qs    = qs.exclude(outcome="PENDING")
                total = qs.count()
                wins  = qs.filter(outcome__in=["TP1","TP2","TP3"]).count()
                if total >= 5:
                    cache[f"btc_{rng}"] = round(wins / total * 100)
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


# -----------------------------------------------------------------------------
# OutcomeChecker (unchanged)
# -----------------------------------------------------------------------------
class OutcomeChecker:
    def check_pending(self) -> int:
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
                    log.info("Auto-outcome: %s %s → %s (%+.2f%%)", sig.signal, sig.symbol, outcome, sig.profit_pct or 0)
            return count
        except Exception as e:
            log.warning("OutcomeChecker error: %s", e)
            return 0

    def _determine(self, sig, since_ms: int) -> tuple[str, float | None]:
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
        best_tp    = None
        trail_sl   = sig.sl
        for _, row in df.iterrows():
            h, l = float(row["high"]), float(row["low"])
            if is_sell:
                if trail_sl is not None and h >= trail_sl:
                    return best_tp or "SL", (best_tp and {"TP1": sig.tp1, "TP2": sig.tp2, "TP3": sig.tp3}.get(best_tp)) or sig.sl
                if l <= sig.tp3:
                    return "TP3", sig.tp3
                if l <= sig.tp2:
                    best_tp  = "TP2"
                    trail_sl = sig.tp1
                elif l <= sig.tp1 and best_tp is None:
                    best_tp  = "TP1"
                    trail_sl = sig.entry_price
            else:
                if trail_sl is not None and l <= trail_sl:
                    return best_tp or "SL", (best_tp and {"TP1": sig.tp1, "TP2": sig.tp2, "TP3": sig.tp3}.get(best_tp)) or sig.sl
                if h >= sig.tp3:
                    return "TP3", sig.tp3
                if h >= sig.tp2:
                    best_tp  = "TP2"
                    trail_sl = sig.tp1
                elif h >= sig.tp1 and best_tp is None:
                    best_tp  = "TP1"
                    trail_sl = sig.entry_price
        return best_tp or "PENDING", ({"TP1": sig.tp1, "TP2": sig.tp2, "TP3": sig.tp3}.get(best_tp)) if best_tp else None

    @staticmethod
    def _calc_pnl(sig, close_price) -> float | None:
        if not close_price or not sig.entry_price:
            return None
        if sig.signal == "SELL":
            return round((sig.entry_price - close_price) / sig.entry_price * 100, 2)
        return round((close_price - sig.entry_price) / sig.entry_price * 100, 2)


# -----------------------------------------------------------------------------
# Scanner (v5 with candidate ranking & adaptive threshold)
# -----------------------------------------------------------------------------
class Scanner:
    def __init__(self, cfg: AppConfig):
        self._cfg     = cfg
        binance       = BinanceClient(cfg.binance, cfg.scan)
        self._btc     = BtcStrengthEngine(binance, cfg.scan)
        self._news    = NewsEngine(os.environ.get("CRYPTOPANIC_API_KEY", ""))
        self._sig     = get_signal_engine(binance)
        self._spot    = SpotSignalEngine(binance)
        self._trader: "BinanceTrader | None" = None
        self._trader_mode: str = ""
        self._wa      = WhatsAppSender(cfg.whatsapp)
        self._bin     = binance
        self._checker = OutcomeChecker()
        self._learner = PatternLearner()
        self._cooldowns: dict[str, datetime] = {}
        self._last_btc_update: datetime | None = None
        self._last_daily_report: datetime | None = None
        self._daily_scan_count: int = 0
        self._pending_symbols: set[str] = set()
        self._pair_cooldowns: dict = {}
        self._delayed_entries: dict = {}
        self._sim = get_simulator()
        # v5 new attributes
        self._adaptive_threshold = cfg.signal.adaptive_threshold_start
        self._last_signal_count = 0
        self._daily_loss_limit_hit = False
        self._last_loss_hit_date = None
        self._scalp_engine = ScalpingEngine(binance)

    # -------------------------------------------------------------------------
    # Main Cycle
    # -------------------------------------------------------------------------
    def run_cycle(self) -> None:
        from dashboard.models import ScanRecord, SignalRecord, NewsItem

        now = timezone.now()
        log.info("=" * 65)
        from zoneinfo import ZoneInfo
        _EAT = ZoneInfo("Africa/Dar_es_Salaam")
        now_eat = now.astimezone(_EAT)
        collected: list[Signal] = []
        spot_signals: list = []
        log.info("Scan cycle started at %s EAT", now_eat.strftime("%H:%M:%S"))

        # Reset daily loss limit if new day
        if self._daily_loss_limit_hit:
            if self._last_loss_hit_date and now_eat.date() > self._last_loss_hit_date:
                self._daily_loss_limit_hit = False
                log.info("🔄 New day – daily loss limit reset")
            else:
                log.warning("🛑 Daily loss limit hit – trading paused for the day")
                return

        # Daily loss circuit breaker
        try:
            from dashboard.models import SignalRecord
            from django.utils import timezone as dtz

            today_start_eat = now_eat.replace(hour=0, minute=0, second=0, microsecond=0)
            today_start_utc = today_start_eat.astimezone(dtz.utc)

            day_losses = SignalRecord.objects.filter(
                created_at__gte=today_start_utc, outcome="SL"
            ).values_list("profit_pct", flat=True)
            total_loss_pct = sum(abs(p or 0) for p in day_losses)

            if total_loss_pct >= self._cfg.auto.daily_loss_limit_pct:
                self._daily_loss_limit_hit = True
                self._last_loss_hit_date = now_eat.date()
                log.warning("🛑 CIRCUIT BREAKER: %.1f%% SL losses today – pausing", total_loss_pct)
                return
        except Exception as e:
            log.debug("Circuit breaker error: %s", e)

        # Auto-resolve pending outcomes
        resolved = self._checker.check_pending()
        if resolved:
            log.info("Auto-resolved %d pending signal outcomes", resolved)

        # Block symbols with pending or today's signals
        try:
            from django.utils import timezone as tz
            today_start = tz.now().replace(hour=0, minute=0, second=0, microsecond=0)
            self._pending_symbols = set(
                SignalRecord.objects
                .filter(created_at__gte=today_start)
                .values_list("symbol", flat=True)
            ) | set(
                SignalRecord.objects
                .filter(outcome="PENDING")
                .values_list("symbol", flat=True)
            )
        except Exception:
            self._pending_symbols = set()

        # Fetch news
        news_items = self._news.get_news(50)
        log.info("News: %d items available", len(news_items))
        for n in news_items:
            try:
                NewsItem.objects.get_or_create(
                    title=n["title"][:298], source=n["source"],
                    defaults={
                        "url": n.get("url",""), "sentiment": n.get("sentiment","neutral"),
                        "published": timezone.now(), "currencies": n.get("currencies",""),
                    }
                )
            except Exception:
                pass

        # BTC Strength
        btc = self._btc.calculate()


        # Get pairs to scan
        top_gainers  = self._bin.get_top_gainers(30)
        liquid_pairs = self._bin.get_trending_pairs(30)
        seen = set()
        gainers = []
        for t in top_gainers + liquid_pairs:
            sym = t["symbol"]
            if sym not in seen:
                seen.add(sym)
                gainers.append(t)

        _n_gainer  = sum(1 for t in gainers if float(t.get("priceChangePercent",0) or 0) >=  5.0)
        _n_loser   = sum(1 for t in gainers if float(t.get("priceChangePercent",0) or 0) <= -5.0)
        _n_neutral = len(gainers) - _n_gainer - _n_loser
        log.info("Pairs to scan: %d [🚀%d gainers | 🔻%d losers | ➖%d neutral] | BTC Score: %d/100",
                 len(gainers), _n_gainer, _n_loser, _n_neutral, btc.score)

        # Crash protection (BTC < 15)
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


        # Process delayed entries (unchanged, kept for compatibility)
        collected: list[Signal] = []
        spot_signals: list = []
        to_remove = []
        for _dsym, _de in list(self._delayed_entries.items()):
            _dsig    = _de["signal"]
            _retries = _de["retries"]
            _added   = _de["added_at"]
            age_min = (now - _added).total_seconds() / 60
            if _retries >= 3 or age_min > 10:
                to_remove.append(_dsym)
                continue
            try:
                _fresh = self._bin.get_ticker(_dsym)
                if not _fresh:
                    _de["retries"] += 1; continue
                _cur_p = float(_fresh.get("lastPrice", 0) or 0)
                _entry_p = float(_dsig.price)
                if _entry_p <= 0:
                    to_remove.append(_dsym); continue
                _drift = abs(_cur_p - _entry_p) / _entry_p * 100
                if _drift > 0.5:
                    to_remove.append(_dsym); continue
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
                    log.info("  ⏰ %s DELAYED ENTRY fired (retry %d)", _dsym, _retries+1)
                    collected.append(_dsig)
                    to_remove.append(_dsym)
                else:
                    _de["retries"] += 1
            except Exception:
                _de["retries"] += 1
        for _k in to_remove:
            self._delayed_entries.pop(_k, None)
        # ---------------------------------------------------------------------
        # SCALP ENGINE – Quick EMA crossover signals
        # ---------------------------------------------------------------------
        scalp_signals = []
        for ticker in gainers[:20]:  # limit to top 20 for speed
            sym = ticker["symbol"]
            if sym in self._pending_symbols:
                continue
            if self._is_in_cooldown(sym, now):
                continue
            scalp_signal = self._scalp_engine.analyze(ticker)
            if scalp_signal and scalp_signal.confidence >= 50:
                scalp_signals.append(scalp_signal)

        # ---------------------------------------------------------------------
        # v5 CANDIDATE COLLECTION WITH RANKING
        # ---------------------------------------------------------------------
        candidates = []
        for ticker in gainers:
            sym = ticker["symbol"]

            if self._is_in_cooldown(sym, now):
                continue
            if sym in self._pending_symbols:
                continue

            signal = self._sig.analyze(ticker, btc)

            if signal:
                # Pattern learner boost (optional)
                boost = self._learner.get_confidence_boost(
                    signal.grade.split()[0], signal.btc_score, sym
                )
                if boost != 0:
                    signal = self._apply_boost(signal, boost)

                # Pre-trade simulation gate
                try:
                    df_5m_sim = self._bin.get_klines(sym, "5m", 150)
                    sim_result = self._sim.simulate(signal, df_5m_sim, label="PRE-TRADE SIM")
                    if not sim_result.approved:
                        log.warning("🚫 SIM BLOCKED %s %s | WR=%.0f%% E=%+.2f%%",
                                    signal.signal, sym, sim_result.win_rate*100, sim_result.expectancy*100)
                        continue
                    signal.factors.append(f"✅ Sim WR={sim_result.win_rate:.0%} E={sim_result.expectancy:+.2%}")
                except Exception as e:
                    log.warning("Sim error %s: %s", sym, e)

                candidates.append(signal)

            time.sleep(self._cfg.alert.binance_rate_limit_seconds)

        log.info("Scan done — %d pairs, %d candidates", len(gainers), len(candidates))

        # Adaptive threshold adjustment
        if not candidates:
            self._adaptive_threshold = max(self._cfg.signal.adaptive_threshold_min,
                                           self._adaptive_threshold - 5)
            log.info("  ⚠️  No candidates — adaptive threshold lowered to %d", self._adaptive_threshold)
        else:
            self._adaptive_threshold = self._cfg.signal.adaptive_threshold_start
            log.info("  ✅ %d candidates found — threshold reset to %d", len(candidates), self._adaptive_threshold)

        # Update engine threshold
        self._sig.threshold = self._adaptive_threshold

        # Sort by confluence (score) descending and take top N
        candidates.sort(key=lambda s: s.confluence, reverse=True)
        max_candidates = self._cfg.signal.max_candidates_per_cycle
        selected = candidates[:max_candidates]

        # Double-check threshold (engine already filtered, but ensure)
        selected = [s for s in selected if s.confluence >= self._adaptive_threshold]

        if selected:
            # Save scan record
            scan_rec = ScanRecord.objects.create(
                pairs_scanned=len(gainers), signals_found=len(selected),
                signals_sent=0, btc_score=btc.score,
                btc_price=btc.price, btc_trend=btc.trend,
            )

            # Save to DB
            for s in selected:
                try:
                    SignalRecord.objects.create(
                        symbol=s.symbol, signal=s.signal,
                        grade=s.grade.split()[0], confidence=s.confidence,
                        confluence=s.confluence, entry_price=s.price,
                        tp1=s.tp1, tp2=s.tp2, tp3=s.tp3, sl=s.sl,
                        gain_24h=s.gain_24h, rsi=getattr(s, 'rsi_1h', 50),
                        btc_score=s.btc_score, btc_trend=s.btc_trend,
                        factors=json.dumps(s.factors),
                    )
                except Exception as e:
                    log.warning("DB save failed %s: %s", s.symbol, e)
                self._cooldowns[s.symbol] = now

            # Build daily summary
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
                        "symbol": sig_rec.symbol,
                        "signal": sig_rec.signal,
                        "grade": sig_rec.grade,
                        "outcome": sig_rec.outcome,
                        "pnl": sig_rec.profit_pct,
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
                        "symbol": sig_rec.symbol,
                        "signal": sig_rec.signal,
                        "grade": sig_rec.grade,
                        "entry_price": sig_rec.entry_price,
                        "tp1": sig_rec.tp1, "tp2": sig_rec.tp2, "tp3": sig_rec.tp3,
                        "sl": sig_rec.sl,
                    })
            except Exception as _e:
                log.debug("Daily data fetch error: %s", _e)
                closed_today, open_positions = [], []

            # Send WhatsApp digest
            import inspect as _inspect
            _fmt_params = _inspect.signature(fmt_digest).parameters
            _extra = {}
            if "closed_today" in _fmt_params:
                _extra["closed_today"]   = closed_today
                _extra["open_positions"] = open_positions
                _extra["scan_number"]    = self._daily_scan_count
            msg = fmt_digest(selected, btc.to_dict(), self._cfg.risk, news_items, **_extra)
            delivered = self._wa.send(msg)

            if delivered:
                self._last_btc_update = now
                scan_rec.signals_sent = len(selected)
                scan_rec.save()
                log.info("Digest delivered (%d signals)", len(selected))
            else:
                for s in selected:
                    self._cooldowns.pop(s.symbol, None)
                log.error("Digest failed — will retry next cycle")

        else:
            log.info("No signals this cycle")
            ScanRecord.objects.create(
                pairs_scanned=len(gainers), signals_found=0,
                signals_sent=0, btc_score=btc.score,
                btc_price=btc.price, btc_trend=btc.trend,
            )

        # Auto-trade execution
        self._run_auto_trade(selected, spot_signals, now)

        # Protection and cleanup (unchanged from original scanner)
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

        # Orphan order cleanup
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
                            _pending.outcome = "TP1"
                            _pending.save(update_fields=["outcome"])
                            log.info("  Signal %s outcome → TP1 (position closed)", _osym)
                        try:
                            from dashboard.models import ScalpPosition
                            ScalpPosition.objects.filter(
                                symbol=_osym, closed=False
                            ).update(closed=True, close_reason="AUTO")
                        except Exception: pass
        except Exception as _oe:
            log.debug("Orphan cleanup: %s", _oe)

        log.info("Cycle complete")
        self._daily_scan_count += 1

    # -------------------------------------------------------------------------
    # Auto-trade helpers (unchanged from original scanner)
    # -------------------------------------------------------------------------
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
        try:
            from dashboard.models import AutoTradeState, SignalRecord
            state = AutoTradeState.get()

            spot_on    = state.spot_enabled
            fut_on     = state.futures_enabled
            if not spot_on and not fut_on:
                return

            spot_trader = self._get_trader_for_mode(state, "spot")    if spot_on    else None
            fut_trader  = self._get_trader_for_mode(state, "futures") if fut_on     else None

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
                log.warning("AUTO-TRADE BLOCKED: Spot API unreachable")
                spot_on = False
                spot_trader = None
            if fut_on and not fut_api_ok:
                log.warning("AUTO-TRADE BLOCKED: Futures API unreachable")
                fut_on = False
                fut_trader = None

            if not spot_on and not fut_on:
                return

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

            spot_bal = spot_trader.get_available_balance()  if spot_trader else 0.0
            fut_bal  = fut_trader.get_available_balance()   if fut_trader  else 0.0
            log.info("Auto-trade | Spot=%s $%.2f | Futures=%s $%.2f | new=%d",
                     "ON" if spot_on else "OFF", spot_bal,
                     "ON" if fut_on  else "OFF", fut_bal,
                     len(new_signals))

            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

            # Spot candidates
            spot_candidates_new = list(spot_signals) if spot_signals else []
            pending_spot = list(SignalRecord.objects.filter(
                created_at__gte=today_start, outcome="PENDING",
                signal="BUY", notes__icontains="SPOT"
            ).exclude(notes__icontains="AUTO_SPOT:YES"))

            def _sig_from_rec(rec):
                class _S: pass
                s = _S()
                s.symbol=rec.symbol; s.signal=rec.signal; s.grade=rec.grade
                s.price = rec.entry_price
                s.tp1=rec.tp1; s.tp2=rec.tp2
                s.tp3=rec.tp3; s.sl=rec.sl
                s.btc_score=rec.btc_score; s.confidence=rec.confidence
                try:
                    import requests as _rq
                    r = _rq.get("https://api.binance.com/api/v3/ticker/price",
                                params={"symbol": rec.symbol}, timeout=4)
                    if r.ok:
                        cur = float(r.json().get("price", rec.entry_price))
                        entry = rec.entry_price or cur
                        if rec.signal == "BUY"  and cur <= rec.sl:
                            return None
                        if rec.signal == "SELL" and cur >= rec.sl:
                            return None
                        if rec.signal == "BUY"  and rec.tp1 > 0 and cur >= rec.tp1:
                            return None
                        if rec.signal == "SELL" and rec.tp1 > 0 and cur <= rec.tp1:
                            return None
                        if entry > 0:
                            divergence = abs(cur - entry) / entry * 100
                            if divergence > 3.0:
                                return None
                except Exception:
                    pass
                return s

            # Execute SPOT
            if spot_on and spot_trader:
                seen_spot = set()
                all_spot  = list(spot_candidates_new) + [_sig_from_rec(r) for r in pending_spot]
                for sig in all_spot:
                    if sig is None: continue
                    grade_key = sig.grade.split()[0]
                    if grade_key not in ("ULTRA","STRONG","STANDARD"):
                        continue
                    if state.spot_trades_today >= state.spot_max_trades:
                        break
                    if sig.signal != "BUY":
                        continue
                    if sig.symbol in seen_spot:
                        continue
                    seen_spot.add(sig.symbol)
                    result = spot_trader.execute_signal(sig, spot_bal)
                    rec_qs = SignalRecord.objects.filter(
                        symbol=sig.symbol, created_at__gte=today_start, outcome="PENDING"
                    ).first()
                    if rec_qs:
                        mark = f"AUTO_SPOT:{'YES' if result.success else 'FAIL'}"
                        rec_qs.notes = (rec_qs.notes or "") + " | " + mark
                        rec_qs.save(update_fields=["notes"])
                    if result.success:
                        spot_bal -= result.qty * result.entry_price
                        state.spot_trades_today += 1
                        state.spot_total += 1
                        state.save(update_fields=["spot_trades_today","spot_total"])

            # Execute FUTURES
            if fut_on and fut_trader:
                seen_fut = set()
                trades_this_cycle = 0
                MAX_PER_CYCLE = 2
                import datetime as _dt_exp
                _now_exp = _dt_exp.datetime.now(tz=_dt_exp.timezone.utc)
                pending_fut = list(SignalRecord.objects.filter(
                    created_at__gte=today_start, outcome="PENDING"
                ).exclude(notes__icontains="AUTO_FUT:YES"))
                _fresh_pending_fut = []
                for _r in pending_fut:
                    _ca = _r.created_at
                    if _ca.tzinfo is None:
                        _ca = _ca.replace(tzinfo=_dt_exp.timezone.utc)
                    _age_min = (_now_exp - _ca).total_seconds() / 60
                    if _age_min > 30:
                        _r.outcome = "EXPIRED"
                        _r.save(update_fields=["outcome"])
                    else:
                        _fresh_pending_fut.append(_r)
                fut_candidates_new = list(new_signals)
                all_fut  = fut_candidates_new + [_sig_from_rec(r) for r in _fresh_pending_fut]
                for sig in all_fut:
                    if sig is None: continue
                    grade_key = sig.grade.split()[0]
                    if grade_key not in ("ULTRA","STRONG","STANDARD"):
                        continue
                    if trades_this_cycle >= MAX_PER_CYCLE:
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
                                continue
                    except Exception:
                        pass
                    if sig.symbol in seen_fut:
                        continue
                    seen_fut.add(sig.symbol)
                    import datetime as _dtc
                    _cool_until = self._pair_cooldowns.get(sig.symbol)
                    if _cool_until and _dtc.datetime.now() < _cool_until:
                        continue
                    try:
                        df_5m_exec = self._bin.get_klines(sig.symbol, "5m", 150)
                        _exec_sim  = self._sim.simulate(sig, df_5m_exec, label="EXEC RE-SIM")
                        if not _exec_sim.approved:
                            continue
                    except Exception:
                        pass
                    result = fut_trader.execute_signal(sig, fut_bal)
                    rec_qs = SignalRecord.objects.filter(
                        symbol=sig.symbol, created_at__gte=today_start, outcome="PENDING"
                    ).first()
                    if rec_qs:
                        mark = f"AUTO_FUT:{'YES' if result.success else 'FAIL'}"
                        rec_qs.notes = (rec_qs.notes or "") + " | " + mark
                        rec_qs.save(update_fields=["notes"])
                    if result.success:
                        fut_bal -= result.qty * result.entry_price
                        state.futures_trades_today += 1
                        state.futures_total += 1

                        state.save(update_fields=["futures_trades_today","futures_total"])
                        try:
                            from dashboard.models import ScalpPosition
                            _pos_side = "LONG" if result.side == "BUY" else "SHORT"
                            ScalpPosition.open(sig.symbol, _pos_side,
                                               result.qty, result.entry_price)
                        except Exception: pass
        except Exception as e:
            log.error("Auto-trade error: %s", e, exc_info=True)

    @staticmethod
    def _apply_boost(signal: Signal, boost: int) -> Signal:
        from dataclasses import replace
        new_conf = max(50, min(95, signal.confidence + boost))
        tag = f"✅ Pattern boost +{boost}%" if boost > 0 else f"⚠️ Pattern penalty {boost}%"
        new_factors = signal.factors + [tag]
        return replace(signal, confidence=new_conf, factors=new_factors)

    def _is_in_cooldown(self, symbol: str, now: datetime) -> bool:
        COOLDOWN_SECONDS = max(
            self._cfg.alert.cooldown_hours * 3600,
            5 * 60
        )
        last = self._cooldowns.get(symbol)
        if last is None:
            return False
        return (now - last).total_seconds() < COOLDOWN_SECONDS