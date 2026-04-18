"""
dashboard/views.py
───────────────────
All API endpoints for the dashboard.
"""

import json
import logging
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

from django.shortcuts import render
from django.utils import timezone
from django.db.models import Count, Avg, Sum, Q, F
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST, require_http_methods
from django.http import JsonResponse
from dashboard.models import AutoTradeState
from config import load_config
from .models import SignalRecord, ScanRecord, CapitalRecord

# ── Scanner reference (set by runall.py on startup) ───────────────────
_scanner_instance = None

def set_scanner(scanner):
    global _scanner_instance
    _scanner_instance = scanner


# ── Trade alerts (pushed from trader, polled by dashboard JS) ─────────
_TRADE_ALERTS = []

def push_trade_alert(level: str, message: str):
    import time
    _TRADE_ALERTS.append({"level": level, "msg": message, "ts": time.time()})
    if len(_TRADE_ALERTS) > 30:
        _TRADE_ALERTS.pop(0)

@require_http_methods(["GET"])
def auto_trade_alerts(request):
    import time
    now = time.time()
    recent = [a for a in _TRADE_ALERTS if now - a["ts"] < 120]
    _TRADE_ALERTS.clear()
    return JsonResponse({"alerts": recent})

logger = logging.getLogger(__name__)
_AT_CACHE: dict = {}   # cache for auto-trade status (avoids hammering Binance)
_EAT   = ZoneInfo("Africa/Dar_es_Salaam")


from django.views.decorators.csrf import ensure_csrf_cookie

@ensure_csrf_cookie
def dashboard(request):
    return render(request, "dashboard/index.html")


# ── Stats ─────────────────────────────────────────────────────

def api_stats(request):
    now = timezone.now()
    def ws(since):
        qs     = SignalRecord.objects.filter(created_at__gte=since)
        closed = qs.exclude(outcome="PENDING")
        total  = closed.count()
        wins   = closed.filter(outcome__in=["TP1","TP2","TP3"]).count()
        losses = closed.filter(outcome="SL").count()
        be     = closed.filter(outcome="BE").count()
        avg_p  = closed.filter(profit_pct__isnull=False).aggregate(a=Avg("profit_pct"))["a"]
        tot_p  = closed.filter(profit_pct__isnull=False).aggregate(s=Sum("profit_pct"))["s"]
        return {
            "total": total, "wins": wins, "losses": losses, "be": be,
            "pending": qs.filter(outcome="PENDING").count(),
            "win_rate": round(wins/total*100,1) if total else 0,
            "avg_profit": round(avg_p,2) if avg_p else 0,
            "total_profit": round(tot_p,2) if tot_p else 0,
        }
    return JsonResponse({
        "today":   ws(now.replace(hour=0,minute=0,second=0,microsecond=0)),
        "week":    ws(now - timedelta(days=7)),
        "month":   ws(now - timedelta(days=30)),
        "alltime": ws(now - timedelta(days=3650)),
    })


# ── Signals with pagination + filters ────────────────────────



def api_signals(request):
    page      = int(request.GET.get("page", 1))
    per_page  = int(request.GET.get("per_page", 20))
    direction = request.GET.get("direction", "")    # BUY / SELL
    grade     = request.GET.get("grade", "")        # ULTRA / STRONG / STANDARD
    symbol    = request.GET.get("symbol", "")
    outcome   = request.GET.get("outcome", "")
    date_from = request.GET.get("date_from", "")    # YYYY-MM-DD
    date_to   = request.GET.get("date_to", "")
    period    = request.GET.get("period", "")       # today / week / month

    qs = SignalRecord.objects.all()

    # Period shortcuts
    now = timezone.now()
    if period == "today":
        qs = qs.filter(created_at__gte=now.replace(hour=0, minute=0, second=0))
    elif period == "week":
        qs = qs.filter(created_at__gte=now - timedelta(days=7))
    elif period == "month":
        qs = qs.filter(created_at__gte=now - timedelta(days=30))
    else:
        # Default: today
        qs = qs.filter(created_at__gte=now.replace(hour=0, minute=0, second=0))

    if direction:
        qs = qs.filter(signal=direction.upper())
    if grade:
        qs = qs.filter(grade=grade.upper())
    if symbol:
        qs = qs.filter(symbol__icontains=symbol.upper())
    if outcome:
        qs = qs.filter(outcome=outcome.upper())
    if date_from:
        try:
            qs = qs.filter(created_at__date__gte=date_from)
        except Exception:
            pass
    if date_to:
        try:
            qs = qs.filter(created_at__date__lte=date_to)
        except Exception:
            pass

    total_count = qs.count()
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    offset = (page - 1) * per_page
    signals = qs[offset:offset + per_page]

    signals_data = []
    for sig in signals:
        local_time = timezone.localtime(sig.created_at)
        signals_data.append({
            "id": sig.id,
            "created_at": local_time.strftime("%Y-%m-%d %H:%M"),
            "symbol": sig.symbol,
            "signal": sig.signal,
            "grade": sig.grade,
            "confidence": sig.confidence,
            "entry": round(sig.entry_price, 6) if sig.entry_price else None,
            "tp1": round(sig.tp1, 6) if sig.tp1 else None,
            "tp2": round(sig.tp2, 6) if sig.tp2 else None,
            "tp3": round(sig.tp3, 6) if sig.tp3 else None,
            "sl": round(sig.sl, 6) if sig.sl else None,
            "rsi": round(sig.rsi, 1) if sig.rsi else None,
            "btc_score": sig.btc_score,
            "outcome": sig.outcome,
            "profit_pct": sig.profit_pct,
            "auto_checked": sig.auto_checked,
            "trigger_type": sig.trigger_type,          # v5 new
            # "score_breakdown": sig.score_breakdown,  # optional – can be added later
        })

    return JsonResponse({
        "signals": signals_data,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total_count,
            "total_pages": total_pages,
        },
    })


def _sig_dict(s: SignalRecord) -> dict:
    return {
        "id": s.id, "symbol": s.symbol, "signal": s.signal,
        "grade": s.grade, "confidence": s.confidence,
        "entry": s.entry_price, "tp1": s.tp1, "tp2": s.tp2, "tp3": s.tp3, "sl": s.sl,
        "gain_24h": s.gain_24h, "rsi": getattr(s, "rsi_1h", getattr(s, "rsi", 0)), "btc_score": s.btc_score,
        "outcome": s.outcome, "profit_pct": s.profit_pct,
        "auto_checked": s.auto_checked,
        "created_at": s.created_at.astimezone(_EAT).strftime("%d %b %H:%M EAT"),
        "created_date": s.created_at.astimezone(_EAT).strftime("%Y-%m-%d"),
    }


# ── Charts ─────────────────────────────────────────────────────

def api_chart_daily(request):
    now  = timezone.now()
    days = []
    for i in range(29, -1, -1):
        ds  = (now - timedelta(days=i)).replace(hour=0,minute=0,second=0,microsecond=0)
        de  = ds + timedelta(days=1)
        qs  = SignalRecord.objects.filter(created_at__gte=ds, created_at__lt=de)
        cl  = qs.exclude(outcome="PENDING")
        w   = cl.filter(outcome__in=["TP1","TP2","TP3"]).count()
        t   = cl.count()
        avg = cl.filter(profit_pct__isnull=False).aggregate(a=Avg("profit_pct"))["a"]
        days.append({
            "date": ds.astimezone(_EAT).strftime("%d %b"),
            "signals": qs.count(), "wins": w,
            "losses": cl.filter(outcome="SL").count(),
            "win_rate": round(w/t*100) if t else 0,
            "avg_profit": round(avg,2) if avg else 0,
        })
    return JsonResponse({"days": days})


def api_grades(request):
    result = []
    for g in ("ULTRA","STRONG","STANDARD"):
        qs    = SignalRecord.objects.filter(grade=g).exclude(outcome="PENDING")
        total = qs.count()
        wins  = qs.filter(outcome__in=["TP1","TP2","TP3"]).count()
        avg   = qs.filter(profit_pct__isnull=False).aggregate(a=Avg("profit_pct"))["a"]
        result.append({
            "grade": g, "total": total, "wins": wins,
            "win_rate": round(wins/total*100,1) if total else 0,
            "avg_profit": round(avg,2) if avg else 0,
        })
    return JsonResponse({"grades": result})


def api_top_pairs(request):
    pairs = (
        SignalRecord.objects.exclude(outcome="PENDING")
        .values("symbol")
        .annotate(
            total=Count("id"),
            wins=Count("id", filter=Q(outcome__in=["TP1","TP2","TP3"])),
            avg_profit=Avg("profit_pct"),
        )
        .order_by("-total")[:15]
    )
    result = []
    for p in pairs:
        p["win_rate"]   = round(p["wins"] / p["total"] * 100, 1) if p["total"] else 0
        p["avg_profit"] = round(p["avg_profit"], 2) if p["avg_profit"] else 0
        result.append(p)
    return JsonResponse({"pairs": result})


def api_scans(request):
    scans = ScanRecord.objects.all()[:48]
    return JsonResponse({"scans": [
        {"time": s.scanned_at.astimezone(_EAT).strftime("%d %b %H:%M EAT"),
         "pairs": s.pairs_scanned, "signals": s.signals_found,
         "btc": s.btc_score, "trend": s.btc_trend, "price": s.btc_price}
        for s in scans
    ]})


def api_outcome_distribution(request):
    dist = (
        SignalRecord.objects.values("outcome")
        .annotate(count=Count("id"))
        .order_by("-count")
    )
    return JsonResponse({"distribution": list(dist)})


# ── Capital tracking ──────────────────────────────────────────

def api_capital(request):
    records = CapitalRecord.objects.all()[:90]
    data    = [{"date": str(r.date), "capital": r.capital_usd, "notes": r.notes} for r in records]
    return JsonResponse({"records": data})


@csrf_exempt
def api_capital_add(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    d = json.loads(request.body)
    rec, created = CapitalRecord.objects.update_or_create(
        date=d.get("date", str(date.today())),
        defaults={"capital_usd": float(d["capital"]), "notes": d.get("notes","")},
    )
    return JsonResponse({"ok": True, "created": created, "capital": rec.capital_usd})


# ── Outcome update ────────────────────────────────────────────

@csrf_exempt
def api_update_outcome(request, signal_id):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    try:
        d       = json.loads(request.body)
        outcome = d.get("outcome","").upper()
        price   = d.get("close_price")
        s       = SignalRecord.objects.get(id=signal_id)
        s.outcome     = outcome
        s.close_price = float(price) if price else None
        if price and s.entry_price:
            p = float(price)
            if s.signal == "SELL":
                s.profit_pct = round((s.entry_price - p) / s.entry_price * 100, 2)
            else:
                s.profit_pct = round((p - s.entry_price) / s.entry_price * 100, 2)
        s.closed_at = timezone.now()
        s.save()
        return JsonResponse({"ok": True, "profit_pct": s.profit_pct})
    except SignalRecord.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)


# ── Report ────────────────────────────────────────────────────

def api_report(request):
    period = request.GET.get("period", "month")
    now    = timezone.now()
    since  = {
        "week":  now - timedelta(days=7),
        "month": now - timedelta(days=30),
        "year":  now - timedelta(days=365),
    }.get(period, now - timedelta(days=30))

    qs     = SignalRecord.objects.filter(created_at__gte=since)
    closed = qs.exclude(outcome="PENDING")
    total  = closed.count()
    wins   = closed.filter(outcome__in=["TP1","TP2","TP3"]).count()

    by_grade = {}
    for g in ("ULTRA","STRONG","STANDARD"):
        gq = closed.filter(grade=g)
        gt = gq.count()
        gw = gq.filter(outcome__in=["TP1","TP2","TP3"]).count()
        by_grade[g] = {
            "total": gt, "wins": gw,
            "win_rate": round(gw/gt*100,1) if gt else 0,
            "avg_profit": round(gq.filter(profit_pct__isnull=False).aggregate(a=Avg("profit_pct"))["a"] or 0, 2),
        }

    by_direction = {}
    for d in ("BUY","SELL"):
        dq = closed.filter(signal=d)
        dt = dq.count()
        dw = dq.filter(outcome__in=["TP1","TP2","TP3"]).count()
        by_direction[d] = {
            "total": dt, "wins": dw,
            "win_rate": round(dw/dt*100,1) if dt else 0,
        }

    top_pairs = list(
        closed.values("symbol")
        .annotate(total=Count("id"), wins=Count("id",filter=Q(outcome__in=["TP1","TP2","TP3"])))
        .order_by("-wins")[:5]
    )
    for p in top_pairs:
        p["win_rate"] = round(p["wins"]/p["total"]*100,1) if p["total"] else 0

    return JsonResponse({
        "period": period,
        "summary": {
            "total": total, "wins": wins,
            "losses": closed.filter(outcome="SL").count(),
            "win_rate": round(wins/total*100,1) if total else 0,
            "total_profit": round(closed.filter(profit_pct__isnull=False).aggregate(s=Sum("profit_pct"))["s"] or 0, 2),
            "avg_profit": round(closed.filter(profit_pct__isnull=False).aggregate(a=Avg("profit_pct"))["a"] or 0, 2),
        },
        "by_grade": by_grade,
        "by_direction": by_direction,
        "top_pairs": top_pairs,
    })


# ── Backtester ────────────────────────────────────────────────

def api_backtest_symbols(request):
    """Return list of top liquid symbols available for backtesting."""
    symbols = [
        "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT",
        "AVAXUSDT","DOTUSDT","LINKUSDT","MATICUSDT","NEARUSDT","APTUSDT",
        "ARBUSDT","OPUSDT","ATOMUSDT","LTCUSDT","DOGEUSDT","UNIUSDT",
        "AAVEUSDT","SHIBUSDT","TRXUSDT","XLMUSDT","ALGOUSDT","FILUSDT",
        "INJUSDT","SUIUSDT","SEIUSDT","TIAUSDT","WLDUSDT","PYTHUSDT",
    ]
    return JsonResponse({"symbols": symbols})


@csrf_exempt
def api_backtest_run(request):
    """
    Run a backtest. POST with:
      symbol  - e.g. "SOLUSDT"
      days    - lookback period (30/60/90/180)
      tp1_pct, tp2_pct, tp3_pct, sl_pct - risk settings
      min_confluence - signal filter
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    import threading
    from src.analysis.backtester import Backtester

    try:
        d = json.loads(request.body)
        symbol  = d.get("symbol", "BTCUSDT").upper()
        days    = int(d.get("days", 90))
        days    = max(30, min(365, days))

        bt = Backtester(
            tp1_pct       = float(d.get("tp1_pct", 3.0)),
            tp2_pct       = float(d.get("tp2_pct", 6.0)),
            tp3_pct       = float(d.get("tp3_pct", 10.0)),
            sl_pct        = float(d.get("sl_pct", 3.0)),
            min_confluence = float(d.get("min_confluence", 5.0)),
        )

        result = bt.run(symbol, days)

        # Serialize trades (convert datetime objects)
        trades_out = []
        for t in result.trades:
            t2 = dict(t)
            for k in ("entry_time", "exit_time"):
                if isinstance(t2.get(k), datetime):
                    t2[k] = t2[k].strftime("%d %b %H:%M")
            trades_out.append(t2)

        return JsonResponse({
            "symbol":        result.symbol,
            "period_days":   result.period_days,
            "total_trades":  result.total_trades,
            "wins":          result.wins,
            "losses":        result.losses,
            "timeouts":      result.timeouts,
            "win_rate":      result.win_rate,
            "avg_win_pct":   result.avg_win_pct,
            "avg_loss_pct":  result.avg_loss_pct,
            "total_pnl_pct": result.total_pnl_pct,
            "max_drawdown":  result.max_drawdown,
            "profit_factor": result.profit_factor,
            "best_trade":    result.best_trade,
            "worst_trade":   result.worst_trade,
            "ultra_wr":      result.ultra_wr,
            "strong_wr":     result.strong_wr,
            "equity_curve":  result.equity_curve,
            "trades":        trades_out,
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)



# ── Auto-trade API ────────────────────────────────────────────


@require_GET
def sim_stats(request):
    """GET: simulation gate statistics — block rate, approved/blocked counts."""
    try:
        from dashboard.models import SignalRecord
        from datetime import timedelta
        from django.utils import timezone
        since = timezone.now() - timedelta(days=7)
        sigs  = SignalRecord.objects.filter(created_at__gte=since)
        total = sigs.count()
        sim_approved = sigs.filter(factors__icontains="Sim: WR=").count()
        sim_blocked  = sigs.filter(factors__icontains="Sim blocked").count()
        block_rate   = round(sim_blocked / max(total, 1) * 100, 1)
        return JsonResponse({
            "total_signals":  total,
            "sim_approved":   sim_approved,
            "sim_blocked":    sim_blocked,
            "block_rate_pct": block_rate,
            "message": f"Sim blocked {sim_blocked} trades in last 7 days ({block_rate}% block rate)",
        })
    except Exception as e:
        return JsonResponse({"error": str(e)})


def ml_insights(request):
    """GET: ML removed in v4.2 — returns disabled status for dashboard compatibility."""
    return JsonResponse({
        "model": {
            "ready":    False,
            "message":  "ML removed — rule-based strategy + HTF cascade v4.2",
            "samples":  0,
            "accuracy": 0,
            "features": [],
        },
        "progress": {}
    })


@csrf_exempt
def auto_trade_reset_counters(request):
    """POST: Reset daily trade counters."""
    try:
        from dashboard.models import AutoTradeState
        state = AutoTradeState.get()
        state.spot_trades_today    = 0
        state.futures_trades_today = 0
        if hasattr(state, "trades_today"):
            state.trades_today = 0
        state.save()
        return JsonResponse({"ok": True, "message": "Daily counters reset"})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)})




def auto_trade_status(request):
    state = AutoTradeState.get()
    cfg = load_config()

    # Build response data
    data = {
        "spot_enabled": state.spot_enabled,
        "spot_risk": state.spot_risk,
        "spot_max_trades": state.spot_max_trades,
        "spot_trades_today": state.spot_trades_today,
        "spot_total": state.spot_total,
        "futures_enabled": state.futures_enabled,
        "futures_risk": state.futures_risk,
        "futures_max_trades": state.futures_max_trades,
        "futures_trades_today": state.futures_trades_today,
        "futures_total": state.futures_total,
        "testnet": cfg.auto.testnet,
    }

    # Try to get live balances and positions if trader is available
    try:
        from src.trading.binance_trader import BinanceTrader

        if cfg.auto.has_keys:
            # Spot trader
            if state.spot_enabled:
                spot_trader = BinanceTrader(
                    api_key=cfg.auto.api_key,
                    api_secret=cfg.auto.api_secret,
                    mode="spot",
                    live=not cfg.auto.testnet,
                )
                bal = spot_trader.get_balance()
                data["spot_balance"] = bal.get("available_balance", 0)
                data["spot_wallet"] = bal.get("wallet_balance", 0)
                data["spot_error"] = bal.get("error", "")
                data["spot_open_orders"] = len(spot_trader.get_open_orders())
                # Spot positions (open orders only, as spot doesn't have positions like futures)
                spot_orders = spot_trader.get_open_orders()
                spot_positions = []
                for o in spot_orders:
                    spot_positions.append({
                        "symbol": o.get("symbol"),
                        "side": o.get("side"),
                        "qty": o.get("origQty"),
                        "price": o.get("price"),
                        "type": o.get("type"),
                    })
                data["spot_positions"] = spot_positions
            else:
                data["spot_balance"] = 0
                data["spot_wallet"] = 0
                data["spot_error"] = ""
                data["spot_open_orders"] = 0
                data["spot_positions"] = []

            # Futures trader
            if state.futures_enabled:
                fut_trader = BinanceTrader(
                    api_key=cfg.auto.api_key,
                    api_secret=cfg.auto.api_secret,
                    mode="futures",
                    live=not cfg.auto.testnet,
                )
                bal = fut_trader.get_balance()
                data["futures_balance"] = bal.get("available_balance", 0)
                data["futures_wallet"] = bal.get("wallet_balance", 0)
                data["futures_pnl"] = bal.get("unrealised_pnl", 0)
                data["futures_error"] = bal.get("error", "")
                data["futures_open_orders"] = len(fut_trader.get_open_orders())

                # Futures positions
                positions = fut_trader.get_positions()
                fut_positions = []
                for p in positions:
                    amt = float(p.get("positionAmt", 0))
                    if abs(amt) < 0.0001:
                        continue
                    entry = float(p.get("entryPrice", 0))
                    mark = float(p.get("markPrice", entry))
                    pnl = float(p.get("unRealizedProfit", 0))
                    notional = abs(amt) * entry
                    pnl_pct = (pnl / notional * 100) if notional > 0 else 0
                    fut_positions.append({
                        "symbol": p.get("symbol"),
                        "side": "LONG" if amt > 0 else "SHORT",
                        "qty": abs(amt),
                        "entry": entry,
                        "mark": mark,
                        "pnl": pnl,
                        "pnl_pct": round(pnl_pct, 2),
                        "leverage": p.get("leverage", 1),
                        "liq": float(p.get("liquidationPrice", 0)),
                        "notional": notional,
                        "orders": [],  # populated below if needed
                    })
                data["futures_positions"] = fut_positions
            else:
                data["futures_balance"] = 0
                data["futures_wallet"] = 0
                data["futures_pnl"] = 0
                data["futures_error"] = ""
                data["futures_open_orders"] = 0
                data["futures_positions"] = []

    except Exception as e:
        data["spot_error"] = str(e)
        data["futures_error"] = str(e)

    # v5: Adaptive threshold from scanner
    try:
        from src.scanner import Scanner
        # Assuming you have a singleton scanner instance; adjust as needed.
        # If scanner is not globally available, you can instantiate a new one.
        # For simplicity, we'll return a default if not found.
        # In practice, you might store the scanner in the app config or use a module-level variable.
        from src.scanner import _scanner_instance  # you would need to set this in scanner.py
        if _scanner_instance:
            data["adaptive_threshold"] = _scanner_instance._adaptive_threshold
        else:
            data["adaptive_threshold"] = 48
    except Exception:
        data["adaptive_threshold"] = 48

    return JsonResponse(data)


@csrf_exempt
@require_POST
@require_POST
def auto_trade_toggle(request):
    """POST: toggle spot/futures independently, update risk/max settings."""
    from .models import AutoTradeState
    state = AutoTradeState.get()
    data  = json.loads(request.body or "{}")

    if "spot_enabled"    in data: state.spot_enabled    = bool(data["spot_enabled"])
    if "futures_enabled" in data: state.futures_enabled = bool(data["futures_enabled"])
    if "spot_risk"       in data: state.spot_risk           = min(5.0, max(0.1, float(data["spot_risk"])))
    if "futures_risk"    in data: state.futures_risk         = min(5.0, max(0.1, float(data["futures_risk"])))
    if "spot_max"        in data: state.spot_max_trades      = max(1, min(20, int(data["spot_max"])))
    if "futures_max"     in data: state.futures_max_trades   = max(1, min(20, int(data["futures_max"])))
    # Legacy fields — keep in sync
    state.enabled = state.spot_enabled or state.futures_enabled
    state.save()

    logger.info("AutoTrade Spot=%s Futures=%s",
                "ON" if state.spot_enabled else "OFF",
                "ON" if state.futures_enabled else "OFF")

    # If just turned ON — immediately execute today's pending signals in background
    just_enabled_spot    = state.spot_enabled    and data.get("spot_enabled")    is True
    just_enabled_futures = state.futures_enabled and data.get("futures_enabled") is True
    if just_enabled_spot or just_enabled_futures:
        import threading
        from config import load_config as _lc
        _cfg = _lc()   # load config here so it's captured in closure
        _state = state
        def _run():
            try:
                _execute_pending_signals(_state, _cfg)
            except Exception as e:
                logger.error("Pending auto-trade error: %s", e)
        threading.Thread(target=_run, daemon=True).start()
        logger.info("Pending signal execution triggered in background")

    return JsonResponse({
        "ok":               True,
        "spot_enabled":     state.spot_enabled,
        "futures_enabled":  state.futures_enabled,
        "spot_risk":        state.spot_risk,
        "futures_risk":     state.futures_risk,
        "spot_max_trades":  state.spot_max_trades,
        "futures_max_trades": state.futures_max_trades,
    })


def _execute_pending_signals(state, cfg):
    """Run immediately when auto-trade is toggled ON — no manual command needed."""
    import requests as _req
    from django.utils import timezone
    from zoneinfo import ZoneInfo
    from .models import SignalRecord
    from src.trading.binance_trader import BinanceTrader

    EAT = ZoneInfo("Africa/Dar_es_Salaam")
    now = timezone.now().astimezone(EAT)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Sync today's trade counter from DB (single source of truth)
    try:
        from dashboard.models import SignalRecord
        from datetime import timedelta
        today_end = today_start + timedelta(days=1)
        actual_spot = SignalRecord.objects.filter(
            created_at__gte=today_start, created_at__lt=today_end,
            notes__icontains="AUTO_SPOT:YES"
        ).count()
        actual_fut  = SignalRecord.objects.filter(
            created_at__gte=today_start, created_at__lt=today_end,
            notes__icontains="AUTO_FUT:YES"
        ).count()
        if (state.spot_trades_today != actual_spot or
                state.futures_trades_today != actual_fut):
            state.spot_trades_today    = actual_spot
            state.futures_trades_today = actual_fut
            state.save(update_fields=["spot_trades_today","futures_trades_today"])
            logger.info("Counters synced → spot=%d fut=%d today", actual_spot, actual_fut)
    except Exception as _ce:
        logger.debug("Counter sync: %s", _ce)

    pending = list(SignalRecord.objects.filter(
        created_at__gte=today_start,
        outcome="PENDING",
    ).order_by("-confidence"))

    logger.info("Auto-execute: %d pending signals today", len(pending))

    def _price(symbol):
        try:
            r = _req.get("https://api.binance.com/api/v3/ticker/price",
                         params={"symbol": symbol}, timeout=5)
            return float(r.json()["price"]) if r.ok else None
        except Exception:
            return None

    class _Sig:
        pass

    seen = set()
    for rec in pending:
        if rec.symbol in seen:
            continue
        seen.add(rec.symbol)

        price = _price(rec.symbol) or rec.entry_price
        sig = _Sig()
        sig.symbol=rec.symbol; sig.signal=rec.signal; sig.grade=rec.grade
        sig.price=price; sig.tp1=rec.tp1; sig.tp2=rec.tp2
        sig.tp3=rec.tp3; sig.sl=rec.sl
        sig.btc_score=rec.btc_score; sig.confidence=rec.confidence

        notes_updated = False

        # Spot — BUY only
        if state.spot_enabled and rec.signal == "BUY" and "AUTO_SPOT:" not in (rec.notes or ""):
            if state.spot_trades_today < state.spot_max_trades:
                try:
                    trader = BinanceTrader(cfg.auto.api_key, cfg.auto.api_secret,
                                           mode="spot", live=not cfg.auto.testnet,
                                           risk_pct=state.spot_risk)
                    result = trader.execute_signal(sig, trader.get_available_balance())
                    mark = "AUTO_SPOT:" + ("YES" if result.success else f"FAIL {result.error[:40]}")
                    rec.notes = (rec.notes or "") + " | " + mark
                    notes_updated = True
                    if result.success:
                        state.spot_trades_today += 1
                        state.spot_total += 1
                        state.save(update_fields=["spot_trades_today", "spot_total"])
                        logger.info("AUTO_SPOT ✅ %s %s @ %.6g", result.side, rec.symbol, result.entry_price)
                    else:
                        logger.warning("AUTO_SPOT ❌ %s — %s", rec.symbol, result.error)
                except Exception as e:
                    logger.error("Spot exec error %s: %s", rec.symbol, e)

        # Futures — BUY (long) + SELL (short)
        if state.futures_enabled and "AUTO_FUT:" not in (rec.notes or ""):
            if state.futures_trades_today < state.futures_max_trades:
                try:
                    trader = BinanceTrader(cfg.auto.api_key, cfg.auto.api_secret,
                                           mode="futures", live=not cfg.auto.testnet,
                                           risk_pct=state.futures_risk)
                    result = trader.execute_signal(sig, trader.get_available_balance())
                    mark = "AUTO_FUT:" + ("YES" if result.success else f"FAIL {result.error[:40]}")
                    rec.notes = (rec.notes or "") + " | " + mark
                    notes_updated = True
                    if result.success:
                        state.futures_trades_today += 1
                        state.futures_total += 1
                        state.save(update_fields=["futures_trades_today", "futures_total"])
                        logger.info("AUTO_FUT ✅ %s %s @ %.6g sl=%s", result.side, rec.symbol, result.entry_price, result.sl_order_id)
                    else:
                        logger.warning("AUTO_FUT ❌ %s — %s", rec.symbol, result.error)
                except Exception as e:
                    logger.error("Futures exec error %s: %s", rec.symbol, e)

        if notes_updated:
            rec.save(update_fields=["notes"])


@csrf_exempt
@require_POST
def auto_trade_emergency_stop(request):
    """POST: cancel all open orders immediately."""
    from config import load_config
    from .models import AutoTradeState

    state = AutoTradeState.get()
    cfg   = load_config()
    result = {"ok": False, "cancelled": [], "errors": []}

    if cfg.auto.api_key and cfg.auto.api_secret:
        try:
            from src.trading.binance_trader import BinanceTrader
            trader = BinanceTrader(
                api_key    = cfg.auto.api_key,
                api_secret = cfg.auto.api_secret,
                mode       = state.mode,
                live       = not cfg.auto.testnet,
            )
            res = trader.cancel_all_orders()
            result["ok"]        = True
            result["cancelled"] = res.get("cancelled", [])
            result["errors"]    = res.get("errors", [])
            logger.warning("EMERGENCY STOP executed — cancelled: %s", res["cancelled"])
        except Exception as e:
            result["errors"].append(str(e))
    else:
        result["errors"].append("No API keys configured")

    # Disable auto-trade after emergency stop
    state.enabled = False
    state.save(update_fields=["enabled"])

    return JsonResponse(result)