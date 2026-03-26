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
from django.http import JsonResponse
from django.utils import timezone
from django.db.models import Count, Avg, Sum, Q, F
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .models import SignalRecord, ScanRecord, CapitalRecord

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
        qs = qs.filter(created_at__gte=now.replace(hour=0,minute=0,second=0))
    elif period == "week":
        qs = qs.filter(created_at__gte=now - timedelta(days=7))
    elif period == "month":
        qs = qs.filter(created_at__gte=now - timedelta(days=30))
    else:
        # Default: today
        qs = qs.filter(created_at__gte=now.replace(hour=0,minute=0,second=0))

    if direction: qs = qs.filter(signal=direction.upper())
    if grade:     qs = qs.filter(grade=grade.upper())
    if symbol:    qs = qs.filter(symbol__icontains=symbol.upper())
    if outcome:   qs = qs.filter(outcome=outcome.upper())
    if date_from:
        try:    qs = qs.filter(created_at__date__gte=date_from)
        except: pass
    if date_to:
        try:    qs = qs.filter(created_at__date__lte=date_to)
        except: pass

    total_count = qs.count()
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    offset      = (page - 1) * per_page
    signals     = qs[offset:offset + per_page]

    return JsonResponse({
        "signals": [_sig_dict(s) for s in signals],
        "pagination": {
            "page": page, "per_page": per_page,
            "total": total_count, "total_pages": total_pages,
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

def auto_trade_status(request):
    """GET: full auto-trade status including live balance + open orders."""
    from .models import AutoTradeState
    from config import load_config
    import os

    state = AutoTradeState.get()
    cfg   = load_config()
    has_keys = bool(cfg.auto.api_key and cfg.auto.api_secret)

    resp = {
        "enabled":          state.enabled,
        "mode":             state.mode,
        "capital_usdt":     state.capital_usdt,
        "risk_per_trade":   state.risk_per_trade,
        "trades_today":     state.trades_today,
        "max_trades_day":   state.max_trades_day,
        "total_auto_trades":state.total_auto_trades,
        "has_keys":         has_keys,
        "testnet":          cfg.auto.testnet,
        "live_balance":     0.0,
        "available_balance": 0.0,
        "unrealised_pnl":    0.0,
        "balance_fetch_ok":  False,
        "open_orders":      0,
        "open_positions":   [],
        "block_reason":     "",   # populated if trades are being blocked
    }

    # Only fetch live Binance data when auto-trade is actively enabled
    # Avoids spamming API with 401s when keys aren't configured or trade is OFF
    if has_keys and state.enabled:
        try:
            from src.trading.binance_trader import BinanceTrader
            trader = BinanceTrader(
                api_key    = cfg.auto.api_key,
                api_secret = cfg.auto.api_secret,
                mode       = state.mode,
                live       = not cfg.auto.testnet,
            )
            bal_info = trader.get_balance()
            resp["live_balance"]       = bal_info["wallet_balance"]    # total deposited — stable
            resp["available_balance"]  = bal_info["available_balance"] # free margin — fluctuates
            resp["unrealised_pnl"]     = bal_info["unrealised_pnl"]
            resp["balance_fetch_ok"]   = bal_info["error"] == ""
            resp["open_orders"]        = len(trader.get_open_orders())
            if state.mode == "futures":
                pos = trader.get_positions()
                resp["open_positions"] = [
                    {
                        "symbol":  p["symbol"],
                        "side":    p["positionSide"],
                        "qty":     p["positionAmt"],
                        "entry":   p["entryPrice"],
                        "pnl":     p["unRealizedProfit"],
                        "pnl_pct": round(
                            float(p["unRealizedProfit"]) /
                            max(abs(float(p["positionAmt"]) * float(p["entryPrice"])), 1) * 100, 2),
                    }
                    for p in pos
                ]
        except Exception as e:
            resp["balance_error"] = str(e)

        # Check if trades would be blocked and why
        wallet_bal = resp.get("live_balance", 0.0)
        avail_bal  = resp.get("available_balance", 0.0)
        if not resp.get("balance_fetch_ok"):
            resp["block_reason"] = "Could not fetch balance — check API key permissions"
        elif wallet_bal <= 0:
            resp["block_reason"] = ("Wallet balance $0.00 — transfer USDT to your "
                + ("Futures wallet" if state.mode == "futures" else "Spot wallet")
                + " on Binance, or check API key has correct permissions")
        elif avail_bal < 5:
            resp["block_reason"] = (f"Available margin ${avail_bal:.2f} is low "
                                    f"(wallet total ${wallet_bal:.2f})")
        elif state.trades_today >= state.max_trades_day:
            resp["block_reason"] = f"Daily trade limit {state.max_trades_day} reached"
        else:
            resp["block_reason"] = ""   # all good

    return JsonResponse(resp)


@csrf_exempt
@require_POST
def auto_trade_toggle(request):
    """POST: update auto-trade settings and enable/disable."""
    from .models import AutoTradeState
    state = AutoTradeState.get()
    data  = json.loads(request.body or "{}")

    if "enabled" in data:
        state.enabled = bool(data["enabled"])
    if "mode" in data and data["mode"] in ("spot", "futures"):
        state.mode = data["mode"]
    if "capital_usdt" in data:
        state.capital_usdt = max(10.0, float(data["capital_usdt"]))
    if "risk_per_trade" in data:
        state.risk_per_trade = min(5.0, max(0.5, float(data["risk_per_trade"])))
    if "max_trades_day" in data:
        state.max_trades_day = max(1, min(20, int(data["max_trades_day"])))
    state.save()

    action = "ENABLED" if state.enabled else "DISABLED"
    logger.info("AutoTrade %s — mode=%s risk=%.1f%% max_trades=%d",
                action, state.mode, state.risk_per_trade, state.max_trades_day)
    return JsonResponse({"ok": True, "enabled": state.enabled, "mode": state.mode})


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