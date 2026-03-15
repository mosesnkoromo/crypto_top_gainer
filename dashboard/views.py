"""
dashboard/views.py
───────────────────
All API endpoints for the dashboard.
"""

import json
from datetime import timedelta, date
from django.shortcuts import render
from django.http import JsonResponse
from django.utils import timezone
from django.db.models import Count, Avg, Sum, Q, F
from django.views.decorators.csrf import csrf_exempt
from .models import SignalRecord, ScanRecord, CapitalRecord


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
        "gain_24h": s.gain_24h, "rsi": s.rsi, "btc_score": s.btc_score,
        "outcome": s.outcome, "profit_pct": s.profit_pct,
        "auto_checked": s.auto_checked,
        "created_at": s.created_at.strftime("%d %b %H:%M"),
        "created_date": s.created_at.strftime("%Y-%m-%d"),
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
            "date": ds.strftime("%d %b"),
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
        {"time": s.scanned_at.strftime("%d %b %H:%M"),
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
