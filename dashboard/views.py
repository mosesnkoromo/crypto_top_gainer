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
from django.views.decorators.http import require_http_methods, require_POST

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
    """GET: independent spot + futures status."""
    from .models import AutoTradeState
    from config import load_config
    state = AutoTradeState.get()
    cfg   = load_config()
    has_keys = cfg.auto.has_keys

    resp = {
        # Spot
        "spot_enabled":      state.spot_enabled,
        "spot_risk":         state.spot_risk,
        "spot_max_trades":   state.spot_max_trades,
        "spot_trades_today": state.spot_trades_today,
        "spot_total":        state.spot_total,
        "spot_balance":      0.0,
        "spot_open_orders":  0,
        # Futures
        "futures_enabled":      state.futures_enabled,
        "futures_risk":         state.futures_risk,
        "futures_max_trades":   state.futures_max_trades,
        "futures_trades_today": state.futures_trades_today,
        "futures_total":        state.futures_total,
        "futures_balance":      0.0,
        "futures_open_orders":  0,
        "futures_positions":    [],
        # Meta
        "has_keys": has_keys,
        "testnet":  cfg.auto.testnet,
        "block_spot":    "",
        "block_futures": "",
    }

    if has_keys and (state.spot_enabled or state.futures_enabled):
        import time as _t
        ts = _t.time()
        cached = _AT_CACHE.get("at_v2")
        if cached and (ts - cached["ts"]) < 60:
            resp.update(cached["data"])
        else:
            live = {
                # Always set defaults so JS never gets undefined (prevents NaN)
                "spot_balance": 0.0, "spot_open_orders": 0,
                "futures_balance": 0.0, "futures_open_orders": 0,
                "futures_positions": [],
            }
            from src.trading.binance_trader import BinanceTrader

            if state.spot_enabled:
                try:
                    st = BinanceTrader(cfg.auto.api_key, cfg.auto.api_secret,
                                       mode="spot", live=not cfg.auto.testnet)
                    b = st.get_balance()   # returns dict
                    live["spot_balance"]      = round(float(b.get("available_balance", 0) or 0), 2)
                    live["spot_wallet"]       = round(float(b.get("wallet_balance",   0) or 0), 2)
                    spot_orders = st.get_open_orders() or []
                    # Ensure it's actually a list of dicts
                    spot_orders = [o for o in spot_orders if isinstance(o, dict)]
                    live["spot_open_orders"] = len(spot_orders)
                    # Group orders by symbol to display as "positions"
                    spot_by_sym = {}
                    for o in spot_orders:
                        sym2 = o.get("symbol","")
                        spot_by_sym.setdefault(sym2, []).append({
                            "order_id": str(o.get("orderId","")),
                            "type":     o.get("type",""),
                            "side":     o.get("side",""),
                            "price":    o.get("price","0"),
                            "stop":     o.get("stopPrice","0"),
                            "qty":      o.get("origQty","0"),
                            "filled":   o.get("executedQty","0"),
                            "status":   o.get("status",""),
                        })
                    live["spot_positions"] = [
                        {"symbol": sym2, "orders": ords}
                        for sym2, ords in spot_by_sym.items()
                    ]
                    live["spot_open_orders_detail"] = [
                        {"symbol": o.get("symbol",""), "type": o.get("type",""),
                         "side": o.get("side",""), "price": o.get("price","0"),
                         "qty": o.get("origQty","0"), "status": o.get("status","")}
                        for o in spot_orders
                    ]
                except Exception as e:
                    live["spot_error"] = str(e)[:80]
                    logger.warning("Spot status fetch: %s", str(e)[:60])

            if state.futures_enabled:
                try:
                    ft = BinanceTrader(cfg.auto.api_key, cfg.auto.api_secret,
                                       mode="futures", live=not cfg.auto.testnet)
                    b = ft.get_balance()   # returns dict
                    live["futures_balance"]     = round(float(b.get("available_balance", 0) or 0), 2)
                    live["futures_wallet"]      = round(float(b.get("wallet_balance",   0) or 0), 2)
                    live["futures_pnl"]         = round(float(b.get("unrealised_pnl",  0) or 0), 2)
                    regular_orders = ft.get_open_orders() or []
                    # Also fetch algo orders (TAKE_PROFIT_MARKET, STOP_MARKET placed via algoOrder)
                    try:
                        algo_resp = ft._req("GET", "/fapi/v1/openAlgoOrders", {})
                        algo_orders = algo_resp if isinstance(algo_resp, list) else                                       (algo_resp or {}).get("orders", []) if isinstance(algo_resp, dict) else []
                        # Normalise algo order fields to match regular order format
                        for ao in algo_orders:
                            if "algoId" in ao and "orderId" not in ao:
                                ao["orderId"]    = ao["algoId"]
                                ao["stop_price"] = ao.get("triggerPrice", "0")
                                ao["type"]       = ao.get("orderType", ao.get("type",""))
                                ao["origQty"]    = ao.get("quantity","0")
                                ao["status"]     = ao.get("algoStatus","NEW")
                    except Exception:
                        algo_orders = []
                    all_orders = regular_orders + algo_orders
                    live["futures_open_orders"] = len(all_orders)
                    pos = ft.get_positions()
                    # Get open orders to match with positions
                    # Build orders-by-symbol using combined regular+algo orders
                    orders_by_sym = {}
                    for o in all_orders:
                        s2 = o.get("symbol","")
                        orders_by_sym.setdefault(s2, []).append({
                            "order_id":   str(o.get("orderId", o.get("algoId",""))),
                            "type":       o.get("type",""),
                            "side":       o.get("side",""),
                            "stop_price": str(o.get("stopPrice", o.get("triggerPrice","0"))),
                            "price":      str(o.get("price","0")),
                            "qty":        str(o.get("origQty", o.get("quantity","0"))),
                            "status":     o.get("status", o.get("algoStatus","NEW")),
                        })

                    live["futures_positions"] = []
                    for p in pos:
                        amt      = float(p["positionAmt"])
                        entry    = float(p["entryPrice"])
                        pnl_usdt = float(p["unRealizedProfit"])
                        notional = abs(amt) * entry
                        pnl_pct  = round(pnl_usdt / max(notional, 0.01) * 100, 2)
                        # In one-way mode: positive amt = LONG, negative = SHORT
                        side = "LONG" if amt > 0 else "SHORT"
                        live["futures_positions"].append({
                            "symbol":   p["symbol"],
                            "side":     side,
                            "qty":      abs(amt),
                            "raw_qty":  amt,
                            "entry":    entry,
                            "mark":     float(p.get("markPrice", 0)),
                            "liq":      float(p.get("liquidationPrice", 0)),
                            "leverage": int(p.get("leverage", 1)),
                            "pnl":      round(pnl_usdt, 4),
                            "pnl_pct":  pnl_pct,
                            "notional": round(notional, 2),
                            "orders":   orders_by_sym.get(p["symbol"], []),
                        })
                except Exception as e:
                    live["futures_error"] = str(e)[:80]
                    logger.warning("Futures status fetch: %s", str(e)[:60])

            _AT_CACHE["at_v2"] = {"ts": ts, "data": live}
            resp.update(live)

    # Block reasons — always cast to float, never compare dicts
    try: sb = float(resp.get("spot_balance", 0) or 0)
    except (TypeError, ValueError): sb = 0.0
    try: fb = float(resp.get("futures_balance", 0) or 0)
    except (TypeError, ValueError): fb = 0.0

    resp["block_spot"]    = ("No API keys" if not has_keys else
                             "Balance $0 — transfer USDT to Spot wallet on Binance" if sb <= 0 else
                             f"Daily limit {state.spot_max_trades} reached" if state.spot_trades_today >= state.spot_max_trades else "")
    resp["block_futures"] = ("No API keys" if not has_keys else
                             "Balance $0 — transfer USDT to Futures wallet on Binance" if fb <= 0 else
                             f"Daily limit {state.futures_max_trades} reached" if state.futures_trades_today >= state.futures_max_trades else "")
    return JsonResponse(resp)


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
    if "spot_max"        in data: state.spot_max_trades      = max(1, min(100, int(data["spot_max"])))
    if "futures_max"     in data: state.futures_max_trades   = max(1, min(100, int(data["futures_max"])))
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
    # Count actual trades executed today from DB — single source of truth
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
    if state.spot_trades_today != actual_spot or state.futures_trades_today != actual_fut:
        state.spot_trades_today    = actual_spot
        state.futures_trades_today = actual_fut
        state.save(update_fields=["spot_trades_today","futures_trades_today"])
        logger.info("Counters corrected → spot=%d fut=%d today", actual_spot, actual_fut)

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


@csrf_exempt
@require_http_methods(["POST"])
def auto_trade_settings(request):
    """
    Partial settings update. Any subset of these keys may be sent:
        spot_risk, futures_risk, spot_max_trades, futures_max_trades
    Invalid values are silently ignored (only valid ones persist).
    """
    from .models import AutoTradeState

    try:
        data = json.loads(request.body or "{}")
    except ValueError:
        return JsonResponse({"ok": False, "error": "Invalid JSON"}, status=400)

    state = AutoTradeState.objects.first()
    if state is None:
        state = AutoTradeState.objects.create()

    updated = []

    # Risk %: 0.1 – 10
    for key in ("spot_risk", "futures_risk"):
        if key in data:
            try:
                v = float(data[key])
                if 0.1 <= v <= 10:
                    setattr(state, key, v)
                    updated.append(key)
            except (TypeError, ValueError):
                pass

    # Max trades/day: 1 – 1000
    for key in ("spot_max_trades", "futures_max_trades"):
        if key in data:
            try:
                v = int(data[key])
                if 1 <= v <= 1000:
                    setattr(state, key, v)
                    updated.append(key)
            except (TypeError, ValueError):
                pass

    if updated:
        state.save(update_fields=updated)

    return JsonResponse({"ok": True, "updated": updated})


@csrf_exempt
@require_http_methods(["POST"])
def auto_trade_reset_counters(request):
    """Zero out today's spot + futures trade counters without touching totals."""
    from .models import AutoTradeState

    state = AutoTradeState.objects.first()
    if state is None:
        return JsonResponse({"ok": False, "error": "No state found"}, status=404)

    state.spot_trades_today = 0
    state.futures_trades_today = 0
    state.save(update_fields=["spot_trades_today", "futures_trades_today"])

    return JsonResponse({"ok": True})