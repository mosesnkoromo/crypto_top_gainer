"""
src/utils/formatter.py — v5
Single combined WhatsApp message: weekly report + open positions + new signals.
Everything in one message, sent every scan cycle.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from config import RiskConfig

_EAT = ZoneInfo("Africa/Dar_es_Salaam")

def _now() -> datetime:
    return datetime.now(_EAT)

from src.analysis.signal_engine import Signal


def _btc_icon(score: int) -> str:
    if score >= 72: return "🚀"
    if score >= 58: return "📈"
    if score >= 43: return "➡️"
    if score >= 28: return "📉"
    return "🔻"

def _outcome_icon(outcome: str) -> str:
    return {"TP1":"🎯","TP2":"🎯🎯","TP3":"🏆","SL":"🛑","BE":"🤝","PENDING":"⏳","TIMEOUT":"⌛"}.get(outcome,"❓")

def _dir_icon(sig: str) -> str:
    return "🟢" if sig == "BUY" else "🔴"


# ── Main combined message ─────────────────────────────────────

def fmt_digest(signals: list[Signal], btc: dict, risk: RiskConfig,
               news_items: list[dict] | None = None,
               closed_today: list | None = None,
               open_positions: list | None = None,
               scan_number: int = 0,
               spot_signals: list | None = None) -> str:

    now     = _now()
    now_str = now.strftime("%d %b  %H:%M EAT")
    day_str = now.strftime("%A, %d %b %Y")
    score   = btc.get("score", 50)
    btc_ic  = _btc_icon(score)

    if score >= 72:   mood = "Very strong 🔥 — ideal for longs"
    elif score >= 58: mood = "Bullish 📈 — BUY signals backed"
    elif score >= 43: mood = "Neutral ➡️ — trade carefully"
    elif score >= 28: mood = "Weak 📉 — favour sells"
    else:             mood = "Very weak 🔻 — protect capital"

    sells = [s for s in signals if s.signal == "SELL"]
    buys  = [s for s in signals if s.signal == "BUY"]

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  📊 SIGNAL REPORT",
        f"  📅 {day_str}",
        f"  🕐 {now_str}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    # ── BTC status ────────────────────────────────────────────
    lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  {btc_ic} BTC  ${btc.get('price',0):,.2f}  ·  {score}/100",
        f"  📊 {btc.get('trend','')}  ·  RSI {btc.get('rsi',0)}",
        f"  💬 {mood}",
        "",
        f"  📡 {len(signals)} New Signal{'s' if len(signals)!=1 else ''}  "
        f"·  🔴 {len(sells)} Sell  🟢 {len(buys)} Buy",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    # ── Spot signals (weekly oversold bounce) ────────────────
    spot_signals = spot_signals or []
    if spot_signals:
        lines += [
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  💎 SPOT OPPORTUNITIES ({len(spot_signals)})",
            f"  📈 Weekly Oversold Bounce Strategy",
            f"  ⚠️  Spot only — BUY and hold. No shorts.",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        for i, s in enumerate(spot_signals, 1):
            grade_parts = s.grade.split()
            g_label = grade_parts[0]
            g_icon  = grade_parts[1] if len(grade_parts) > 1 else ""

            # Size suggestion based on grade + BTC
            if g_label == "PRIME":
                size_note = "Full position (100%)"
            elif g_label == "GOOD":
                size_note = "Normal position (70%)"
            else:
                size_note = "Small position (40%)"
            if s.btc_score < 45:
                size_note += " — BTC weak, reduce size"

            reasons = [f.replace("✅ ","").replace("⚠️ ","~ ") for f in s.factors if f.startswith("✅")][:4]

            lines += [
                "",
                f"  ┌─────────────────────────",
                f"  │ 🟢 [{i}] {s.symbol}  {g_icon} {g_label}",
                f"  │ 🎯 {s.confidence}% confidence  ·  Score {s.score}",
                f"  │ ⏱  Hold {s.hold_weeks}",
                f"  ├─────────────────────────",
                f"  │ 📍 BUY (spot) — own the coin",
                f"  │ 💰 Entry   {s.price}",
                f"  │ 🟢 Entry   {s.entry_type}",
                f"  │ 📏 Size    {size_note}",
                f"  ├─────────────────────────",
                f"  │ 📊 Weekly RSI  {s.weekly_rsi}  (oversold < 38)",
                f"  │ 📊 Daily RSI   {s.daily_rsi}  ·  4H RSI {s.rsi_4h}",
                f"  │ 📦 Volume      {s.volume_note}",
                f"  ├─────────────────────────",
                f"  │ 🎯 TP1  {s.tp1}  (+{s.tp1_pct}%)  → sell 40%",
                f"  │ 🎯 TP2  {s.tp2}  (+{s.tp2_pct}%)  → sell 35%",
                f"  │ 🏆 TP3  {s.tp3}  (+{s.tp3_pct}%)  → sell 25%",
                f"  │ 🛑 SL   {s.sl}  (-{s.sl_pct}%) below weekly low",
                f"  │    ↳ Hold through dips above this level",
                f"  ├─────────────────────────",
                *[f"  │  · {r}" for r in reasons],
                f"  └─────────────────────────",
            ]
        lines += [
            "",
            "  💡 SPOT REMINDERS",
            "  ✅ You OWN the coin — dips are normal",
            "  ✅ SL is structural, not tight — hold above it",
            "  ✅ DCA: split entry over 2-3 days if unsure",
            "  ✅ TP levels are real resistance — not guesses",
            "  ✅ Weekly chart drives the trade — ignore hourly noise",
        ]

    # ── Spot signals (weekly oversold bounce) ────────────────
    spot_signals = spot_signals or []
    if spot_signals:
        lines += [
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  💎 SPOT OPPORTUNITIES ({len(spot_signals)})",
            f"  📈 Weekly Oversold Bounce Strategy",
            f"  ⚠️  Spot only — BUY and hold. No shorts.",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        for i, s in enumerate(spot_signals, 1):
            grade_parts = s.grade.split()
            g_label = grade_parts[0]
            g_icon  = grade_parts[1] if len(grade_parts) > 1 else ""

            # Size suggestion based on grade + BTC
            if g_label == "PRIME":
                size_note = "Full position (100%)"
            elif g_label == "GOOD":
                size_note = "Normal position (70%)"
            else:
                size_note = "Small position (40%)"
            if s.btc_score < 45:
                size_note += " — BTC weak, reduce size"

            reasons = [f.replace("✅ ","").replace("⚠️ ","~ ") for f in s.factors if f.startswith("✅")][:4]

            lines += [
                "",
                f"  ┌─────────────────────────",
                f"  │ 🟢 [{i}] {s.symbol}  {g_icon} {g_label}",
                f"  │ 🎯 {s.confidence}% confidence  ·  Score {s.score}",
                f"  │ ⏱  Hold {s.hold_weeks}",
                f"  ├─────────────────────────",
                f"  │ 📍 BUY (spot) — own the coin",
                f"  │ 💰 Entry   {s.price}",
                f"  │ 🟢 Entry   {s.entry_type}",
                f"  │ 📏 Size    {size_note}",
                f"  ├─────────────────────────",
                f"  │ 📊 Weekly RSI  {s.weekly_rsi}  (oversold < 38)",
                f"  │ 📊 Daily RSI   {s.daily_rsi}  ·  4H RSI {s.rsi_4h}",
                f"  │ 📦 Volume      {s.volume_note}",
                f"  ├─────────────────────────",
                f"  │ 🎯 TP1  {s.tp1}  (+{s.tp1_pct}%)  → sell 40%",
                f"  │ 🎯 TP2  {s.tp2}  (+{s.tp2_pct}%)  → sell 35%",
                f"  │ 🏆 TP3  {s.tp3}  (+{s.tp3_pct}%)  → sell 25%",
                f"  │ 🛑 SL   {s.sl}  (-{s.sl_pct}%) below weekly low",
                f"  │    ↳ Hold through dips above this level",
                f"  ├─────────────────────────",
                *[f"  │  · {r}" for r in reasons],
                f"  └─────────────────────────",
            ]
        lines += [
            "",
            "  💡 SPOT REMINDERS",
            "  ✅ You OWN the coin — dips are normal",
            "  ✅ SL is structural, not tight — hold above it",
            "  ✅ DCA: split entry over 2-3 days if unsure",
            "  ✅ TP levels are real resistance — not guesses",
            "  ✅ Weekly chart drives the trade — ignore hourly noise",
        ]

    # ── New signals (or "no signals" notice) ────────────────
    if not signals:
        lines += ["", "  😴 No new signals this scan", "  🔍 Markets scanned. Waiting for setup..."]
    else:
        groups = {"ULTRA": [], "STRONG": [], "STANDARD": []}
        for s in signals:
            groups.get(s.grade.split()[0], groups["STANDARD"]).append(s)

        grade_meta = {
            "ULTRA":    ("🔥", "ULTRA",    "~85%"),
            "STRONG":   ("⚡", "STRONG",   "~75%"),
            "STANDARD": ("📌", "STANDARD", "~63%"),
        }

        num = 1
        for key in ("ULTRA", "STRONG", "STANDARD"):
            grp = groups[key]
            if not grp:
                continue
            icon, label, acc = grade_meta[key]
            lines += ["", f"  {icon} {label}  ·  accuracy {acc}", ""]

            for s in grp:
                is_sell  = s.signal == "SELL"
                dir_icon = "🔴" if is_sell else "🟢"
                action   = "SELL / SHORT" if is_sell else "BUY / LONG"
                rsi_4h   = getattr(s, "rsi_4h",   getattr(s, "rsi_1h", 0))
                rsi_d    = getattr(s, "rsi_daily", 0)
                hold     = getattr(s, "hold_time", "5-15 min")
                strats   = getattr(s, "strategies_hit", [])
                strat_str = " + ".join(strats) if strats else "Confluence"
                reasons  = [f.replace("✅ ","").replace("⚠️ ","~ ") for f in s.factors if f.startswith("✅")][:4]

                sent = s.news_sentiment
                if isinstance(sent, dict): sent = sent.get("label","neutral")
                sent_line = "  │  📰 Positive news" if sent=="positive" else ("  │  📰 Negative news" if sent=="negative" else "")

                # Strategy type label
                strategy = getattr(s, "strategy", "TREND")
                strat_label = {"MEAN_REV": "📊 Mean Reversion",
                               "BREAKOUT": "🚀 Breakout",
                               "TREND":    "📈 Trend Follow"}.get(strategy, "📈 Trend Follow")

                lines += [
                    f"  ┌─────────────────────────",
                    f"  │ {dir_icon} [{num}] {s.symbol}  ·  {s.confidence}% conf",
                    f"  │ {strat_label}  ·  Hold {hold}",
                    f"  ├─────────────────────────",
                    f"  │ 📍 {action}",
                    f"  │ 💰 Entry  {s.price}",
                    f"  │ 📊 RSI  D:{rsi_d:.0f}  4H:{rsi_4h:.0f}",
                    f"  ├─────────────────────────",
                    f"  │ TP1  {s.tp1}  (+{risk.tp1_pct}%)  → {risk.tp1_close_pct}%",
                    f"  │ TP2  {s.tp2}  (+{risk.tp2_pct}%)  → {risk.tp2_close_pct}%",
                    f"  │ TP3  {s.tp3}  (+{risk.tp3_pct}%)  → {risk.tp3_close_pct}%",
                    f"  │ 🛑 SL   {s.sl}  (-{risk.sl_pct}%)",
                    f"  ├─────────────────────────",
                    *[f"  │  · {r}" for r in reasons],
                ]
                if sent_line: lines.append(sent_line)
                lines += [f"  └─────────────────────────", ""]
                num += 1

    # ── Weekly stats from DB ──────────────────────────────────
    try:
        from dashboard.models import SignalRecord
        from collections import defaultdict
        from zoneinfo import ZoneInfo as _ZI
        _ez = _ZI("Africa/Dar_es_Salaam")
        week_start = now - timedelta(days=7)
        week_qs    = SignalRecord.objects.filter(created_at__gte=week_start)
        w_closed   = list(week_qs.exclude(outcome="PENDING").order_by("-closed_at"))
        w_open     = list(week_qs.filter(outcome="PENDING").order_by("created_at"))
        w_wins     = [s for s in w_closed if s.outcome in ("TP1","TP2","TP3")]
        w_losses   = [s for s in w_closed if s.outcome == "SL"]
        w_total    = len(w_closed)
        w_wr       = round(len(w_wins) / w_total * 100) if w_total else 0
        w_pnl      = sum(s.profit_pct or 0 for s in w_closed if s.profit_pct is not None)

        # Grade breakdown
        ultra_c  = [s for s in w_closed if "ULTRA"  in s.grade]
        strong_c = [s for s in w_closed if "STRONG" in s.grade]
        u_wr = round(len([s for s in ultra_c  if s.outcome in ("TP1","TP2","TP3")]) / len(ultra_c)  * 100) if ultra_c  else 0
        s_wr = round(len([s for s in strong_c if s.outcome in ("TP1","TP2","TP3")]) / len(strong_c) * 100) if strong_c else 0

        if w_wr >= 65:     w_badge = "🏆 Outstanding"
        elif w_wr >= 55:   w_badge = "✅ Good"
        elif w_total == 0: w_badge = "📡 No closed trades"
        else:              w_badge = "⚠️ Below target"

        lines += [
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  📅 THIS WEEK  ·  {w_badge}",
            "",
            f"  🎯 Win Rate   {w_wr}%  ({len(w_wins)}W · {len(w_losses)}L · {len(w_open)} open)",
            f"  {'📈' if w_pnl>=0 else '📉'} Week P&L   {'+' if w_pnl>=0 else ''}{w_pnl:.1f}%",
            f"  🔥 Ultra      {u_wr}%  ({len(ultra_c)} trades)",
            f"  ⚡ Strong     {s_wr}%  ({len(strong_c)} trades)",
        ]

        # ── Closed trades: grouped by close date, showing SL/TP1/TP2/TP3 counts ──
        if w_closed:
            # Group by close date (EAT)
            by_day = defaultdict(list)
            for s in w_closed:
                day_key = s.closed_at.astimezone(_ez).strftime("%A %d %b") if s.closed_at else "Unknown"
                by_day[day_key].append(s)

            # Sort days newest first
            sorted_days = sorted(by_day.keys(),
                key=lambda d: next(s.closed_at for s in by_day[d] if s.closed_at),
                reverse=True)

            lines += ["", "  📋 CLOSED TRADES BY DAY"]

            for day in sorted_days:
                day_sigs = by_day[day]
                day_tp1  = sum(1 for s in day_sigs if s.outcome == "TP1")
                day_tp2  = sum(1 for s in day_sigs if s.outcome == "TP2")
                day_tp3  = sum(1 for s in day_sigs if s.outcome == "TP3")
                day_sl   = sum(1 for s in day_sigs if s.outcome == "SL")
                day_wins = day_tp1 + day_tp2 + day_tp3
                day_wr   = round(day_wins / len(day_sigs) * 100)
                day_pnl  = sum(s.profit_pct or 0 for s in day_sigs if s.profit_pct is not None)
                pnl_sign = "+" if day_pnl >= 0 else ""

                lines += [
                    "",
                    f"  📆 {day}",
                    f"  ├ Trades  {len(day_sigs)}  ·  WR {day_wr}%  ·  P&L {pnl_sign}{day_pnl:.1f}%",
                    f"  ├ 🏆 TP3    {day_tp3}",
                    f"  ├ 🎯🎯 TP2  {day_tp2}",
                    f"  ├ 🎯 TP1    {day_tp1}",
                    f"  └ 🛑 SL     {day_sl}",
                ]

        # ── Open positions: grouped by date opened, arranged vertically ──
        if w_open:
            # Group by open date
            open_by_day = defaultdict(list)
            for s in w_open:
                day_key = s.created_at.astimezone(_ez).strftime("%A %d %b")
                open_by_day[day_key].append(s)

            sorted_open_days = sorted(open_by_day.keys(),
                key=lambda d: next(s.created_at for s in open_by_day[d]),
                reverse=True)

            lines += ["", f"  ⏳ OPEN POSITIONS ({len(w_open)})"]

            for day in sorted_open_days:
                day_sigs = open_by_day[day]
                lines += ["", f"  📆 Opened {day}"]
                for s in day_sigs:
                    dir_ic  = _dir_icon(s.signal)
                    grade   = s.grade.split()[0]
                    open_t  = s.created_at.astimezone(_ez).strftime("%H:%M")
                    lines += [
                        f"  │",
                        f"  ├ {dir_ic} {s.symbol} [{grade}]  ·  {open_t}",
                        f"  │  Entry  {s.entry_price:,.5g}  ·  SL {s.sl:,.5g}",
                        f"  │  TP1 {s.tp1:,.5g}  TP2 {s.tp2:,.5g}  TP3 {s.tp3:,.5g}",
                    ]
                lines.append("  │")
        else:
            lines += ["", "  😴 No open positions"]

    except Exception as _exc:
        lines += ["", f"  📡 Weekly stats: not available ({_exc})"]

    # ── Footer ────────────────────────────────────────────────
    lines += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  ✅ After TP1: move SL to entry",
        "  ✅ After TP2: move SL to TP1",
        "  ✅ Max 2% capital per trade",
        "  ⚠️  Not financial advice. DYOR.",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    return "\n".join(lines)


# ── BTC update (when no signals) ─────────────────────────────

def fmt_btc_update(btc: dict) -> str:
    score = btc.get("score", 50)
    icon  = _btc_icon(score)
    if score >= 58: advice = "✅ Good for BUY signals"
    elif score >= 43: advice = "⚠️ Trade carefully"
    else: advice = "🔻 Avoid new longs"
    return (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  {icon} BTC UPDATE  ·  {_now().strftime('%H:%M EAT')}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"  Score  {score}/100  ·  {btc.get('trend','')}\n"
        f"  Price  ${btc.get('price',0):,.2f}\n"
        f"  RSI    {btc.get('rsi',0)}  ·  Vol {btc.get('vol_ratio',0)}x avg\n"
        f"  MACD   {'Bullish 📈' if btc.get('macd_bull') else 'Bearish 📉'}\n"
        f"\n"
        f"  {advice}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


# ── No signals scan ───────────────────────────────────────────

def fmt_no_signals(btc: dict) -> str:
    icon = _btc_icon(btc.get("score", 50))
    return (
        f"🔎 Scan done — no new signals.\n"
        f"{icon} BTC {btc.get('score',50)}/100  ·  {btc.get('trend','')}\n"
        f"💵 ${btc.get('price',0):,.2f}  ·  RSI {btc.get('rsi',0)}\n"
        f"🕐 {_now().strftime('%H:%M')} EAT"
    )


# ── Startup ───────────────────────────────────────────────────

def fmt_startup(version, scan_interval, top_gainers, min_gain, tp1, tp2, tp3, sl):
    return (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🤖 SIGNAL BOT v{version}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"  ✅ Bot started successfully\n"
        f"  📡 Scanning top gainers + liquid pairs\n"
        f"  ⏱  Every {scan_interval} min\n"
        f"  🎯 TP  {tp1}% / {tp2}% / {tp3}%\n"
        f"  🛑 SL  {sl}%\n"
        f"  ✨ ULTRA 🔥 & STRONG ⚡ signals only\n"
        f"\n"
        f"  🕐 {_now().strftime('%Y-%m-%d %H:%M')} EAT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


# ── Backwards compat (daily report still usable standalone) ──

def fmt_daily_report(closed_signals, open_signals, btc, scan_count=0):
    return fmt_btc_update(btc)