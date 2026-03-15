"""
src/utils/formatter.py — v3
Beautiful, human-readable single digest message.
"""

from datetime import datetime, timezone as tz
from zoneinfo import ZoneInfo
from config import RiskConfig

_EAT = ZoneInfo("Africa/Dar_es_Salaam")

def _now() -> datetime:
    """Current time in East Africa Time (EAT = UTC+3)."""
    return datetime.now(_EAT)
from src.analysis.signal_engine import Signal


def fmt_digest(signals: list[Signal], btc: dict, risk: RiskConfig,
               news_items: list[dict] | None = None) -> str:

    now   = _now().strftime("%d %b %Y  %H:%M EAT")
    sells = [s for s in signals if s.signal == "SELL"]
    buys  = [s for s in signals if s.signal == "BUY"]

    # BTC mood
    score = btc["score"]
    if score >= 72:   mood = "Bitcoin is very strong. Good conditions for buying altcoins."
    elif score >= 58: mood = "Bitcoin is bullish. Buy signals carry good weight."
    elif score >= 43: mood = "Bitcoin is neutral. Trade carefully and keep sizes small."
    elif score >= 28: mood = "Bitcoin is weak. Favour sell signals, avoid new longs."
    else:             mood = "Bitcoin is very weak. Consider exiting altcoin positions."

    btc_icon = "🟢" if score >= 58 else ("🟡" if score >= 43 else "🔴")

    lines = [
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  📊 BTC STRENGTH SCAN REPORT",
        f"  {now}",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"",
        f"{btc_icon} BITCOIN STATUS",
        f"",
        f"Price   ${btc['price']:,.2f}",
        f"Score   {btc['score']}/100  —  {btc['trend']}",
        f"RSI     {btc['rsi']}   |   Volume  {btc['vol_ratio']}x avg",
        f"EMA20   ${btc['ema20']:,.2f}   |   EMA50  ${btc['ema50']:,.2f}",
        f"MACD    {'Bullish ✅' if btc['macd_bull'] else 'Bearish ⚠️'}",
    ]

    # News sentiment for BTC if available
    if news_items:
        btc_news = [n for n in news_items if "BTC" in n.get("currencies","").upper() or "Bitcoin" in n.get("title","")][:2]
        if btc_news:
            lines.append(f"News    {btc_news[0]['title'][:60]}...")

    lines += [
        f"",
        f"{mood}",
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  🎯 {len(signals)} SIGNAL{'S' if len(signals)!=1 else ''} FOUND",
        f"  🔴 {len(sells)} Sell   🟢 {len(buys)} Buy",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    # Group by grade
    groups = {"ULTRA": [], "STRONG": [], "STANDARD": []}
    for s in signals:
        groups.get(s.grade.split()[0], groups["STANDARD"]).append(s)

    grade_headers = {
        "ULTRA":    ("🔥", "ULTRA SIGNALS", "Highest accuracy ~85%"),
        "STRONG":   ("⚡", "STRONG SIGNALS", "Good accuracy ~75%"),
        "STANDARD": ("📌", "STANDARD SIGNALS", "Moderate ~63%"),
    }

    num = 1
    for key in ("ULTRA", "STRONG", "STANDARD"):
        group = groups[key]
        if not group:
            continue

        icon, title, subtitle = grade_headers[key]
        lines += [
            f"",
            f"{icon} {title}  ({subtitle})",
            f"",
        ]

        for s in group:
            is_sell     = s.signal == "SELL"
            dir_icon    = "🔴" if is_sell else "🟢"
            action_word = "SELL / EXIT" if is_sell else "BUY / LONG"

            # Top 3 clean reasons
            reasons = [
                f.replace("✅ ", "").replace("⚠️ ", "~ ")
                for f in s.factors if f.startswith("✅")
            ][:3]

            # News sentiment badge
            sent = s.news_sentiment.get("label", "neutral") if isinstance(s.news_sentiment, dict) else s.news_sentiment
            sent_tag = "  📰 Positive news" if sent == "positive" else ("  📰 Negative news" if sent == "negative" else "")

            lines += [
                f"  ──────────────────────",
                f"  {dir_icon} [{num}] {s.symbol}   {s.confidence}% confidence",
                f"",
                f"  Action   {action_word}",
                f"  Entry    {s.price}",
                f"  Gain     +{s.gain_24h:.1f}%  |  RSI {s.rsi}",
                f"  StochRSI K:{s.stoch_rsi['k']} D:{s.stoch_rsi['d']}",
                f"  BB %B    {s.bb_pct_b:.2f}  |  Williams %R {s.williams}",
                f"",
                f"  Take Profits",
                f"  TP1  {s.tp1}  (+{risk.tp1_pct}%)  → close {risk.tp1_close_pct}%",
                f"  TP2  {s.tp2}  (+{risk.tp2_pct}%)  → close {risk.tp2_close_pct}%",
                f"  TP3  {s.tp3}  (+{risk.tp3_pct}%)  → close {risk.tp3_close_pct}%",
                f"  Stop {s.sl}  (-{risk.sl_pct}%)",
                f"",
                f"  Why",
                *[f"  • {r}" for r in reasons],
            ]
            if sent_tag:
                lines.append(f"  {sent_tag}")
            num += 1

    lines += [
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  📋 REMINDERS",
        f"",
        f"  • Set stop loss BEFORE entering",
        f"  • Move SL to breakeven after TP1",
        f"  • Max 2% capital per trade",
        f"  • ULTRA & STRONG only for best results",
        f"",
        f"  ⚠️ Not financial advice. DYOR.",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    return "\n".join(lines)


def fmt_btc_update(btc: dict) -> str:
    icon = "🟢" if btc["score"] >= 58 else ("🟡" if btc["score"] >= 43 else "🔴")
    return (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  📊 BTC STRENGTH UPDATE\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"{icon} Score   {btc['score']}/100\n"
        f"   Trend   {btc['trend']}\n"
        f"   Price   ${btc['price']:,.2f}\n"
        f"   RSI     {btc['rsi']}\n"
        f"   Volume  {btc['vol_ratio']}x avg\n"
        f"   MACD    {'Bullish ✅' if btc['macd_bull'] else 'Bearish ⚠️'}\n"
        f"\n"
        f"   🕐 {_now().strftime('%H:%M')} EAT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def fmt_startup(version, scan_interval, top_gainers, min_gain, tp1, tp2, tp3, sl):
    return (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🤖 BTC STRENGTH BOT v{version}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"  ✅ Bot started successfully\n"
        f"\n"
        f"  Scanning    Top {top_gainers} gainers\n"
        f"  Frequency   Every {scan_interval} minutes\n"
        f"  Min gain    {min_gain}%\n"
        f"  TP levels   {tp1}% / {tp2}% / {tp3}%\n"
        f"  Stop loss   {sl}%\n"
        f"  Signals     ULTRA & STRONG only\n"
        f"\n"
        f"  You'll receive one digest message\n"
        f"  per scan when signals are found.\n"
        f"\n"
        f"  🕐 {_now().strftime('%Y-%m-%d %H:%M')} EAT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def fmt_no_signals(btc: dict) -> str:
    return (
        f"🔎 Scan complete — no qualifying signals.\n"
        f"BTC: {btc['score']}/100  {btc['trend']}\n"
        f"🕐 {_now().strftime('%H:%M')} EAT"
    )