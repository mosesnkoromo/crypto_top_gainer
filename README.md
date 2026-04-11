# 🤖 BTC Scalp Bot v4.2.0

> **Institutional-grade crypto scalp trading bot** — top-down multi-timeframe analysis, rule-based signal engine, pre-trade simulation gate, and fully automated Binance Futures execution with WhatsApp alerts.

---

## Table of Contents

1. [What's New in v4.2](#whats-new-in-v42)
2. [Strategy Overview](#strategy-overview)
3. [Signal Gate Pipeline](#signal-gate-pipeline)
4. [Project Structure](#project-structure)
5. [Prerequisites](#prerequisites)
6. [Quick Start](#quick-start)
7. [Configuration Reference](#configuration-reference)
8. [Running the Bot](#running-the-bot)
9. [Dashboard](#dashboard)
10. [Deploying 24/7](#deploying-247)
11. [Troubleshooting](#troubleshooting)
12. [Disclaimer](#disclaimer)

---

## What's New in v4.2

### 🗑️ ML Removed
The ML signal filter (ml_filter.py, ml_historical.py) was removed entirely after analysis showed it was memorising training data (AUC=0.983, precision=100%) rather than learning genuine patterns. All decisions are now rule-based — transparent, auditable, and faster.

### 🌐 Top-Down HTF Cascade (Biggest Change)
Root cause of previous losses: the bot was firing BUY signals on 5m charts while the Daily and 4H trends were clearly bearish. The fix is a full top-down cascade:

```
1W → 1D → 4H → 1H → 15m → 5m
```

Each timeframe votes BULL (+weight) / BEAR (-weight) / NEUTRAL (0):

| Timeframe | Role | Weight | Gate |
|-----------|------|--------|------|
| 1 Week | Market structure | 3 | Soft |
| 1 Day | Trend direction | 3 | **Hard** |
| 4 Hour | Intermediate | 3 | **Hard** |
| 1 Hour | Entry zone | 2 | Soft |
| 15 Min | Fine timing | 1 | Soft |

**Hard gate:** If 1D AND 4H are both bearish → BUY is permanently blocked regardless of 5m pattern. This alone stops the majority of losing trades.

**Soft gate:** If total HTF score ≤ −4 → BUY blocked. If ≥ +4 → SELL blocked.

**HTF bonus:** Aligned timeframes add 0–20 pts to confluence score.

**ADX 4H:** Trend strength proxy. Higher ADX = stronger trend = higher confidence.

### 🔗 Indicator Correlation Gate
Counts how many indicators conflict with the trade direction. If ≥2 conflict, the signal is incoherent and blocked:

- **BUY conflicts:** WT overbought (>50), TQI choppy without WT gold cross, CCI >85, RSI >75
- **SELL conflicts:** WT oversold (<−50), same TQI check, CCI <−85, RSI <25

### 🔬 Institutional Simulator (v4.2)
Complete rewrite of the pre-trade simulation:

| Feature | Old | New |
|---------|-----|-----|
| Entry sampling | Fixed every 6 candles | Event-based (EMA+volume+ATR filter) |
| Hold time | Fixed 3 candles | Adaptive 2–8 (based on ATR/momentum) |
| TP exit | All-in at TP1 | Partial: 50% TP1 / 30% TP2 / 20% runner |
| SL after TP1 | Static | Moved to breakeven |
| After TP2 | Static | Trailing stop (price − 1×ATR) |
| Spread/slippage | None | 0.05% + 0.03% |
| TP/SL scaling | Absolute price (bug) | Scaled per sim entry |
| Timeout | Neutral | Actual P&L (can be positive) |
| Timeout block | High timeout penalty | Block only if mostly negative |
| Drawdown | Not tracked | Max drawdown ≤ 15% gate |
| Monte Carlo | No | 50 shuffled runs ≥ 50% profitable |
| Quality score | No | 4-factor quality gate |
| Volatility check | No | Dead market block (<0.3% range) |

### 🔢 Futures Symbol Normalization
Some tokens trade on Binance Futures with a 1000× prefix (LUNCUSDT → 1000LUNCUSDT). The bot now normalises automatically with a static list + dynamic fallback via exchangeInfo.

---

## Strategy Overview

### Signal Generation

```
BTC Strength Score → Regime Detection → HTF Cascade → Indicators → Score → Simulate → Execute
```

**Regime Detection (TQI)**
The Trend Quality Index (0–1) classifies each pair:
- `Strong_Trend_Impulse` (TQI > 0.75) — threshold 62, SL = 1.2×ATR
- `Trending` (TQI > 0.50) — threshold 68, SL = 1.5×ATR
- `Choppy_Range` (TQI ≤ 0.50) — threshold 80, SL = 1.8×ATR

**Confluence Scoring (0–100)**
- Regime bonus: +5 to +15
- Strong displacement candle: +12
- WaveTrend gold cross from oversold: +15
- WaveTrend alignment: +5
- MACD momentum: +8
- RSI in ideal zone: +8
- BTC alignment: +6
- HTF cascade bonus: +0 to +20

**Grades**
- `ULTRA` (≥88): Full position, highest confidence
- `STRONG` (≥75): 75% position
- `STANDARD` (≥62): 50% position

### Trade Management
- **TP1** at 0.8× SL distance (50% of position)
- **TP2** at 1.8× SL distance (30% of position)
- **TP3** at 3.0× SL distance (20% of position / runner)
- **Trailing stop** activates after TP1 (price − 1×ATR for longs)
- **SL** moves to breakeven after TP1

---

## Signal Gate Pipeline

Every signal passes through 14 sequential gates. Failure at any gate = trade blocked:

```
 1. Session time           06:00–23:00 UTC only
 2. ATR ≥ 0.3%            Not a flat/dead market
 3. TQI regime            Classify trend quality
 4. HTF cascade ★         1W→1D→4H→1H→15m alignment [HARD GATE]
 5. Confluence score      62–80 minimum by regime
 6. BTC filter            BUY needs BTC ≥ 30, SELL needs BTC ≤ 70
 7. RSI hard blocks       5m RSI >80 or 1H RSI >82 = no BUY
 8. Indicator correlation ≥2 conflicts = BLOCK (WT+TQI+CCI+RSI)
 9. WaveTrend gate        Gold cross confirmation
10. Displacement filter   Strong candle + volume confirmation
11. 1m Sniper             EMA9/21 + MACD on 1m chart
12. Pre-trade simulator   WR≥50%, E≥+0.08%, Monte Carlo, quality score
13. Execution re-sim      Re-run simulator at execution moment
14. Circuit breaker       Daily SL losses < 4%, max 2 open positions
```

---

## Project Structure

```
btc_bot_v4/
├── src/
│   ├── analysis/
│   │   ├── signal_engine.py       ★ Signal generation + HTF cascade
│   │   ├── signal_simulator.py    ★ Institutional pre-trade simulation
│   │   ├── btc_strength.py          BTC market strength score
│   │   ├── indicators.py            Technical indicator library
│   │   ├── news_engine.py           News sentiment scoring
│   │   └── trade_state_machine.py   Position state management
│   ├── trading/
│   │   └── binance_trader.py      ★ Execution + 1000-prefix normalization
│   ├── data/
│   │   └── binance_client.py        Binance REST client
│   ├── alerts/
│   │   └── whatsapp.py              WhatsApp delivery via Whapi
│   └── utils/
│       ├── logger.py                Structured logging
│       └── formatter.py             WhatsApp message formatting
├── dashboard/
│   ├── templates/
│   │   └── index.html             ★ Web dashboard (HTF card, sim stats)
│   ├── models.py                    SignalRecord, AutoTradeState, ScalpPosition
│   ├── views.py                   ★ API endpoints (ML endpoint stubbed)
│   └── urls.py                      URL routing
├── src/scanner.py                 ★ Main scan loop (2min cycle)
├── config.py                        Configuration dataclasses
├── manage.py                        Django entrypoint
└── requirements.txt                 Python dependencies

★ = modified in v4.2
```

---

## Prerequisites

- Python 3.11+
- Binance account with Futures API keys enabled
- Whapi account for WhatsApp delivery
- macOS / Linux (Windows: use WSL)

---

## Quick Start

```bash
# 1. Clone and enter project
cd "btc_bot_v4"

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Run migrations
python manage.py migrate

# 6. Start the bot
python manage.py runall
```

---

## Configuration Reference

```env
# Binance
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BINANCE_TESTNET=false

# Auto-trading
AUTO_TRADE_ENABLED=true
AUTO_TRADE_MODE=futures          # spot | futures
AUTO_TRADE_RISK_PCT=1.5          # % of balance per trade
AUTO_TRADE_DAILY_LOSS_LIMIT=6.0  # % daily loss limit

# WhatsApp
WHAPI_TOKEN=your_whapi_token
WHAPI_CHANNEL_ID=your_channel_id

# Optional
CRYPTOPANIC_API_KEY=             # News sentiment (optional)
```

---

## Running the Bot

```bash
# Start everything (scanner + dashboard + auto-trade)
python manage.py runall

# Dashboard only
python manage.py runserver

# Check signal outcomes manually
python manage.py check_outcomes
```

**Dashboard:** http://127.0.0.1:8000

---

## Dashboard

The web dashboard shows:

- **Live positions** — open futures positions with P&L
- **Signal history** — all signals with outcome (TP1/TP2/TP3/SL/PENDING)
- **Strategy Engine card** — HTF cascade gates, 14-gate pipeline
- **Pre-Trade Sim card** — approved/blocked count, block rate
- **Auto-trade controls** — enable/disable spot and futures, risk settings
- **BTC strength** — real-time BTC score with RSI

---

## Deploying 24/7

### Render / Railway (recommended)

```bash
# Procfile already configured
web: python manage.py runall
```

Set environment variables in the platform dashboard. The bot runs the scanner every 2 minutes and keeps the web server alive for the dashboard.

### VPS (systemd)

```ini
[Unit]
Description=BTC Scalp Bot v4.2
After=network.target

[Service]
WorkingDirectory=/home/user/btc_bot_v4
ExecStart=/home/user/btc_bot_v4/.venv/bin/python manage.py runall
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `SignalEngine.__init__() takes 2 positional arguments` | Old scanner calling new engine with extra args | Replace scanner.py with v4.2 |
| `ModuleNotFoundError: src.analysis.ml_filter` | Old ML files deleted but still imported | Replace signal_engine.py, scanner.py, views.py with v4.2 |
| `Invalid symbol -1121` for LUNCUSDT | 1000-prefix missing on Binance Futures | Replace binance_trader.py with v4.2 |
| `-4120 Wrong endpoint` on SL | STOP_MARKET rejected by Binance | v4.2 falls back to STOP_LIMIT automatically |
| `Bot thread crashed: too few valid entries` | Simulator couldn't find 6 valid sim entries | Normal — auto-approved and skipped |
| BUY blocked in bear market | BTC score < 30 or 1D+4H both bearish | Expected behavior — protecting capital |
| High block rate (80%+) | Bear market / all TFs misaligned | Expected — bot waits for quality setups |

---

## Signal Quality Log Examples

```
# Good signal — all TFs aligned
📊 LUNCUSDT HTF: 1W~ 1D↑ 4H↑ 1H↑ 15m↑ | score=+9 | ADX4H=47
✅ LUNCUSDT BUY | score=82/100 🟢🟢 STRONG | HTF:1W~1D↑4H↑1H↑15m↑(+9)

# Blocked — major TFs bearish
📊 ILVUSDT HTF: 1W↓ 1D↓ 4H↓ 1H↑ 15m~ | score=-7
❌ ILVUSDT BUY | MAJOR TFs BEARISH (1D↓+4H↓) → blocked

# Blocked — indicator conflicts
❌ ZAMAUSDT BUY | 2 indicator conflicts (WT=64↑overbought CCI=93↑overbought) → skip

# Blocked — simulator failed
🚫 SIM BLOCKED BUY TAOUSDT | WR=23% E=-0.54% | low quality setup (score=1/4)
```

---

## Disclaimer

This bot is for educational and personal use only. Cryptocurrency trading involves substantial risk of loss. Past performance of the simulator does not guarantee future results. Never trade with money you cannot afford to lose. The authors are not responsible for any financial losses.

---

*BTC Scalp Bot v4.2.0 — Rule-based + HTF Cascade + Institutional Simulator*