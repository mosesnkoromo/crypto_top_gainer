# BTC Strength Bot v3 — Django Dashboard + WhatsApp Alerts

## What's New in v3

| Feature | v2 | v3 |
|---|---|---|
| Technical indicators | RSI, EMA, MACD, ATR, Volume | + Bollinger Bands, StochRSI, Williams %R, OBV |
| News analysis | None | CryptoPanic integration (sentiment scoring) |
| Signal filter | Fires every 15 min | Only sends when STRONG/ULTRA found |
| Dashboard | None | Full Django analytics dashboard |
| Signal storage | None | PostgreSQL — every signal saved with outcome tracking |
| Win/loss stats | None | Daily, weekly, monthly with charts |
| Deployment | Local only | Render.com (free tier) |
| Confluence threshold | 3.0 | 3.8 (higher accuracy) |

---

## Quick Start (Local)

```bash
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS / Linux
source .venv/bin/activate

# 1. Setup
cp .env.example .env        # fill in WHATSAPP_NUMBER and WHAPI_TOKEN

# 2. Install
pip install -r requirements.txt

python manage.py makemigrations dashboard

python3 manage.py makemigrations
# 3. Database
python manage.py migrate

# 4. Create admin user (for /admin panel)
python manage.py createsuperuser

# 5. Run dashboard
python manage.py  runall
```

---

## Deploy to Render (Free)

1. Push code to a GitHub repo
2. Go to [render.com](https://render.com) → New → Blueprint
3. Connect your repo — Render reads `render.yaml` automatically
4. Set environment variables in the Render dashboard:
   - `WHATSAPP_NUMBER`
   - `WHAPI_TOKEN`
   - `CRYPTOPANIC_API_KEY` (optional)
5. Deploy — your dashboard will be live at `https://your-app.onrender.com`

---

## Recording Trade Outcomes

Visit your dashboard → click any **Pending** badge in the signals table → enter the outcome (TP1/TP2/TP3/SL). The win rate stats update immediately.

Or use the Django admin at `/admin` for bulk updates.

---

## Signal Grade Filter

By default the bot only sends **STRONG** and **ULTRA** signals (`MIN_SEND_GRADE = "STRONG"` in `signal_engine.py`). To also receive STANDARD signals, change it to `"STANDARD"`.

---

## Technical Indicators Used

| Indicator | Purpose |
|---|---|
| RSI (14) | Overbought / oversold detection |
| EMA 20/50 | Trend direction |
| MACD | Momentum direction |
| ATR | Volatility / EMA distance |
| Bollinger Bands %B | Overextension beyond bands |
| StochRSI | Secondary overbought confirmation |
| Williams %R | Reversal timing |
| OBV trend | Volume accumulation / distribution |
| News sentiment | Fundamental analysis boost/penalty |
| BTC Strength score | Master macro filter |

---

## Disclaimer

Not financial advice. Cryptocurrency trading involves significant risk of loss.
