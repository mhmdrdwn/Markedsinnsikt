# Markedsinnsikt AI

AI-powered marketing analytics dashboard for multi-client, multi-channel campaign performance. Built with Dash, XGBoost, and a multi-provider LLM layer (Groq → Gemini → Mistral).

## Features

**Analytics**
- KPI cards — total spend, revenue, conversions, ROAS, CTR with week-over-week trends
- Charts — ROAS by channel, conversions by campaign, spend distribution, weekly trend
- Anomaly notifications — ROAS drops ≥25% and spend spikes ≥50%
- CSV export and per-tab PDF reports

**Machine Learning**
- XGBoost forecasting — next-week ROAS per channel with 90% prediction intervals
- Walk-forward backtesting — XGBoost vs LinearRegression with MAE, RMSE, bias, direction accuracy
- Isolation Forest — multivariate anomaly detection across spend, ROAS, and CTR
- Z-score anomaly detection — statistical outliers on ROAS history
- Budget reallocation suggestions — ROI-based channel shift recommendations

**AI Insights**
- Structured analysis — executive decision, summary, key insights, anomalies, prioritised recommendations
- Multi-turn chat assistant — full campaign context, clickable example questions
- Multi-provider fallback — Groq → Gemini → Mistral, automatic on rate limits or failures
- Goal-aware — evaluates Brand Awareness on CPM, Lead Gen on CPL, Direct Sales on ROAS, App Installs on CPI

## Live demo

**https://markedsinnsikt.onrender.com**

## Running locally

**Prerequisites:** Python 3.11+. On macOS, XGBoost requires OpenMP:
```bash
brew install libomp
```

```bash
# 1. Clone and set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add API keys
cp .env.example .env
# Edit .env:
#   GROQ_API_KEY=gsk_...
#   GEMINI_API_KEY=...
#   GEMINI_MODEL=gemini-2.5-flash
#   MISTRAL_API_KEY=...

# 4. Start the dashboard
python main.py
```

Open **http://localhost:8050**

## Running with Docker

```bash
docker build -t markedsinnsikt .
docker run -p 8050:8050 --env-file .env markedsinnsikt
```

## Deploying on Render

1. Push the repo to GitHub
2. Create a Web Service on [Render](https://render.com), connect the repo, set runtime to `Docker`
3. Add environment variables: `GROQ_API_KEY`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `MISTRAL_API_KEY`

## Project structure

```
.
├── main.py               # Entry point — run locally or via gunicorn
├── data.py               # Synthetic dataset generator
│
├── app/
│   ├── main.py           # Dash app — layout, callbacks, PDF/CSV export
│   └── components/       # Shared UI components (reserved for future use)
│
├── ml/
│   ├── features.py       # Shared helpers: lag features, z-score
│   ├── models.py         # XGBoost forecasting, linear regression baseline, budget reallocation
│   ├── backtesting.py    # Walk-forward validation, business impact calculation
│   └── anomaly.py        # Z-score and Isolation Forest anomaly detection
│
├── ai/
│   └── insights.py       # Context builder, multi-provider LLM calls, prompts
│
├── tests/
│   ├── test_data.py
│   ├── test_ml_models.py
│   └── test_ai_assistant.py
│
├── assets/
│   └── style.css         # Custom CSS (auto-loaded by Dash)
├── requirements.txt
├── Dockerfile
└── .env                  # API keys (not committed)
```

## Data note

All campaign data is synthetically generated for demonstration purposes. No real client data is used.
