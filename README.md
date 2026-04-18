# Markedsinnsikt AI

AI-powered marketing analytics dashboard for multi-client, multi-channel campaign performance. Built with Dash, XGBoost, SHAP, and a multi-provider LLM layer (Groq → Gemini → Mistral).

## Live demo

**https://markedsinnsikt-dashboard-production.up.railway.app**

## Features

**Analytics**
- KPI cards — total spend, revenue, conversions, ROAS, CTR with week-over-week trends
- Charts — ROAS by channel, conversions by campaign, spend distribution, weekly trend
- Anomaly notifications — ROAS drops ≥25% and spend spikes ≥50%
- CSV export and per-tab HTML reports

**Machine Learning**
- XGBoost forecasting — next-week ROAS per channel with 90% prediction intervals
- SHAP explanations — global feature importance and local per-prediction attribution
- Walk-forward backtesting — XGBoost vs LinearRegression with MAE, RMSE, bias, direction accuracy
- Isolation Forest — multivariate anomaly detection across spend, ROAS, and CTR
- Z-score anomaly detection — statistical outliers on ROAS history
- Budget reallocation suggestions — ROI-based channel shift recommendations

**AI Insights (v3.0)**
- Agentic chat — LLM calls live data tools (channel performance, weekly trends, anomalies, comparisons) via Groq function calling
- RAG-lite context — question-aware data slice selection instead of full context dump
- Groundedness eval — scores AI output 0–100 against actual data, shown in UI
- Observability — per-call latency, token usage, and provider displayed on every response
- Versioned prompts — `ai/prompts.py` tracks prompt history from v1.0 → v3.0
- Structured analysis — executive decision, summary, key insights, anomalies, prioritised recommendations
- Multi-provider fallback — Groq → Gemini → Mistral, automatic on rate limits or failures
- Goal-aware — evaluates Brand Awareness on CPM, Lead Gen on CPL, Direct Sales on ROAS, App Installs on CPI

## Data

Campaign data is based on **Meta's open-source Robyn MMM dataset** (208 weeks, Nov 2015 – Nov 2019). Weekly spend trajectories for Meta Ads and Google Ads are real simulation data with adstock effects from Robyn's Marketing Mix Model. The dataset is committed as `data/robyn_weekly.parquet` — no download required at runtime.

## Running locally

**Prerequisites:** Python 3.11+. On macOS, XGBoost requires OpenMP:
```bash
brew install libomp
```

```bash
# 1. Clone and set up virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add API keys
cp .env.example .env
# Edit .env:
#   GROQ_API_KEY=gsk_...
#   GEMINI_API_KEY=...        (optional — fallback provider)
#   GEMINI_MODEL=gemini-2.5-flash
#   MISTRAL_API_KEY=...       (optional — fallback provider)

# 4. Start the dashboard
gunicorn app.main:server --bind 0.0.0.0:8050 --timeout 120
```

Open **http://localhost:8050**

## Running with Docker

```bash
docker build -t markedsinnsikt .
docker run -p 8050:8050 --env-file .env markedsinnsikt
```

## Deploying on Railway

1. Push the repo to GitHub
2. Create a new project on [Railway](https://railway.app), connect the repo — Railway auto-detects the Dockerfile
3. Add environment variables: `GROQ_API_KEY`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `MISTRAL_API_KEY`

## Project structure

```
.
├── app/
│   └── main.py              # Dash app — layout, callbacks, HTML export
│
├── ai/
│   ├── __init__.py          # Public exports
│   ├── insights.py          # Context builder, LLM calls, agentic tool loop, RAG-lite
│   ├── prompts.py           # Versioned prompt templates (v3.0)
│   ├── tools.py             # Tool definitions + executor for function calling
│   └── evals.py             # Groundedness scorer (0–100)
│
├── ml/
│   ├── __init__.py          # Public exports
│   ├── features.py          # Lag features, z-score helpers
│   ├── models.py            # XGBoost forecasting + SHAP, linear regression, budget reallocation
│   ├── backtesting.py       # Walk-forward validation, business impact calculation
│   └── anomaly.py           # Z-score and Isolation Forest anomaly detection
│
├── data/
│   ├── __init__.py          # get_dataset() — single public entry point
│   ├── robyn.py             # Robyn MMM loader, reshape, parquet cache
│   └── robyn_weekly.parquet # Committed dataset — 1,890 rows, 208 weeks (51 KB)
│
├── tests/
│   ├── test_data.py         # Robyn dataset schema and value range tests
│   ├── test_ml_models.py    # XGBoost, backtesting, anomaly detection tests
│   └── test_ai_assistant.py # Context builder, tools, evals, RAG-lite tests
│
├── assets/
│   └── style.css            # Custom CSS (auto-loaded by Dash)
│
├── Dockerfile
├── render.yaml
├── requirements.txt
└── .env                     # API keys (not committed)
```
