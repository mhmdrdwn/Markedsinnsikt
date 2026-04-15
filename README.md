# Markedsinnsikt AI

Markedsinnsikt AI is a decision-support tool that transforms marketing performance data into actionable insights and recommendations across campaigns, channels, and clients.

The system combines structured data with AI reasoning and machine learning to help marketing teams move from reporting to action — surfacing what is happening, why it might be happening, and what to do about it.

Built with Dash, XGBoost, and a multi-provider AI fallback chain (Groq → Gemini → Mistral).

## Features

### Analytics & KPIs
- **Multi-client filtering** — switch between clients, campaigns, and channels; all charts and insights update dynamically
- **KPI cards** — total spend, revenue, conversions, ROAS, CTR with week-over-week trend arrows
- **Performance charts** — ROAS by channel, conversions by campaign, spend distribution, weekly trend — each with a rule-based interpretation line
- **Anomaly notifications** — bell icon alerts for ROAS drops ≥25% and spend spikes ≥50%

### AI Insights
- **AI Insights** — one-click structured analysis: summary, key insights, anomalies, and prioritised recommendations
- **AI Assistant** — multi-turn chat with full campaign context; clickable example questions to get started
- **Multi-provider fallback** — tries Groq → Gemini → Mistral automatically; if one provider hits rate limits or fails, the next is used transparently
- **Goal-aware analysis** — evaluates Brand Awareness on CPM/reach, Lead Gen on CPL, Direct Sales on ROAS, App Installs on CPI
- **Cross-client benchmarking** — compares selected client against portfolio averages
- **Audience segment analysis** — ranks segments by ROAS and conversions

### Machine Learning (ml_models.py)
- **XGBoost forecasting** — predicts next week's spend and ROAS per channel using lag features (lag_1, lag_2, rolling mean, trend index); trained with `XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)`
- **90% prediction intervals** — computed from training residuals, displayed as shaded confidence bands
- **Walk-forward backtesting** — compares LinearRegression vs XGBoost using expanding-window evaluation; reports MAE, RMSE, winner, and improvement percentage per channel
- **Isolation Forest anomaly detection** — multi-dimensional detection across spend + ROAS + CTR per campaign/week using `StandardScaler` + `IsolationForest(contamination=0.10)`
- **Z-score anomaly detection** — statistical detection on ROAS history per campaign
- **Budget reallocation suggestions** — rule-based recommendations based on channel ROAS ranking

## Live demo

**https://markedsinnsikt.onrender.com**

## Running locally

### Prerequisites

- Python 3.11+
- On **macOS**, XGBoost requires OpenMP:
  ```bash
  brew install libomp
  ```

### Setup

1. Clone the repo and create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy the env file and add your API keys:

   ```bash
   cp .env.example .env
   ```

   ```
   GROQ_API_KEY=gsk_...
   GEMINI_API_KEY=...
   GEMINI_MODEL=gemini-2.5-flash
   MISTRAL_API_KEY=...
   ```

4. Start the dashboard:

   ```bash
   python dashboard.py
   ```

   Open **http://localhost:8050**

## Running with Docker

```bash
docker build -t markedsinnsikt .
docker run -p 8050:8050 --env-file .env markedsinnsikt
```

## Deploying on Render

1. Push the repo to GitHub
2. Create a new Web Service on [Render](https://render.com), connect the repo
3. Set **Language** to `Docker`
4. Add environment variables: `GROQ_API_KEY`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `MISTRAL_API_KEY`
5. Deploy — the app will be available at your Render URL

## Project structure

```
.
├── dashboard.py      # Dash app — UI layout, callbacks, direct function calls
├── ai_assistant.py   # Multi-provider AI (Groq → Gemini → Mistral), context builder, insights, chat
├── ml_models.py      # XGBoost forecasting, backtesting, Isolation Forest, z-score anomaly detection
├── data.py           # Synthetic dataset generator
├── assets/
│   └── style.css     # Custom CSS (auto-loaded by Dash)
├── requirements.txt
├── Dockerfile
└── .env              # API keys (not committed)
```

## Data note

All campaign data is synthetically generated for demonstration purposes. No real client data is used.
