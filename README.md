# Markedsinnsikt AI

Markedsinnsikt AI is a decision-support tool that transforms marketing performance data into actionable insights and recommendations across campaigns, channels, and clients.

The system combines structured data with AI reasoning to help marketing teams move from reporting to action — surfacing what is happening, why it might be happening, and what to do about it.

Built with Dash and Groq (llama-3.3-70b-versatile).

## Features

- **Multi-client filtering** — switch between clients, campaigns, and channels; all charts and insights update dynamically
- **KPI cards** — total spend, revenue, conversions, ROAS, CTR with week-over-week trend arrows
- **Performance charts** — ROAS by channel, conversions by campaign, spend distribution, weekly trend — each with a rule-based interpretation line
- **Anomaly notifications** — bell icon alerts for ROAS drops ≥25% and spend spikes ≥50%
- **AI Insights** — one-click structured analysis: summary, key insights, anomalies, and prioritised recommendations (powered by Groq)
- **AI Assistant** — multi-turn chat with full campaign context; clickable example questions to get started
- **Goal-aware analysis** — evaluates Brand Awareness on CPM/reach, Lead Gen on CPL, Direct Sales on ROAS, App Installs on CPI
- **Cross-client benchmarking** — compares selected client against portfolio averages
- **Audience segment analysis** — ranks segments by ROAS and conversions
- **Predictive signals** — flags 3+ consecutive weeks of declining ROAS and accelerating spend

## Live demo

**https://compete-positioning-device-explained.trycloudflare.com**

> Note: Cloudflare Tunnel URL may change on restart. Update this link if it does.

## Running with Docker

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed

### Setup

1. Copy the example env file and add your Groq API key:

   ```bash
   cp .env.example .env
   # edit .env and set GROQ_API_KEY=gsk_...
   ```

2. Build and start:

   ```bash
   docker build -t markedsinnsikt .
   docker run -p 8050:8050 --env-file .env markedsinnsikt
   ```

3. Open the dashboard at **http://localhost:8050**

## Running locally (without Docker)

### Prerequisites

- Python 3.11+

### Setup

1. Activate the virtual environment:

   ```bash
   source "/Users/mohamedradwan/Desktop/projects/Markedsinnsikt AI/.venv/bin/activate"
   ```

2. Install dependencies:

   ```bash
   pip3 install -r requirements.txt
   ```

3. Copy the env file and add your Groq API key:

   ```bash
   cp .env.example .env
   # edit .env and set GROQ_API_KEY=gsk_...
   ```

4. Start the dashboard:

   ```bash
   python dashboard.py
   ```

   Open the dashboard at **http://localhost:8050**

## Deploying on Render

1. Push the repo to GitHub
2. Create a new Web Service on [Render](https://render.com), connect the repo
3. Set **Language** to `Docker`
4. Add environment variable `GROQ_API_KEY` in the Render dashboard
5. Deploy — the app will be available at your Render URL

## Project structure

```
.
├── dashboard.py      # Dash frontend (all UI, callbacks, direct function calls)
├── ai_assistant.py   # Groq LLM integration (context builder, insights, chat)
├── data.py           # Synthetic dataset generator
├── api.py            # FastAPI backend (optional — kept for API-only use)
├── requirements.txt
├── Dockerfile
└── .env              # Groq API key (not committed)
```

## Data note

All campaign data is synthetically generated for demonstration purposes. No real client data is used.
