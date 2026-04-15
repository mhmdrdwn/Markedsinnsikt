# Markedsinnsikt AI

AI-powered marketing analytics assistant for multi-client, multi-channel campaigns. Built with FastAPI, Dash, and Groq.

## Live demo

**https://compete-positioning-device-explained.trycloudflare.com**

> Note: Cloudflare Tunnel URL may change on restart. Update this link if it does.

## Services

| Service | URL | Description |
|---|---|---|
| Dashboard (public) | https://compete-positioning-device-explained.trycloudflare.com | Dash frontend |
| Dashboard (local) | http://localhost:8050 | Dash frontend |
| API | http://localhost:8000 | FastAPI backend |
| API Docs | http://localhost:8000/docs | Interactive API explorer |

## Running with Docker

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose installed

### Setup

1. Copy the example env file and add your Groq API key:

   ```bash
   cp .env.example .env
   # then edit .env and set GROQ_API_KEY=gsk_...
   ```

2. Build and start both services:

   ```bash
   docker compose up --build
   ```

3. Open the dashboard at **http://localhost:8050**

### Stopping

```bash
docker compose down
```

### Rebuilding after code changes

```bash
docker compose up --build
```

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
   # then edit .env and set GROQ_API_KEY=gsk_...
   ```

### Start the API (Terminal 1)

```bash
uvicorn api:app --reload
```

### Start the Dashboard (Terminal 2)

```bash
python dashboard.py
```

Open the dashboard at **http://localhost:8050**

## Project structure

```
.
├── api.py            # FastAPI backend (data, KPIs, charts, AI endpoints)
├── dashboard.py      # Dash frontend (consumes the API)
├── ai_assistant.py   # Groq LLM integration
├── data.py           # Synthetic dataset generator
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env              # Groq API key (not committed)
```

## AI features

- **Generate Insights** — automated performance analysis across campaigns and channels
- **Ask the AI Analyst** — free-form Q&A about your campaign data

Both features require a [Groq API key](https://console.groq.com). Enter it in the sidebar or set `GROQ_API_KEY` in `.env`.
