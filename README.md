---
layout: default
title: iStockPick
permalink: /
---

# iStockPick

AI-powered stock analysis platform that delivers BUY / SELL / HOLD recommendations by combining technical indicators, fundamental metrics, social-media sentiment, and AI scoring.

**Production URL:** `https://api.istockpick.ai`

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Scoring System](#scoring-system)
- [Sample Scripts](#sample-scripts)
- [Deployment](#deployment)
- [License](#license)

---

## Features

- **Technical Analysis** — RSI, MACD, Bollinger Bands, moving-average crossovers, volume ratios
- **Fundamental Analysis** — P/E, PEG, ROE, debt ratios, revenue & earnings growth
- **Sentiment Analysis** — X/Twitter, Reddit (r/wallstreetbets, r/stocks, …), and RSS news feeds (WSJ, CNBC, Reuters, Financial Times)
- **AI Sentiment Scoring** — OpenAI GPT-4o-mini evaluates aggregated sentiment
- **Multi-Agent System** — Token-based agent registration with per-agent, per-model weight customization
- **Portfolio Tracking** — Daily / weekly / monthly performance tracking with a model leaderboard
- **Batch Recommendations** — Analyze up to 25 symbols in a single request
- **Verbose & Simple Modes** — Action-only responses or full detailed breakdowns

---

## Architecture

```
┌────────────────────────────────────┐
│  construction_server_fixed.py      │  Port 8001 — HTML UI + API proxy
└──────────────┬─────────────────────┘
               │ proxies /api/* requests
               ▼
┌────────────────────────────────────┐
│  stock_analyst/api.py (FastAPI)    │  Port 8000 — REST API
├────────────────────────────────────┤
│  web_analyzer.py                   │  Orchestrates analysis pipeline
│  technical.py                      │  Technical indicators
│  fundamental.py                    │  Fundamental metrics
│  config.py                         │  Environment & constants
└──────────────┬─────────────────────┘
               │
   ┌───────────┼───────────┐
   ▼           ▼           ▼
 yfinance   Alpaca API   Stooq        ← Market data (multi-provider fallback)
               │
   ┌───────────┼───────────┐
   ▼           ▼           ▼
 X/Twitter   Reddit     RSS Feeds     ← Sentiment sources
               │
               ▼
          OpenAI API                   ← AI sentiment scoring
```

Data is persisted in flat JSON files under `stock-analyst/data/`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Web Server | Python `http.server` (construction server) |
| Market Data | yfinance, Alpaca Markets API, Stooq |
| Sentiment | X/Twitter API v2, Reddit API, RSS feeds |
| AI Scoring | OpenAI API (GPT-4o-mini) |
| Validation | SEC.gov (ticker lookup) |
| Data Processing | pandas, numpy |
| Persistence | File-based JSON (`agents_db.txt`, `weights.txt`, `portfolio.txt`) |

---

## Project Structure

```
istockpick/
├── README.md
├── SKILL.md                         # Deployment playbook
├── construction_server.py           # HTTP server (legacy)
├── construction_server_fixed.py     # HTTP server (production)
│
└── stock-analyst/
    ├── stock_analyst/
    │   ├── __init__.py
    │   ├── api.py                   # FastAPI REST API
    │   ├── config.py                # Environment variables & constants
    │   ├── fundamental.py           # Fundamental analysis
    │   ├── technical.py             # Technical indicators
    │   └── web_analyzer.py          # Analysis orchestrator
    │
    ├── scripts/
    │   ├── process_tweets_fixed.py
    │   └── movers_catalyst_fixed.py
    │
    ├── samples/
    │   ├── istockpick_reco_scan.py   # Scan S&P 500 / DOW / NASDAQ
    │   └── istockpick_reco_detail.py # Detailed single-symbol analysis
    │
    └── data/
        ├── agents_db.txt            # Agent credentials
        ├── weights.txt              # Per-agent model weights
        └── portfolio.txt            # Portfolio positions & performance
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- (Optional) Alpaca, Twitter/X, and OpenAI API keys for full functionality

### Installation

```bash
cd stock-analyst
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn yfinance pydantic python-dotenv pandas numpy
```

### Running the API Server

```bash
cd stock-analyst
uvicorn stock_analyst.api:app --host 0.0.0.0 --port 8000
```

### Running the Web Server (HTML UI + API Proxy)

```bash
python3 construction_server_fixed.py
# Serves on http://0.0.0.0:8001
```

---

## Configuration

Create a `.env` file in the project root (or set environment variables directly):

```dotenv
# Alpaca Market Data (optional — falls back to yfinance/Stooq)
APCA_API_KEY_ID=...
APCA_API_SECRET_KEY=...
ALPACA_DATA_BASE_URL=https://data.alpaca.markets

# Twitter / X Sentiment
TWITTER_BEARER_TOKEN=...
# or
X_BEARER_TOKEN=...

# OpenAI AI Sentiment Scoring
OPENAI_API_KEY=...
OPENAI_MEDIA_MODEL=gpt-4o-mini        # default

# News (optional)
NEWS_API_KEY=...

# Runtime Paths (optional)
STOCK_ANALYST_PATH=...                  # custom path to stock-analyst/
ANALYSIS_RUNTIME_PYTHON=...             # python interpreter override
```

Alpaca credentials can also be placed in `~/.configuration/alpaca/credentials.json`.

---

## API Reference

### Public Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/status` | Server status |
| `GET` | `/lookup?q=<ticker>` | Stock symbol lookup |
| `GET` | `/analyze?q=<ticker>` | Web-based analysis page |

### Authenticated Endpoints

All authenticated endpoints require `agent_name` and `agent_token`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/agents/register` | Register a new agent |
| `GET` / `POST` | `/api/v1/recommendation` | Single-stock recommendation |
| `GET` / `POST` | `/api/v1/recommendations` | Batch recommendations (max 25) |
| `GET` / `POST` | `/api/v1/scoring-data` | Raw scoring inputs & weights |
| `GET` | `/api/v1/weights` | List all modifiable scoring weights with ranges |
| `GET` | `/api/v1/model-leaderboard` | Portfolio performance leaderboard |
| `GET` / `POST` | `/api/v1/portfolio` | Read / update portfolio positions |

### Recommendation Response Modes

**Default** (`verbose=false`):
```json
{ "recommendation": "BUY" }
```

**Verbose** (`verbose=true`):
```json
{
  "stock_analysis": { ... },
  "sentiment_analysis": { ... },
  "ai_recommendation": { ... },
  "scoring_weights": { ... },
  "model_name": "default"
}
```

### Scoring-Data Response

```json
{
  "price": 185.50,
  "snapshot": { ... },
  "scoring_inputs": { ... },
  "scoring_weights": { ... },
  "input": "AAPL",
  "resolved_symbol": "AAPL",
  "company": "Apple Inc.",
  "generated_at": "2026-02-21T12:00:00Z"
}
```

Supports optional `weights` override (JSON object) and `model_name` selection.

---

## Scoring System

Recommendations are computed from a configurable weighted scoring model:

| Weight Key | Default | Description |
|---|---|---|
| `base_score` | 50.0 | Starting score |
| `trend_bullish` | 15.0 | Bonus for bullish trend |
| `trend_bearish` | 15.0 | Penalty for bearish trend |
| `high_volume_bonus` | 8.0 | Bonus when volume ratio exceeds threshold |
| `ma_bullish_bonus` | 7.0 | Bonus for bullish MA crossover |
| `ma_bearish_penalty` | 7.0 | Penalty for bearish MA crossover |
| `price_above_ma_bonus` | 5.0 | Bonus when price is above 50-day MA |
| `price_below_ma_penalty` | 5.0 | Penalty when price is below 50-day MA |
| `volume_ratio_threshold` | 1.5 | Volume ratio trigger |
| `action_buy_threshold` | 65.0 | Score >= this triggers BUY |
| `action_sell_threshold` | 35.0 | Score <= this triggers SELL |

Scores are clamped to the 0–100 range. Values between the buy and sell thresholds result in a HOLD.

Weight resolution order: request override > saved agent/model weights > defaults.

---

## Sample Scripts

Run from the `stock-analyst/` directory:

```bash
# Detailed recommendation for specific symbols
python3 samples/istockpick_reco_detail.py

# Scan S&P 500 / DOW / NASDAQ tickers
python3 samples/istockpick_reco_scan.py
```

---

## Deployment

The production instance runs at `api.istockpick.ai` behind an Nginx reverse proxy.

```bash
# Validate before deploying
python -m py_compile stock_analyst/api.py
python -m py_compile stock_analyst/web_analyzer.py
python -m py_compile ../construction_server_fixed.py

# Start the FastAPI backend
cd stock-analyst
uvicorn stock_analyst.api:app --host 0.0.0.0 --port 8000

# Or start the construction server (serves HTML UI + proxies API)
python3 construction_server_fixed.py   # port 8001
```

See `SKILL.md` for the full deployment playbook including smoke tests.

---

## License

Private — All rights reserved.
