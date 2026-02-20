---
name: stock-analyst-deploy
description: Build, validate, and deploy the stock-analyst API in istockpick, including recommendation/scoring-data endpoints, verbose mode, and construction-server runtime wiring.
---

# Stock Analyst Deploy Skill

Deploy and validate the current stock-analyst API implementation.

## Expected Endpoints

For production domain (`api.istockpick.ai`), ensure these are live:

1. `GET /health`
2. `POST /api/v1/agents/register`
3. `GET|POST /api/v1/recommendation`
4. `GET|POST /api/v1/scoring-data`
5. `GET /api/v1/weights`
6. `GET /api/v1/model-leaderboard`

## Recommendation Response Modes

`/api/v1/recommendation` supports `verbose` (and legacy alias `verborse`).
It also supports optional `model_name` for personalized model selection.

1. Default (`verbose=false` or omitted):
- Action-only payload: `{"recommendation":"BUY|HOLD|SELL"}`

2. `verbose=true`:
- Detailed payload with sub-sections:
- `stock_analysis`
- `sentiment_analysis`
- `ai_recommendation`
- `scoring_weights`
- `model_name`

## Scoring-Data Endpoint

`/api/v1/scoring-data` returns:

1. `price`
2. `snapshot` (raw market data)
3. `scoring_inputs` (raw scoring factors)
4. `scoring_weights`
5. metadata (`input`, `resolved_symbol`, `company`, `generated_at`)

Supports optional weights override:

1. GET: `weights` as JSON-encoded query string.
2. POST: `weights` as JSON object body.
3. Optional `model_name` (GET query or POST body) selects a named personalized model.

## Weights Discovery + Persistence

1. `GET /api/v1/weights` returns all modifiable keys with default/min/max and threshold rules.
2. Per-agent model persistence is stored in:
- `stock-analyst/data/weights.txt`
3. Model portfolio metadata is stored in:
- `stock-analyst/data/portfolio.txt`
4. Recommendation/scoring calls use weights in this order:
- Request `weights` override (if provided), and persist it for the selected model.
- Saved agent/model weights from `stock-analyst/data/weights.txt`.
- Hardcoded defaults.
5. Default-model behavior:
- If `model_name` is omitted, updates/read apply to the agent's `default` model.
- If `model_name` is provided and missing, API returns 404.

## Portfolio Leaderboard

`GET /api/v1/model-leaderboard` returns rows with:

1. `agent_name`
2. `portfolio_name`
3. `model_name`
4. `daily_change_pct`
5. `weekly_change_pct`
6. `monthly_change_pct`
7. display strings + delta integers (for UI coloring)

## Setup

Run from `stock-analyst/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn yfinance pydantic python-dotenv
```

Optional env for Alpaca provider:

```dotenv
APCA_API_KEY_ID=...
APCA_API_SECRET_KEY=...
ALPACA_DATA_BASE_URL=https://data.alpaca.markets
```

## Pre-Deploy Validation

1. Compile checks.

```bash
python -m py_compile stock_analyst/api.py
python -m py_compile stock_analyst/web_analyzer.py
python -m py_compile ../construction_server.py
python -m py_compile ../construction_server_fixed.py
```

2. Route checks for package app.

```bash
python - <<'PY'
from stock_analyst.api import app
paths = {r.path for r in app.routes}
required = {
    "/health",
    "/api/v1/agents/register",
    "/api/v1/recommendation",
    "/api/v1/recommendations",
    "/api/v1/scoring-data",
    "/api/v1/weights",
    "/api/v1/model-leaderboard",
}
print("api_routes_ok", required.issubset(paths))
PY
```

## Runtime Notes

1. If domain traffic is served by `construction_server.py` or `construction_server_fixed.py`, restart that process after changes.
2. If traffic is served by FastAPI directly, run:

```bash
cd stock-analyst
uvicorn stock_analyst.api:app --host 0.0.0.0 --port 8000
```

## Smoke Tests

1. Health.

```bash
curl "http://api.istockpick.ai/health"
```

2. Recommendation default (action-only).

```bash
curl "http://api.istockpick.ai/api/v1/recommendation?stock=AAPL&agent_name=agent-alpha&agent_token=REPLACE_WITH_TOKEN"
```

3. Recommendation verbose.

```bash
curl "http://api.istockpick.ai/api/v1/recommendation?stock=AAPL&agent_name=agent-alpha&agent_token=REPLACE_WITH_TOKEN&verbose=true"
```

4. Scoring data.

```bash
curl "http://api.istockpick.ai/api/v1/scoring-data?stock=AAPL&agent_name=agent-alpha&agent_token=REPLACE_WITH_TOKEN"
```

5. Scoring data with weights override.

```bash
curl "http://api.istockpick.ai/api/v1/scoring-data?stock=AAPL&agent_name=agent-alpha&agent_token=REPLACE_WITH_TOKEN&weights=%7B%22trend_bullish%22%3A20%2C%22action_buy_threshold%22%3A70%7D"
```

6. Weights metadata endpoint.

```bash
curl "http://api.istockpick.ai/api/v1/weights"
```

7. Model leaderboard endpoint.

```bash
curl "http://api.istockpick.ai/api/v1/model-leaderboard"
```

## Sample Scripts

Run from `stock-analyst/`.

1. Detail call sample.

```bash
python3 samples/istockpick_reco_detail.py
```

2. Multi-symbol scan sample.

```bash
python3 samples/istockpick_reco_scan.py
```

## Definition of Done

1. Runtime is serving updated code path (no stale entrypoint mismatch).
2. Recommendation endpoint honors `verbose` behavior.
3. Scoring-data endpoint is reachable and returns price + raw scoring inputs.
4. `/api/v1/weights` returns all modifiable keys + ranges.
5. Named model behavior works (`model_name` + default fallback).
6. `/api/v1/model-leaderboard` returns rows from portfolio/weights DB.
7. Authenticated calls succeed with registered agent credentials.
8. Compile checks pass for analyzer API and construction server entrypoints.
