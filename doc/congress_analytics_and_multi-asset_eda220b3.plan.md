---
name: Congress Analytics and Multi-Asset
overview: "Add two major features: (1) Congressional trading analytics with seasonal ROI reporting using the `capitolgains` Python package to scrape official Senate/House EFDS filings, and (2) full BUY/SELL/HOLD recommendation pipeline for crypto, options, and futures via yfinance."
todos:
  - id: congress-module
    content: Create backend/stock_analyst/congress.py with trade fetching via capitolgains, ROI computation, and seasonal analysis
    status: completed
  - id: crypto-module
    content: Create backend/stock_analyst/crypto.py with crypto snapshot and scoring functions
    status: completed
  - id: futures-module
    content: Create backend/stock_analyst/futures.py with futures snapshot and scoring functions
    status: completed
  - id: options-module
    content: Create backend/stock_analyst/options.py with options chain, snapshot, and scoring functions
    status: completed
  - id: web-analyzer-update
    content: Update web_analyzer.py to route generate_full_analysis() by asset_type
    status: completed
  - id: server-endpoints
    content: "Update server.py: add congress endpoints, asset_type param, symbol resolution for crypto/futures/options, asset-type weight defaults"
    status: completed
  - id: api-endpoints
    content: "Update api.py: add congress endpoints, asset_type param, options chain endpoint"
    status: completed
  - id: init-exports
    content: Update __init__.py to export new modules
    status: completed
  - id: deps
    content: Add capitolgains to pyproject.toml and requirements.txt, run uv lock
    status: completed
  - id: frontend-update
    content: "Update frontend/index.html: congress trading card, asset type selector, options chain display"
    status: completed
  - id: update-docs
    content: Update README.md and SKILL.md with new features, endpoints, and asset types
    status: completed
isProject: false
---

# Congress Analytics and Multi-Asset Support

## Feature 1: Congressional Trading Analytics

### Data Source

Use the open-source **`capitolgains`** Python package (`pip install capitolgains`) which scrapes official government EFDS filings from senate.gov and house.gov. Supports Senate records from 2012+ and House from 1995+, including Periodic Transaction Reports (PTRs) with buy/sell/exchange details.

### New Module: `backend/stock_analyst/congress.py`

Core functions:

- `fetch_trades(year, chamber)` -- pull PTRs from Senate and/or House via capitolgains, cache results in `backend/data/congress_cache/`
- `compute_trade_roi(trades)` -- for each buy, find the matching sell (or use current price for open positions) via yfinance to compute realized/unrealized ROI
- `seasonal_summary(trades, year)` -- aggregate trades and ROI by quarter (Q1-Q4), producing seasonal patterns
- `yearly_report(year)` -- top traders by ROI, best/worst picks, aggregate stats
- `top_performers(n)` -- leaderboard of politicians by annualized ROI

Data model for a processed trade:

```python
{
    "politician": "Nancy Pelosi",
    "chamber": "House",
    "symbol": "AAPL",
    "action": "Purchase",
    "date": "2025-03-15",
    "amount_range": "$100,001 - $250,000",
    "price_at_trade": 185.50,
    "current_price": 210.00,
    "roi_pct": 13.2,
    "holding_days": 120,
    "quarter": "Q1",
    "year": 2025
}
```

### New API Endpoints

Add to both [backend/server.py](backend/server.py) and [backend/stock_analyst/api.py](backend/stock_analyst/api.py):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/congress/trades` | Recent congressional trades. Query params: `year`, `chamber` (senate/house/all), `symbol`, `politician` |
| `GET` | `/api/v1/congress/roi` | Yearly ROI report. Query params: `year`, `chamber`, `top_n` |
| `GET` | `/api/v1/congress/seasonal` | Seasonal breakdown (Q1-Q4 aggregates). Query params: `year`, `chamber` |

All public endpoints (no agent auth required).

### Frontend: Congress Section

Add a new card to [frontend/index.html](frontend/index.html) below the leaderboard:

- **Congress Trading card** with a table showing recent trades (politician, symbol, action, date, ROI)
- **Seasonal ROI chart** -- simple bar/grid showing Q1-Q4 ROI by year
- Year selector dropdown
- Chamber filter (Senate / House / All)

### Caching

Congressional data changes infrequently (filings have 30-45 day lag). Cache fetched/processed data in `backend/data/congress_cache/` as JSON files keyed by `{year}_{chamber}.json`. Re-fetch when file is older than 24 hours.

---

## Feature 2: Crypto, Options, and Futures Support

### Asset Type Classification

Introduce an `asset_type` enum used throughout the pipeline:

```python
class AssetType(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    OPTION = "option"
    FUTURE = "future"
```

### Symbol Resolution

Update `_resolve_symbol_from_input()` in [backend/server.py](backend/server.py) and the lookup logic in [backend/stock_analyst/api.py](backend/stock_analyst/api.py):

- **Crypto**: detect `-USD` suffix or known prefixes (`BTC`, `ETH`, `SOL`, etc.) and resolve to yfinance format `BTC-USD`. Skip SEC validation for crypto symbols.
- **Futures**: detect `=F` suffix (e.g., `ES=F`, `GC=F`, `CL=F`) or known contract names. Skip SEC validation.
- **Options**: accept underlying symbol; option chain data is fetched separately via `yf.Ticker(symbol).option_chain(date)`.
- **Stocks**: existing logic unchanged (SEC validation + yfinance search).

Auto-detection order: check futures suffix -> check crypto pattern -> fall back to stock lookup.

### New Module: `backend/stock_analyst/crypto.py`

- `get_crypto_snapshot(symbol)` -- price, 24h volume, 24h change, market cap via yfinance `BTC-USD` style tickers
- `get_crypto_sentiment(snapshot, weights)` -- adapted scoring: uses momentum, volume spikes, and trend (no fundamental P/E ratios)
- Crypto-specific default weights (higher volatility thresholds)

### New Module: `backend/stock_analyst/futures.py`

- `get_futures_snapshot(symbol)` -- price, volume, open interest, contract expiry via yfinance `XX=F` tickers
- `get_futures_sentiment(snapshot, weights)` -- adapted scoring: trend-heavy, volume-ratio-heavy (no dividend or P/E metrics)
- Futures-specific default weights

### New Module: `backend/stock_analyst/options.py`

- `get_options_chain(symbol, expiry_date)` -- calls/puts via `yf.Ticker(symbol).option_chain()`
- `get_options_snapshot(symbol)` -- underlying price + nearest-expiry chain summary (max pain, put/call ratio, IV rank)
- `get_options_sentiment(snapshot, weights)` -- scoring based on put/call ratio, IV percentile, and underlying trend
- Options-specific default weights

### Pipeline Integration

Update [backend/stock_analyst/web_analyzer.py](backend/stock_analyst/web_analyzer.py):

- `generate_full_analysis(symbol, weights, asset_type)` -- add `asset_type` parameter (default `"stock"` for backward compatibility)
- Route to the appropriate snapshot/sentiment functions based on asset type
- Media/social analysis stays the same across all asset types (X/Reddit/news mentions exist for crypto and futures too)

Update scoring weights in [backend/server.py](backend/server.py):

- `SCORING_WEIGHT_DEFAULTS` stays as the stock default
- Add `CRYPTO_WEIGHT_DEFAULTS`, `FUTURES_WEIGHT_DEFAULTS`, `OPTIONS_WEIGHT_DEFAULTS` with tuned values (e.g., crypto has higher volatility thresholds)
- Weight selection: request override > saved agent weights > asset-type defaults

### API Changes

Add optional `asset_type` parameter to existing endpoints:

- `GET /api/v1/recommendation?stock=BTC-USD&asset_type=crypto&...`
- `GET /api/v1/scoring-data?stock=ES=F&asset_type=future&...`
- `POST /api/v1/recommendation` body: `{"stock": "BTC-USD", "asset_type": "crypto", ...}`

New endpoint for options chains:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/options/chain` | Options chain data. Query params: `symbol`, `expiry` |

Default `asset_type` is `"stock"` so all existing calls are backward-compatible.

### Frontend Updates

Update [frontend/index.html](frontend/index.html):

- Add an **asset type selector** (Stock / Crypto / Options / Futures) next to the lookup input
- Pass the selected asset type in the fetch calls to `/lookup` and `/analyze`
- For options: show a chain summary card below the analysis (calls/puts table, put/call ratio, IV)
- For crypto/futures: same analysis card layout but with adapted labels (e.g., "24h Volume" instead of "Daily Volume")

### SEC Validation Bypass

Update `lookup_public_stock()` in [backend/server.py](backend/server.py) to skip SEC ticker validation for crypto (`*-USD`), futures (`*=F`), and options (underlying is still a stock, so validation still applies to the underlying).

---

## Dependencies

Add to [backend/pyproject.toml](backend/pyproject.toml) and [backend/requirements.txt](backend/requirements.txt):

```
capitolgains
```

yfinance already covers crypto, options, and futures -- no new dependency needed for that.

---

## Files Changed / Created

| File | Action |
|------|--------|
| `backend/stock_analyst/congress.py` | **Create** -- congressional trade fetching, ROI computation, seasonal analysis |
| `backend/stock_analyst/crypto.py` | **Create** -- crypto snapshot and scoring |
| `backend/stock_analyst/futures.py` | **Create** -- futures snapshot and scoring |
| `backend/stock_analyst/options.py` | **Create** -- options chain, snapshot, and scoring |
| `backend/stock_analyst/web_analyzer.py` | **Edit** -- add `asset_type` routing in `generate_full_analysis()` |
| `backend/server.py` | **Edit** -- new congress endpoints, asset_type param on existing endpoints, symbol resolution updates, SEC bypass, asset-type weight defaults |
| `backend/stock_analyst/api.py` | **Edit** -- same endpoint additions as server.py |
| `backend/stock_analyst/__init__.py` | **Edit** -- export new modules |
| `backend/pyproject.toml` | **Edit** -- add `capitolgains` dependency |
| `backend/requirements.txt` | **Edit** -- add `capitolgains` |
| `frontend/index.html` | **Edit** -- congress card, asset type selector, options chain display |
| `README.md` | **Edit** -- document new features, endpoints, asset types |
| `SKILL.md` | **Edit** -- add new endpoints to expected endpoints and smoke tests |
