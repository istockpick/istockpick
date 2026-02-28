"""Polymarket prediction-market integration -- searches markets by ticker/keyword,
fetches live prices via CLOB API, and detects edge against iStockPick signals."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_CACHE_DIR = _BACKEND_DIR / "data" / "polymarket_cache"
_CACHE_MAX_AGE_SECONDS = 900  # 15 min

_GAMMA_API_BASE = "https://gamma-api.polymarket.com"
_CLOB_API_BASE = "https://clob.polymarket.com"
_POLYMARKET_EVENT_URL = "https://polymarket.com/event"

_HTTP_TIMEOUT = 15  # seconds

# Keywords that suggest a market is related to stock/crypto price movements
_PRICE_KEYWORDS = [
    "price", "hit", "reach", "above", "below", "rise", "fall", "drop",
    "crash", "surge", "rally", "close", "trade", "worth", "market cap",
    "all-time high", "ath", "bullish", "bearish", "moon",
]

# Well-known ticker-to-company mappings for search broadening
_TICKER_COMPANY_MAP = {
    "AAPL": "Apple",
    "GOOGL": "Google",
    "GOOG": "Google",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "META": "Meta",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "NFLX": "Netflix",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "DOGE-USD": "Dogecoin",
    "XRP-USD": "XRP",
}


# ---------------------------------------------------------------------------
# Caching (same pattern as congress.py)
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return _CACHE_DIR / f"{safe_key}.json"


def _is_cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < _CACHE_MAX_AGE_SECONDS


def _read_cache(path: Path) -> Optional[list]:
    if not _is_cache_fresh(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except Exception:
        return None


def _write_cache(path: Path, data) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

def search_markets(query: str, limit: int = 10, active: bool = True) -> list[dict]:
    """Search Gamma API for prediction markets matching a keyword/ticker.

    Returns a list of event dicts, each containing nested markets with prices
    and volumes.
    """
    cache_key = f"search_{query.lower()}_{limit}_{active}"
    cached = _read_cache(_cache_path(cache_key))
    if cached is not None:
        return cached

    params = {
        "limit": limit,
        "search": query,
    }
    if active:
        params["active"] = "true"
        params["closed"] = "false"

    try:
        resp = httpx.get(
            f"{_GAMMA_API_BASE}/events",
            params=params,
            timeout=_HTTP_TIMEOUT,
            headers={"User-Agent": "iStockPick/1.0"},
        )
        resp.raise_for_status()
        events = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning("Gamma API search failed (%s): HTTP %s", query, exc.response.status_code)
        return []
    except Exception as exc:
        logger.warning("Gamma API search failed (%s): %s", query, exc)
        return []

    if not isinstance(events, list):
        events = []

    results = _normalize_events(events)
    _write_cache(_cache_path(cache_key), results)
    return results


def get_market_price(token_id: str) -> dict:
    """Get current midpoint price and order book summary from CLOB API."""
    result = {"token_id": token_id, "midpoint": None, "best_bid": None, "best_ask": None}

    # Fetch midpoint
    try:
        resp = httpx.get(
            f"{_CLOB_API_BASE}/midpoint",
            params={"token_id": token_id},
            timeout=_HTTP_TIMEOUT,
            headers={"User-Agent": "iStockPick/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
        mid = data.get("mid")
        if mid is not None:
            result["midpoint"] = float(mid)
    except Exception as exc:
        logger.warning("CLOB midpoint fetch failed (token=%s): %s", token_id, exc)

    # Fetch order book top-of-book
    try:
        resp = httpx.get(
            f"{_CLOB_API_BASE}/book",
            params={"token_id": token_id},
            timeout=_HTTP_TIMEOUT,
            headers={"User-Agent": "iStockPick/1.0"},
        )
        resp.raise_for_status()
        book = resp.json()
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        if bids:
            result["best_bid"] = float(bids[0].get("price", 0))
        if asks:
            result["best_ask"] = float(asks[0].get("price", 0))
        result["order_book"] = {
            "bids": bids[:5],
            "asks": asks[:5],
        }
    except Exception as exc:
        logger.warning("CLOB book fetch failed (token=%s): %s", token_id, exc)

    return result


def get_market_detail(condition_id: str) -> dict:
    """Get full market detail from Gamma API by condition_id."""
    cache_key = f"detail_{condition_id}"
    cached = _read_cache(_cache_path(cache_key))
    if cached is not None:
        return cached

    try:
        resp = httpx.get(
            f"{_GAMMA_API_BASE}/markets",
            params={"condition_id": condition_id},
            timeout=_HTTP_TIMEOUT,
            headers={"User-Agent": "iStockPick/1.0"},
        )
        resp.raise_for_status()
        markets = resp.json()
    except Exception as exc:
        logger.warning("Gamma API market detail failed (%s): %s", condition_id, exc)
        return {}

    if isinstance(markets, list) and markets:
        market = markets[0]
    elif isinstance(markets, dict):
        market = markets
    else:
        return {}

    result = _normalize_market(market)
    _write_cache(_cache_path(cache_key), result)
    return result


# ---------------------------------------------------------------------------
# Edge Detection
# ---------------------------------------------------------------------------

def _signal_to_probability(signal: str, confidence: float) -> float:
    """Convert an iStockPick signal + confidence into an implied probability
    that the asset's price will go UP.

    BUY@80 confidence  => 0.80 probability of price increase
    SELL@80 confidence => 0.20 probability of price increase (= 0.80 of decline)
    HOLD               => ~0.50 neutral
    """
    # Normalize confidence to 0-1 range
    conf = max(0.0, min(100.0, confidence)) / 100.0

    signal_upper = (signal or "HOLD").upper()
    if signal_upper == "BUY":
        return conf
    elif signal_upper == "SELL":
        return 1.0 - conf
    else:  # HOLD
        return 0.50


def _classify_market_direction(title: str) -> str:
    """Classify whether a market question is asking about a POSITIVE outcome
    (price going up / reaching a target) or a NEGATIVE outcome (crash / decline).

    Returns "positive", "negative", or "neutral".
    """
    title_lower = (title or "").lower()

    negative_terms = ["crash", "fall", "drop", "decline", "below", "under", "lose", "down", "bear"]
    positive_terms = ["reach", "hit", "above", "over", "rise", "surge", "rally", "up", "bull", "moon", "ath", "high"]

    neg_count = sum(1 for term in negative_terms if term in title_lower)
    pos_count = sum(1 for term in positive_terms if term in title_lower)

    if neg_count > pos_count:
        return "negative"
    if pos_count > neg_count:
        return "positive"
    return "neutral"


def _is_price_relevant(title: str, symbol: str) -> bool:
    """Check if a market title is related to price movements of the given asset."""
    title_lower = (title or "").lower()
    symbol_lower = symbol.lower()

    # Must mention the symbol or a related company name
    company = _TICKER_COMPANY_MAP.get(symbol.upper(), "").lower()
    has_symbol = symbol_lower in title_lower
    has_company = bool(company) and company in title_lower

    if not has_symbol and not has_company:
        return False

    # Should have at least one price-related keyword
    for kw in _PRICE_KEYWORDS:
        if kw in title_lower:
            return True

    # Also accept if the title contains a dollar amount (price target)
    if "$" in title_lower:
        return True

    return False


def find_opportunities(
    symbol: str,
    signal: str,
    confidence: float,
    limit: int = 5,
) -> list[dict]:
    """Search for Polymarket markets related to `symbol`, compare iStockPick
    signal against market implied probability, and return ranked opportunities
    with edge calculation.

    Returns: list of opportunity dicts sorted by edge (descending).
    """
    # Broaden search with multiple queries
    queries = [symbol]
    company = _TICKER_COMPANY_MAP.get(symbol.upper())
    if company:
        queries.append(company)
    queries.append(f"{symbol} price")

    # Collect all unique markets from search results
    seen_ids = set()
    all_markets = []
    for query in queries:
        events = search_markets(query, limit=20)
        for event in events:
            for market in event.get("markets", []):
                market_id = market.get("condition_id") or market.get("question_id") or market.get("title")
                if market_id and market_id not in seen_ids:
                    seen_ids.add(market_id)
                    market["_event_title"] = event.get("title", "")
                    market["_event_slug"] = event.get("slug", "")
                    all_markets.append(market)

    # Filter to price-relevant markets
    our_up_prob = _signal_to_probability(signal, confidence)
    opportunities = []

    for market in all_markets:
        title = market.get("question") or market.get("title") or market.get("_event_title", "")
        if not _is_price_relevant(title, symbol):
            continue

        # Get the market price (YES token price = implied probability)
        market_price = _extract_market_price(market)
        if market_price is None:
            continue

        direction = _classify_market_direction(title)

        # Calculate edge
        if direction == "negative":
            # Market asks about a negative event (crash, drop).
            # Our implied prob of negative event = 1 - our_up_prob
            our_prob = 1.0 - our_up_prob
            edge = abs(our_prob - market_price)
            if our_prob < market_price:
                suggested_side = "NO"
                edge_signed = market_price - our_prob
            else:
                suggested_side = "YES"
                edge_signed = our_prob - market_price
        else:
            # Market asks about a positive event (price reaching a target, going up).
            our_prob = our_up_prob
            edge = abs(our_prob - market_price)
            if our_prob > market_price:
                suggested_side = "YES"
                edge_signed = our_prob - market_price
            else:
                suggested_side = "NO"
                edge_signed = market_price - our_prob

        # Build market URL
        slug = market.get("_event_slug") or ""
        event_url = f"{_POLYMARKET_EVENT_URL}/{slug}" if slug else ""

        opportunities.append({
            "market_title": title,
            "market_url": event_url,
            "market_price": round(market_price, 4),
            "our_implied_prob": round(our_prob, 4),
            "edge": round(edge_signed, 4),
            "suggested_side": suggested_side,
            "direction": direction,
            "condition_id": market.get("condition_id", ""),
            "token_id": _extract_token_id(market),
            "volume_24h": _safe_float(market.get("volume24hr") or market.get("volume")),
            "liquidity": _safe_float(market.get("liquidity")),
        })

    # Sort by edge descending
    opportunities.sort(key=lambda o: o["edge"], reverse=True)
    return opportunities[:limit]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_events(events: list) -> list[dict]:
    """Normalize raw Gamma API event objects into a consistent structure."""
    results = []
    for event in events:
        if not isinstance(event, dict):
            continue
        markets_raw = event.get("markets") or []
        markets = [_normalize_market(m) for m in markets_raw if isinstance(m, dict)]
        results.append({
            "title": event.get("title", ""),
            "slug": event.get("slug", ""),
            "description": event.get("description", ""),
            "active": event.get("active", True),
            "closed": event.get("closed", False),
            "liquidity": _safe_float(event.get("liquidity")),
            "volume": _safe_float(event.get("volume")),
            "markets": markets,
            "created_at": event.get("createdAt") or event.get("created_at", ""),
            "end_date": event.get("endDate") or event.get("end_date", ""),
        })
    return results


def _normalize_market(market: dict) -> dict:
    """Normalize a raw Gamma API market object."""
    tokens = market.get("clobTokenIds") or market.get("tokens") or []
    outcomes = market.get("outcomes") or market.get("outcomePrices") or []

    # Parse outcome prices if they're a JSON string
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except (json.JSONDecodeError, TypeError):
            outcomes = []

    return {
        "question": market.get("question") or market.get("title", ""),
        "condition_id": market.get("conditionId") or market.get("condition_id", ""),
        "question_id": market.get("questionId") or market.get("question_id", ""),
        "slug": market.get("slug", ""),
        "active": market.get("active", True),
        "closed": market.get("closed", False),
        "outcomes": outcomes,
        "outcome_prices": _parse_outcome_prices(market),
        "tokens": tokens if isinstance(tokens, list) else [],
        "volume": _safe_float(market.get("volume")),
        "volume24hr": _safe_float(market.get("volume24hr")),
        "liquidity": _safe_float(market.get("liquidity")),
        "end_date": market.get("endDate") or market.get("end_date", ""),
        "description": market.get("description", ""),
    }


def _parse_outcome_prices(market: dict) -> list[float]:
    """Extract outcome prices from a market dict. Returns [yes_price, no_price]."""
    raw = market.get("outcomePrices") or market.get("outcome_prices") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(raw, list):
        return [_safe_float(p) for p in raw]
    return []


def _extract_market_price(market: dict) -> Optional[float]:
    """Extract the YES token price (implied probability) from a market dict."""
    prices = market.get("outcome_prices") or _parse_outcome_prices(market)
    if prices and len(prices) >= 1 and prices[0] is not None:
        price = prices[0]
        if 0.0 <= price <= 1.0:
            return price

    # Try bestBid/bestAsk midpoint or last trade price
    best_bid = _safe_float(market.get("bestBid"))
    best_ask = _safe_float(market.get("bestAsk"))
    if best_bid is not None and best_ask is not None:
        mid = (best_bid + best_ask) / 2.0
        if 0.0 <= mid <= 1.0:
            return mid

    return None


def _extract_token_id(market: dict) -> str:
    """Extract the YES token ID from a market dict."""
    tokens = market.get("tokens") or []
    if isinstance(tokens, list):
        # clobTokenIds is typically [yes_token, no_token]
        if tokens and isinstance(tokens[0], str):
            return tokens[0]
        for t in tokens:
            if isinstance(t, dict):
                outcome = (t.get("outcome") or "").lower()
                if outcome == "yes":
                    return t.get("token_id", "")
        if tokens and isinstance(tokens[0], dict):
            return tokens[0].get("token_id", "")
    return ""


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
