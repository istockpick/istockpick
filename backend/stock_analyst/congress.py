"""Congressional trading analytics -- fetches Senate/House EFDS filings,
computes trade ROI, and produces seasonal summaries."""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_CACHE_DIR = _BACKEND_DIR / "data" / "congress_cache"
_CACHE_MAX_AGE_SECONDS = 86400  # 24 hours
_AMOUNT_MIDPOINTS = {
    "$1,001 - $15,000": 8000,
    "$15,001 - $50,000": 32500,
    "$50,001 - $100,000": 75000,
    "$100,001 - $250,000": 175000,
    "$250,001 - $500,000": 375000,
    "$500,001 - $1,000,000": 750000,
    "$1,000,001 - $5,000,000": 3000000,
    "$5,000,001 - $25,000,000": 15000000,
    "$25,000,001 - $50,000,000": 37500000,
    "Over $50,000,000": 75000000,
}


def _amount_midpoint(amount_range: str) -> float:
    if not amount_range:
        return 0.0
    cleaned = amount_range.strip()
    if cleaned in _AMOUNT_MIDPOINTS:
        return float(_AMOUNT_MIDPOINTS[cleaned])
    numbers = re.findall(r"[\d,]+", cleaned)
    if len(numbers) >= 2:
        lo = float(numbers[0].replace(",", ""))
        hi = float(numbers[1].replace(",", ""))
        return (lo + hi) / 2
    if numbers:
        return float(numbers[0].replace(",", ""))
    return 0.0


def _cache_path(year: int, chamber: str) -> Path:
    return _CACHE_DIR / f"{year}_{chamber.lower()}.json"


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


def _write_cache(path: Path, data: list) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2))
    tmp.replace(path)


def _normalize_trade(raw: dict, chamber: str) -> Optional[dict]:
    """Normalize a raw trade record from capitolgains into our standard format."""
    politician = (
        raw.get("member_name")
        or raw.get("first_name", "") + " " + raw.get("last_name", "")
        or raw.get("name", "")
    ).strip()
    if not politician:
        return None

    symbol = (raw.get("ticker") or raw.get("symbol") or "").strip().upper()
    action = (raw.get("transaction_type") or raw.get("type") or "").strip()
    date_str = (raw.get("transaction_date") or raw.get("date") or "").strip()
    amount_range = (raw.get("amount") or raw.get("amount_range") or "").strip()

    if not symbol or not date_str:
        return None

    # Determine quarter
    quarter = None
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        quarter = f"Q{(dt.month - 1) // 3 + 1}"
        year = dt.year
    except Exception:
        year = None

    return {
        "politician": politician,
        "chamber": chamber.title(),
        "symbol": symbol,
        "action": action,
        "date": date_str[:10] if len(date_str) >= 10 else date_str,
        "amount_range": amount_range,
        "amount_midpoint": _amount_midpoint(amount_range),
        "quarter": quarter,
        "year": year,
    }


def fetch_trades(year: int = None, chamber: str = "all") -> list[dict]:
    """Fetch congressional trades for a given year and chamber.
    Returns a list of normalized trade dicts.

    ``chamber`` can be 'senate', 'house', or 'all'.
    """
    if year is None:
        year = datetime.now().year

    chamber = (chamber or "all").lower()
    chambers_to_query = (
        ["senate", "house"] if chamber == "all" else [chamber]
    )

    all_trades = []
    for ch in chambers_to_query:
        cache = _cache_path(year, ch)
        cached = _read_cache(cache)
        if cached is not None:
            all_trades.extend(cached)
            continue

        trades = _fetch_from_provider(year, ch)
        _write_cache(cache, trades)
        all_trades.extend(trades)

    all_trades.sort(key=lambda t: t.get("date", ""), reverse=True)
    return all_trades


def _fetch_from_provider(year: int, chamber: str) -> list[dict]:
    """Try capitolgains first, fall back to an empty list with a warning."""
    try:
        return _fetch_via_capitolgains(year, chamber)
    except Exception as exc:
        logger.warning("capitolgains fetch failed for %s/%s: %s", year, chamber, exc)
        return []


def _fetch_via_capitolgains(year: int, chamber: str) -> list[dict]:
    """Pull PTR data via the capitolgains package."""
    try:
        from capitolgains import Congress
    except ImportError:
        logger.warning("capitolgains package not installed; returning empty trades")
        return []

    congress = Congress()
    raw_reports = []

    try:
        if chamber == "senate":
            raw_reports = congress.get_senate_disclosures(
                report_type="ptr",
                year=year,
            )
        elif chamber == "house":
            raw_reports = congress.get_house_disclosures(
                report_type="ptr",
                year=year,
            )
    except Exception as exc:
        logger.warning("capitolgains query error: %s", exc)
        return []

    if not isinstance(raw_reports, list):
        raw_reports = []

    normalized = []
    for raw in raw_reports:
        if not isinstance(raw, dict):
            continue
        trade = _normalize_trade(raw, chamber)
        if trade:
            normalized.append(trade)

    return normalized


def compute_trade_roi(trades: list[dict]) -> list[dict]:
    """Enrich trades with ROI by looking up the price at trade date and
    the current price via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available for ROI computation")
        return trades

    symbols = {t["symbol"] for t in trades if t.get("symbol")}
    price_cache: dict[str, float] = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info if hasattr(ticker, "fast_info") else {}
            current = getattr(info, "last_price", None) or info.get("lastPrice")
            if current:
                price_cache[symbol] = float(current)
        except Exception:
            pass

    enriched = []
    for trade in trades:
        t = dict(trade)
        symbol = t.get("symbol", "")
        current_price = price_cache.get(symbol)
        t["current_price"] = current_price

        # Attempt to get price at trade date
        trade_date = t.get("date", "")
        price_at_trade = None
        if symbol and trade_date:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=trade_date, period="5d")
                if hist is not None and not hist.empty:
                    price_at_trade = float(hist["Close"].iloc[0])
            except Exception:
                pass

        t["price_at_trade"] = price_at_trade

        if price_at_trade and current_price and price_at_trade > 0:
            t["roi_pct"] = round((current_price - price_at_trade) / price_at_trade * 100, 2)
        else:
            t["roi_pct"] = None

        if trade_date:
            try:
                dt = datetime.strptime(trade_date[:10], "%Y-%m-%d")
                t["holding_days"] = (datetime.now() - dt).days
            except Exception:
                t["holding_days"] = None
        else:
            t["holding_days"] = None

        enriched.append(t)

    return enriched


def seasonal_summary(trades: list[dict], year: int = None) -> dict:
    """Aggregate trades by quarter to produce seasonal patterns."""
    if year is None:
        year = datetime.now().year

    year_trades = [t for t in trades if t.get("year") == year]
    quarters = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}

    for trade in year_trades:
        q = trade.get("quarter")
        if q in quarters:
            quarters[q].append(trade)

    summary = {}
    for q, q_trades in quarters.items():
        roi_values = [t["roi_pct"] for t in q_trades if t.get("roi_pct") is not None]
        total_volume = sum(t.get("amount_midpoint", 0) for t in q_trades)
        buys = sum(1 for t in q_trades if "purchase" in (t.get("action") or "").lower())
        sells = sum(1 for t in q_trades if "sale" in (t.get("action") or "").lower())
        avg_roi = round(sum(roi_values) / len(roi_values), 2) if roi_values else None

        summary[q] = {
            "trade_count": len(q_trades),
            "buy_count": buys,
            "sell_count": sells,
            "avg_roi_pct": avg_roi,
            "total_estimated_volume": total_volume,
            "top_symbols": _top_symbols(q_trades, 5),
        }

    return {
        "year": year,
        "quarters": summary,
        "total_trades": len(year_trades),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def yearly_report(year: int = None, chamber: str = "all", top_n: int = 10) -> dict:
    """Yearly ROI report with top performers."""
    if year is None:
        year = datetime.now().year

    trades = fetch_trades(year=year, chamber=chamber)
    trades = compute_trade_roi(trades)

    # Aggregate by politician
    by_politician: dict[str, list] = {}
    for t in trades:
        name = t.get("politician", "Unknown")
        by_politician.setdefault(name, []).append(t)

    leaderboard = []
    for politician, ptrades in by_politician.items():
        roi_values = [t["roi_pct"] for t in ptrades if t.get("roi_pct") is not None]
        avg_roi = round(sum(roi_values) / len(roi_values), 2) if roi_values else None
        total_volume = sum(t.get("amount_midpoint", 0) for t in ptrades)
        chamber_val = ptrades[0].get("chamber", "Unknown") if ptrades else "Unknown"
        leaderboard.append({
            "politician": politician,
            "chamber": chamber_val,
            "trade_count": len(ptrades),
            "avg_roi_pct": avg_roi,
            "total_estimated_volume": total_volume,
            "top_symbols": _top_symbols(ptrades, 3),
        })

    leaderboard.sort(key=lambda x: x.get("avg_roi_pct") or -9999, reverse=True)

    seasonal = seasonal_summary(trades, year)

    return {
        "year": year,
        "chamber": chamber,
        "total_trades": len(trades),
        "top_performers": leaderboard[:top_n],
        "seasonal": seasonal["quarters"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _top_symbols(trades: list[dict], n: int) -> list[dict]:
    counts: dict[str, int] = {}
    for t in trades:
        sym = t.get("symbol", "")
        if sym:
            counts[sym] = counts.get(sym, 0) + 1
    sorted_symbols = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]
    return [{"symbol": s, "count": c} for s, c in sorted_symbols]
