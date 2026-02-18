import csv
import io
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Optional

_SYMBOL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9\.\-]{0,9}$")
_MAX_HTTP_RESPONSE_BYTES = 2_000_000
_DEFAULT_ALPACA_CREDS_PATH = os.path.expanduser("~/.configuration/alpaca/credentionals.json")
_DEFAULT_ALPACA_CREDS_ALT_PATH = os.path.expanduser("~/.configuration/alpaca/credentials.json")
logger = logging.getLogger(__name__)

DEFAULT_SCORING_WEIGHTS = {
    "base_score": 50.0,
    "trend_bullish": 15.0,
    "trend_bearish": 15.0,
    "high_volume_bonus": 8.0,
    "ma_bullish_bonus": 7.0,
    "ma_bearish_penalty": 7.0,
    "price_above_ma_bonus": 5.0,
    "price_below_ma_penalty": 5.0,
    "volume_ratio_threshold": 1.5,
    "sentiment_buy_threshold": 65.0,
    "sentiment_sell_threshold": 35.0,
    "action_buy_threshold": 65.0,
    "action_sell_threshold": 35.0,
}

_WEIGHT_LIMITS = {
    "base_score": (0.0, 100.0),
    "trend_bullish": (0.0, 100.0),
    "trend_bearish": (0.0, 100.0),
    "high_volume_bonus": (0.0, 100.0),
    "ma_bullish_bonus": (0.0, 100.0),
    "ma_bearish_penalty": (0.0, 100.0),
    "price_above_ma_bonus": (0.0, 100.0),
    "price_below_ma_penalty": (0.0, 100.0),
    "volume_ratio_threshold": (0.1, 10.0),
    "sentiment_buy_threshold": (0.0, 100.0),
    "sentiment_sell_threshold": (0.0, 100.0),
    "action_buy_threshold": (0.0, 100.0),
    "action_sell_threshold": (0.0, 100.0),
}


def _fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "iStockPick/1.0 (admin@istockpick.ai)"})
    with urllib.request.urlopen(req, timeout=8) as response:
        body = response.read(_MAX_HTTP_RESPONSE_BYTES + 1)
        if len(body) > _MAX_HTTP_RESPONSE_BYTES:
            raise ValueError("Upstream response too large")
        return body.decode("utf-8", errors="ignore")


def _fetch_json(url: str, headers=None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=8) as response:
        body = response.read(_MAX_HTTP_RESPONSE_BYTES + 1)
        if len(body) > _MAX_HTTP_RESPONSE_BYTES:
            raise ValueError("Upstream response too large")
    return json.loads(body.decode("utf-8", errors="ignore"))


def _to_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def _load_alpaca_credentials() -> dict:
    env_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    env_secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    env_base = os.getenv("ALPACA_DATA_BASE_URL") or os.getenv("ALPACA_BASE_URL")
    if env_key and env_secret:
        return {
            "key_id": env_key.strip(),
            "secret_key": env_secret.strip(),
            "base_url": (env_base or "https://data.alpaca.markets").strip(),
        }

    cred_path = os.getenv("ALPACA_CREDENTIALS_PATH", _DEFAULT_ALPACA_CREDS_PATH).strip()
    for path in (cred_path, _DEFAULT_ALPACA_CREDS_ALT_PATH):
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.loads(f.read())

        key_id = (
            payload.get("APCA_API_KEY_ID")
            or payload.get("api_key")
            or payload.get("key_id")
            or payload.get("ALPACA_API_KEY")
        )
        secret_key = (
            payload.get("APCA_API_SECRET_KEY")
            or payload.get("api_secret")
            or payload.get("secret_key")
            or payload.get("ALPACA_SECRET_KEY")
        )
        base_url = (
            payload.get("base_url")
            or payload.get("data_base_url")
            or "https://data.alpaca.markets"
        )
        if key_id and secret_key:
            return {
                "key_id": str(key_id).strip(),
                "secret_key": str(secret_key).strip(),
                "base_url": str(base_url).strip(),
            }
    raise ValueError("Alpaca credentials are not configured")


def _build_snapshot(symbol: str, closes: list, volumes: list, open_price: float, high_price: float, low_price: float) -> dict:
    if not closes:
        raise ValueError(f"No historical close data for {symbol}")
    close_price = closes[-1]
    if close_price is None or close_price <= 0:
        raise ValueError(f"Invalid quote data for {symbol}")

    current_volume = volumes[-1] if volumes else 0
    prev_close = closes[-2] if len(closes) > 1 else close_price
    change_pct = ((close_price - prev_close) / prev_close * 100) if prev_close else 0
    ma50 = sum(closes[-50:]) / min(len(closes), 50)
    ma200 = sum(closes[-200:]) / min(len(closes), 200)
    avg_volume_20 = sum(volumes[-20:]) / min(len(volumes), 20) if volumes else 0

    trend = "NEUTRAL"
    if change_pct > 2:
        trend = "BULLISH"
    elif change_pct < -2:
        trend = "BEARISH"

    return {
        "symbol": symbol,
        "name": symbol,
        "price": close_price,
        "change_pct": change_pct,
        "volume_ratio": (current_volume / avg_volume_20) if avg_volume_20 else 1,
        "trend": trend,
        "fifty_day_avg": ma50,
        "two_hundred_day_avg": ma200,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "volume": current_volume,
    }


def _snapshot_from_stooq(symbol: str) -> dict:
    code = f"{symbol.lower()}.us"
    # Latest quote (Open,High,Low,Close,Volume)
    quote_url = f"https://stooq.com/q/l/?s={urllib.parse.quote(code)}&i=d"
    quote_raw = _fetch_text(quote_url).strip()
    parts = quote_raw.split(",")
    if len(parts) < 8:
        raise ValueError(f"No quote data for {symbol}")

    open_price = _to_float(parts[3], 0)
    high_price = _to_float(parts[4], 0)
    low_price = _to_float(parts[5], 0)

    # Historical data for trend + averages
    hist_url = f"https://stooq.com/q/d/l/?s={urllib.parse.quote(code)}&i=d"
    hist_csv = _fetch_text(hist_url)
    reader = csv.DictReader(io.StringIO(hist_csv))
    rows = [r for r in reader if r.get("Close")]
    if not rows:
        raise ValueError(f"No historical data for {symbol}")

    closes = [_to_float(r["Close"], 0) for r in rows if _to_float(r["Close"], None) is not None]
    volumes = [_to_float(r["Volume"], 0) for r in rows if _to_float(r["Volume"], None) is not None]

    return _build_snapshot(symbol, closes, volumes, open_price, high_price, low_price)


def _snapshot_from_alpaca(symbol: str) -> dict:
    creds = _load_alpaca_credentials()
    base_url = creds["base_url"].rstrip("/")
    headers = {
        "APCA-API-KEY-ID": creds["key_id"],
        "APCA-API-SECRET-KEY": creds["secret_key"],
    }
    params = urllib.parse.urlencode(
        {
            "timeframe": "1Day",
            "limit": "250",
            "adjustment": "raw",
            "feed": "iex",
        }
    )
    url = f"{base_url}/v2/stocks/{urllib.parse.quote(symbol)}/bars?{params}"
    payload = _fetch_json(url, headers=headers)

    bars = payload.get("bars") or []
    if not bars:
        raise ValueError(f"No Alpaca bar data for {symbol}")

    closes = [_to_float(bar.get("c"), None) for bar in bars if _to_float(bar.get("c"), None) is not None]
    volumes = [_to_float(bar.get("v"), 0) for bar in bars if _to_float(bar.get("v"), None) is not None]
    if not closes:
        raise ValueError(f"Invalid Alpaca close data for {symbol}")

    last_bar = bars[-1]
    open_price = _to_float(last_bar.get("o"), closes[-1])
    high_price = _to_float(last_bar.get("h"), closes[-1])
    low_price = _to_float(last_bar.get("l"), closes[-1])

    return _build_snapshot(symbol, closes, volumes, open_price, high_price, low_price)


def _snapshot_from_yfinance(symbol: str) -> dict:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1y", interval="1d", auto_adjust=False)
    if hist is None or hist.empty:
        raise ValueError(f"No yfinance historical data for {symbol}")

    closes = [float(v) for v in hist["Close"].dropna().tolist()]
    volumes = [float(v) for v in hist["Volume"].fillna(0).tolist()]
    open_price = float(hist["Open"].dropna().iloc[-1]) if not hist["Open"].dropna().empty else closes[-1]
    high_price = float(hist["High"].dropna().iloc[-1]) if not hist["High"].dropna().empty else closes[-1]
    low_price = float(hist["Low"].dropna().iloc[-1]) if not hist["Low"].dropna().empty else closes[-1]

    return _build_snapshot(symbol, closes, volumes, open_price, high_price, low_price)


def get_stock_snapshot(symbol: str) -> dict:
    symbol = (symbol or "").strip().upper()
    if not _SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError("Invalid symbol format")

    try:
        return _snapshot_from_alpaca(symbol)
    except Exception as alpaca_exc:
        logger.warning("alpaca snapshot failed for %s: %s", symbol, alpaca_exc)
        try:
            return _snapshot_from_stooq(symbol)
        except Exception as stooq_exc:
            logger.warning("stooq snapshot failed for %s: %s", symbol, stooq_exc)
            try:
                return _snapshot_from_yfinance(symbol)
            except Exception as yf_exc:
                raise ValueError(f"All market data providers failed for {symbol}: {yf_exc}") from yf_exc


def _resolve_scoring_weights(weights: Optional[dict]) -> dict:
    resolved = dict(DEFAULT_SCORING_WEIGHTS)
    if weights is None:
        return resolved
    if not isinstance(weights, dict):
        raise ValueError("weights must be a JSON object")

    unknown = set(weights.keys()) - set(DEFAULT_SCORING_WEIGHTS.keys())
    if unknown:
        unknown_str = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown weight keys: {unknown_str}")

    for key, value in weights.items():
        try:
            numeric = float(value)
        except Exception as exc:
            raise ValueError(f"Weight '{key}' must be numeric") from exc
        lo, hi = _WEIGHT_LIMITS[key]
        if numeric < lo or numeric > hi:
            raise ValueError(f"Weight '{key}' must be between {lo} and {hi}")
        resolved[key] = numeric

    if resolved["sentiment_sell_threshold"] >= resolved["sentiment_buy_threshold"]:
        raise ValueError("sentiment_sell_threshold must be less than sentiment_buy_threshold")
    if resolved["action_sell_threshold"] >= resolved["action_buy_threshold"]:
        raise ValueError("action_sell_threshold must be less than action_buy_threshold")

    return resolved


def _build_scoring_breakdown(snapshot: dict, weights: dict) -> dict:
    sentiment_start = weights["base_score"]
    trend_delta = 0.0
    if snapshot["trend"] == "BULLISH":
        trend_delta = weights["trend_bullish"]
    elif snapshot["trend"] == "BEARISH":
        trend_delta = -weights["trend_bearish"]

    high_volume_applied = snapshot.get("volume_ratio", 1) > weights["volume_ratio_threshold"]
    volume_delta = weights["high_volume_bonus"] if high_volume_applied else 0.0

    ma_bullish = snapshot["fifty_day_avg"] > snapshot["two_hundred_day_avg"]
    ma_delta = weights["ma_bullish_bonus"] if ma_bullish else -weights["ma_bearish_penalty"]

    sentiment_pre_clamp = sentiment_start + trend_delta + volume_delta + ma_delta
    sentiment_score = max(0, min(100, sentiment_pre_clamp))

    action_delta = (
        weights["price_above_ma_bonus"]
        if snapshot["price"] > snapshot["fifty_day_avg"]
        else -weights["price_below_ma_penalty"]
    )
    action_pre_clamp = sentiment_score + action_delta
    action_score = max(0, min(100, action_pre_clamp))

    return {
        "sentiment_inputs": {
            "base_score": sentiment_start,
            "trend": snapshot["trend"],
            "trend_delta": trend_delta,
            "volume_ratio": snapshot.get("volume_ratio", 1),
            "volume_ratio_threshold": weights["volume_ratio_threshold"],
            "high_volume_applied": high_volume_applied,
            "volume_delta": volume_delta,
            "fifty_day_avg": snapshot["fifty_day_avg"],
            "two_hundred_day_avg": snapshot["two_hundred_day_avg"],
            "ma_bullish": ma_bullish,
            "ma_delta": ma_delta,
            "pre_clamp_score": sentiment_pre_clamp,
            "score": sentiment_score,
            "bullish_threshold": weights["sentiment_buy_threshold"],
            "bearish_threshold": weights["sentiment_sell_threshold"],
        },
        "action_inputs": {
            "price": snapshot["price"],
            "fifty_day_avg": snapshot["fifty_day_avg"],
            "price_above_fifty_day_avg": snapshot["price"] > snapshot["fifty_day_avg"],
            "action_delta": action_delta,
            "pre_clamp_confidence": action_pre_clamp,
            "confidence": action_score,
            "buy_threshold": weights["action_buy_threshold"],
            "sell_threshold": weights["action_sell_threshold"],
        },
    }


def get_sentiment(snapshot: dict, weights: Optional[dict] = None) -> dict:
    active_weights = _resolve_scoring_weights(weights)
    score = active_weights["base_score"]
    drivers = []

    if snapshot["trend"] == "BULLISH":
        score += active_weights["trend_bullish"]
        drivers.append("positive price momentum")
    elif snapshot["trend"] == "BEARISH":
        score -= active_weights["trend_bearish"]
        drivers.append("negative price momentum")

    if snapshot.get("volume_ratio", 1) > active_weights["volume_ratio_threshold"]:
        score += active_weights["high_volume_bonus"]
        drivers.append("elevated trading volume")

    if snapshot["fifty_day_avg"] > snapshot["two_hundred_day_avg"]:
        score += active_weights["ma_bullish_bonus"]
        drivers.append("50D MA above 200D MA")
    else:
        score -= active_weights["ma_bearish_penalty"]
        drivers.append("50D MA below 200D MA")

    score = max(0, min(100, score))
    label = "neutral"
    if score >= active_weights["sentiment_buy_threshold"]:
        label = "bullish"
    elif score <= active_weights["sentiment_sell_threshold"]:
        label = "bearish"

    return {
        "score": score,
        "label": label,
        "summary": f"Market sentiment appears {label} (score {score}/100).",
        "drivers": drivers,
        "weights_used": active_weights,
    }


def get_ai_recommendation(snapshot: dict, sentiment: dict, weights: Optional[dict] = None) -> dict:
    active_weights = _resolve_scoring_weights(weights)
    score = sentiment["score"]

    if snapshot["price"] > snapshot["fifty_day_avg"]:
        score += active_weights["price_above_ma_bonus"]
    else:
        score -= active_weights["price_below_ma_penalty"]

    score = max(0, min(100, score))

    action = "HOLD"
    if score >= active_weights["action_buy_threshold"]:
        action = "BUY"
    elif score <= active_weights["action_sell_threshold"]:
        action = "SELL"

    return {
        "action": action,
        "confidence": score,
        "summary": f"AI recommendation: {action} based on trend, volume, and sentiment.",
        "disclaimer": "For informational purposes only, not financial advice.",
        "weights_used": active_weights,
    }


def generate_full_analysis(symbol: str, weights: Optional[dict] = None) -> dict:
    active_weights = _resolve_scoring_weights(weights)
    snapshot = get_stock_snapshot(symbol)
    sentiment = get_sentiment(snapshot, active_weights)
    ai_reco = get_ai_recommendation(snapshot, sentiment, active_weights)

    stock_summary = {
        "summary": f"${snapshot['price']:.2f} • Trend: {snapshot['trend']} • 1D change: {snapshot['change_pct']:.2f}%",
        "details": snapshot,
    }

    return {
        "symbol": snapshot["symbol"],
        "company": snapshot["name"],
        "stock_analysis": stock_summary,
        "sentiment_analysis": sentiment,
        "ai_recommendation": ai_reco,
        "scoring_weights": active_weights,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_scoring_data(symbol: str, weights: Optional[dict] = None) -> dict:
    active_weights = _resolve_scoring_weights(weights)
    snapshot = get_stock_snapshot(symbol)
    breakdown = _build_scoring_breakdown(snapshot, active_weights)

    return {
        "symbol": snapshot["symbol"],
        "company": snapshot["name"],
        "price": snapshot["price"],
        "snapshot": snapshot,
        "scoring_inputs": breakdown,
        "scoring_weights": active_weights,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
