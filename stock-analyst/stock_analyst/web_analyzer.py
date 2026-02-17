import csv
import io
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone

_SYMBOL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9\.\-]{0,9}$")
_MAX_HTTP_RESPONSE_BYTES = 2_000_000
_DEFAULT_ALPACA_CREDS_PATH = os.path.expanduser("~/.configuration/alpaca/credentionals.json")
_DEFAULT_ALPACA_CREDS_ALT_PATH = os.path.expanduser("~/.configuration/alpaca/credentials.json")
logger = logging.getLogger(__name__)


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


def get_sentiment(snapshot: dict) -> dict:
    score = 50
    drivers = []

    if snapshot["trend"] == "BULLISH":
        score += 15
        drivers.append("positive price momentum")
    elif snapshot["trend"] == "BEARISH":
        score -= 15
        drivers.append("negative price momentum")

    if snapshot.get("volume_ratio", 1) > 1.5:
        score += 8
        drivers.append("elevated trading volume")

    if snapshot["fifty_day_avg"] > snapshot["two_hundred_day_avg"]:
        score += 7
        drivers.append("50D MA above 200D MA")
    else:
        score -= 7
        drivers.append("50D MA below 200D MA")

    score = max(0, min(100, score))
    label = "neutral"
    if score >= 65:
        label = "bullish"
    elif score <= 35:
        label = "bearish"

    return {
        "score": score,
        "label": label,
        "summary": f"Market sentiment appears {label} (score {score}/100).",
        "drivers": drivers,
    }


def get_ai_recommendation(snapshot: dict, sentiment: dict) -> dict:
    score = sentiment["score"]

    if snapshot["price"] > snapshot["fifty_day_avg"]:
        score += 5
    else:
        score -= 5

    score = max(0, min(100, score))

    action = "HOLD"
    if score >= 65:
        action = "BUY"
    elif score <= 35:
        action = "SELL"

    return {
        "action": action,
        "confidence": score,
        "summary": f"AI recommendation: {action} based on trend, volume, and sentiment.",
        "disclaimer": "For informational purposes only, not financial advice.",
    }


def generate_full_analysis(symbol: str) -> dict:
    snapshot = get_stock_snapshot(symbol)
    sentiment = get_sentiment(snapshot)
    ai_reco = get_ai_recommendation(snapshot, sentiment)

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
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
