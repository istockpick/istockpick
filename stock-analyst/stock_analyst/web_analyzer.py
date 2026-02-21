import csv
import io
import json
import logging
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional

_SYMBOL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9\.\-]{0,9}$")
_MAX_HTTP_RESPONSE_BYTES = 2_000_000
_MAX_MEDIA_ITEMS_PER_SOURCE = 20
_MAX_MEDIA_AI_MENTIONS = 30
_MEDIA_AI_ENABLED = True
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


def _post_json(url: str, payload: dict, headers=None) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=12) as response:
        body = response.read(_MAX_HTTP_RESPONSE_BYTES + 1)
        if len(body) > _MAX_HTTP_RESPONSE_BYTES:
            raise ValueError("Upstream response too large")
    return json.loads(body.decode("utf-8", errors="ignore"))


def _to_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    return " ".join(text.split()).strip()


def _iso_from_epoch(value) -> Optional[str]:
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _normalize_media_item(source: str, title: str, url: str, created_at: Optional[str], text: str = "", publisher: str = "") -> dict:
    return {
        "source": source,
        "publisher": publisher,
        "title": (title or "").strip(),
        "text": (text or "").strip(),
        "url": (url or "").strip(),
        "created_at": created_at,
    }


def _safe_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _dedupe_media_items(items: list[dict], limit: int) -> list[dict]:
    seen = set()
    deduped = []
    for item in items:
        key = (item.get("url") or "").strip() or (item.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _build_media_query(symbol: str, company: Optional[str] = None) -> str:
    tokens = [symbol.upper()]
    if company:
        cleaned = company.strip()
        if cleaned and cleaned.upper() != symbol.upper():
            tokens.append(f"\"{cleaned}\"")
    return " OR ".join(tokens)


def _score_mentions_fallback(mentions: list[dict]) -> dict:
    positive_terms = {
        "beat", "beats", "bull", "bullish", "buy", "upgrade", "upgraded", "growth",
        "strong", "surge", "outperform", "record", "momentum", "expand", "profit",
    }
    negative_terms = {
        "miss", "misses", "bear", "bearish", "sell", "downgrade", "downgraded", "weak",
        "drop", "plunge", "underperform", "loss", "risk", "lawsuit", "cut", "decline",
    }
    pos_hits = 0
    neg_hits = 0
    for item in mentions:
        text = f"{item.get('title', '')} {item.get('text', '')}".lower()
        pos_hits += sum(1 for term in positive_terms if term in text)
        neg_hits += sum(1 for term in negative_terms if term in text)

    total = pos_hits + neg_hits
    if total <= 0:
        return {
            "positive_score": 0,
            "negative_score": 0,
            "method": "fallback",
            "reason": "No sentiment-bearing keywords found.",
        }

    positive_score = int(round(100 * (pos_hits / total)))
    negative_score = max(0, min(100, 100 - positive_score))
    return {
        "positive_score": positive_score,
        "negative_score": negative_score,
        "method": "fallback",
        "reason": f"Keyword heuristic from {len(mentions)} mentions.",
    }


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = text.strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _score_mentions_with_ai(source_name: str, symbol: str, company: Optional[str], mentions: list[dict]) -> dict:
    key_detected = bool((os.getenv("OPENAI_API_KEY") or "").strip())
    if not _MEDIA_AI_ENABLED:
        fallback = _score_mentions_fallback(mentions)
        fallback["reason"] = "AI scoring disabled; using fallback heuristic."
        fallback["key_detected"] = key_detected
        fallback["openai_reachable"] = None
        return fallback

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        fallback = _score_mentions_fallback(mentions)
        fallback["reason"] = "OPENAI_API_KEY not configured. " + fallback["reason"]
        fallback["key_detected"] = False
        fallback["openai_reachable"] = None
        return fallback

    model = (os.getenv("OPENAI_MEDIA_MODEL") or "gpt-4o-mini").strip()
    sample = []
    for item in mentions[:_MAX_MEDIA_AI_MENTIONS]:
        sample.append(
            {
                "title": item.get("title", ""),
                "text": item.get("text", ""),
                "publisher": item.get("publisher", ""),
                "created_at": item.get("created_at", ""),
            }
        )
    user_prompt = {
        "task": "Score sentiment for stock mentions by source",
        "source": source_name,
        "symbol": symbol,
        "company": company or symbol,
        "mentions": sample,
        "instructions": [
            "Return strict JSON only.",
            "Compute positive_score and negative_score as integers from 0 to 100.",
            "Scores represent bullish vs bearish sentiment in these mentions only.",
            "If mentions are mixed/neutral, keep both moderate; if no signal set both to 0.",
            "Include short reason string.",
            "Format: {\"positive_score\":int,\"negative_score\":int,\"reason\":string}",
        ],
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a finance sentiment scorer. Output strict JSON only."},
            {"role": "user", "content": json.dumps(user_prompt)},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = _post_json(
            "https://api.openai.com/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        content = (
            (((resp.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
        )
        parsed = _extract_json_object(content)
        if not parsed:
            raise ValueError("AI response JSON parse failed")

        pos = max(0, min(100, _safe_int(parsed.get("positive_score"), 0)))
        neg = max(0, min(100, _safe_int(parsed.get("negative_score"), 0)))
        reason = str(parsed.get("reason", "")).strip() or "AI scored from media mentions."
        return {
            "positive_score": pos,
            "negative_score": neg,
            "method": "ai",
            "reason": reason,
            "model": model,
            "key_detected": True,
            "openai_reachable": True,
        }
    except Exception as exc:
        fallback = _score_mentions_fallback(mentions)
        fallback["reason"] = f"AI scoring failed ({exc}). " + fallback["reason"]
        fallback["key_detected"] = True
        fallback["openai_reachable"] = False
        return fallback


def _fetch_x_posts(symbol: str, company: Optional[str], limit: int) -> tuple[str, list[dict], Optional[str]]:
    bearer = (
        os.getenv("X_BEARER_TOKEN")
        or os.getenv("TWITTER_BEARER_TOKEN")
        or ""
    ).strip()
    if not bearer:
        return "unavailable", [], "X_BEARER_TOKEN/TWITTER_BEARER_TOKEN not configured"

    query = f"({_build_media_query(symbol, company)}) (stock OR shares OR earnings) lang:en -is:retweet"
    params = urllib.parse.urlencode(
        {
            "query": query,
            "max_results": str(max(10, min(limit, 100))),
            "tweet.fields": "created_at,lang",
        }
    )
    url = f"https://api.twitter.com/2/tweets/search/recent?{params}"
    headers = {"Authorization": f"Bearer {bearer}"}

    try:
        payload = _fetch_json(url, headers=headers)
        raw_items = payload.get("data", [])
        items = []
        for row in raw_items:
            if not isinstance(row, dict):
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            tweet_id = (row.get("id") or "").strip()
            tweet_url = f"https://x.com/i/web/status/{tweet_id}" if tweet_id else ""
            items.append(
                _normalize_media_item(
                    source="x",
                    publisher="X",
                    title=text[:180],
                    text=text,
                    url=tweet_url,
                    created_at=row.get("created_at"),
                )
            )
        return "ok", _dedupe_media_items(items, limit), None
    except Exception as exc:
        return "error", [], str(exc)


def _fetch_reddit_forum_posts(symbol: str, company: Optional[str], limit: int) -> tuple[str, list[dict], Optional[str]]:
    forums = ["wallstreetbets", "stocks", "investing", "SecurityAnalysis"]
    query = _build_media_query(symbol, company)
    all_items = []

    try:
        for forum in forums:
            params = urllib.parse.urlencode(
                {
                    "q": query,
                    "restrict_sr": "1",
                    "sort": "new",
                    "t": "week",
                    "limit": str(max(5, min(limit, 50))),
                }
            )
            url = f"https://www.reddit.com/r/{forum}/search.json?{params}"
            payload = _fetch_json(url, headers={"User-Agent": "iStockPick/1.0"})
            children = (((payload.get("data") or {}).get("children")) or [])
            for entry in children:
                data = (entry or {}).get("data") or {}
                title = (data.get("title") or "").strip()
                if not title:
                    continue
                permalink = (data.get("permalink") or "").strip()
                post_url = f"https://www.reddit.com{permalink}" if permalink else ""
                all_items.append(
                    _normalize_media_item(
                        source="reddit",
                        publisher=f"r/{forum}",
                        title=title,
                        text=_strip_html(data.get("selftext") or ""),
                        url=post_url,
                        created_at=_iso_from_epoch(data.get("created_utc")),
                    )
                )
        return "ok", _dedupe_media_items(all_items, limit), None
    except Exception as exc:
        return "error", [], str(exc)


def _fetch_news_rss_items(source: str, feed_url: str, symbol: str, company: Optional[str], limit: int) -> list[dict]:
    raw_xml = _fetch_text(feed_url)
    root = ET.fromstring(raw_xml)
    query_terms = [symbol.upper()]
    if company:
        query_terms.extend([t for t in re.split(r"\s+", company.upper().strip()) if t])
    query_terms = [term for term in query_terms if term]

    items = []
    for node in root.findall(".//item"):
        title = _strip_html(node.findtext("title", default=""))
        link = (node.findtext("link", default="") or "").strip()
        pub_date = (node.findtext("pubDate", default="") or "").strip() or None
        description = _strip_html(node.findtext("description", default=""))
        haystack = f"{title} {description}".upper()
        if query_terms and not any(term in haystack for term in query_terms):
            continue
        items.append(
            _normalize_media_item(
                source="major_news",
                publisher=source,
                title=title,
                text=description,
                url=link,
                created_at=pub_date,
            )
        )
        if len(items) >= limit:
            break
    return items


def _fetch_major_news(symbol: str, company: Optional[str], limit: int) -> tuple[str, list[dict], Optional[str]]:
    feeds = {
        "WSJ": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "Reuters": "https://feeds.reuters.com/reuters/businessNews",
        "Financial Times": "https://www.ft.com/markets?format=rss",
    }
    all_items = []
    errors = []
    for source, url in feeds.items():
        try:
            all_items.extend(_fetch_news_rss_items(source, url, symbol, company, limit))
        except Exception as exc:
            errors.append(f"{source}: {exc}")

    status = "ok" if all_items else ("error" if errors else "ok")
    reason = "; ".join(errors) if errors else None
    return status, _dedupe_media_items(all_items, limit), reason


def generate_media_analysis(symbol: str, company: Optional[str] = None, max_items_per_source: int = 10) -> dict:
    cleaned_symbol = (symbol or "").strip().upper()
    if not _SYMBOL_PATTERN.fullmatch(cleaned_symbol):
        raise ValueError("Invalid symbol format")

    limit = max(1, min(int(max_items_per_source), _MAX_MEDIA_ITEMS_PER_SOURCE))
    x_status, x_items, x_reason = _fetch_x_posts(cleaned_symbol, company, limit)
    reddit_status, reddit_items, reddit_reason = _fetch_reddit_forum_posts(cleaned_symbol, company, limit)
    news_status, news_items, news_reason = _fetch_major_news(cleaned_symbol, company, limit)
    x_scores = _score_mentions_with_ai("X", cleaned_symbol, company, x_items)
    reddit_scores = _score_mentions_with_ai("Reddit", cleaned_symbol, company, reddit_items)
    news_scores = _score_mentions_with_ai("Major News", cleaned_symbol, company, news_items)
    key_detected = bool((os.getenv("OPENAI_API_KEY") or "").strip())

    return {
        "symbol": cleaned_symbol,
        "company": (company or cleaned_symbol),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_only": True,
        "summary": (
            "Media sources retrieved from X, Reddit stock forums, and major news. "
            "Bullish/bearish inference logic is not enabled yet."
        ),
        "sources": {
            "x": {
                "status": x_status,
                "items": x_items,
                "reason": x_reason,
                "positive_score": x_scores.get("positive_score", 0),
                "negative_score": x_scores.get("negative_score", 0),
                "score_reason": x_scores.get("reason"),
                "score_method": x_scores.get("method"),
                "score_model": x_scores.get("model"),
            },
            "reddit": {
                "status": reddit_status,
                "items": reddit_items,
                "reason": reddit_reason,
                "positive_score": reddit_scores.get("positive_score", 0),
                "negative_score": reddit_scores.get("negative_score", 0),
                "score_reason": reddit_scores.get("reason"),
                "score_method": reddit_scores.get("method"),
                "score_model": reddit_scores.get("model"),
            },
            "major_news": {
                "status": news_status,
                "items": news_items,
                "reason": news_reason,
                "positive_score": news_scores.get("positive_score", 0),
                "negative_score": news_scores.get("negative_score", 0),
                "score_reason": news_scores.get("reason"),
                "score_method": news_scores.get("method"),
                "score_model": news_scores.get("model"),
            },
        },
        "sub_scores": {
            "x": {
                "positive_score": x_scores.get("positive_score", 0),
                "negative_score": x_scores.get("negative_score", 0),
            },
            "reddit": {
                "positive_score": reddit_scores.get("positive_score", 0),
                "negative_score": reddit_scores.get("negative_score", 0),
            },
            "major_news": {
                "positive_score": news_scores.get("positive_score", 0),
                "negative_score": news_scores.get("negative_score", 0),
            },
        },
        "counts": {
            "x": len(x_items),
            "reddit": len(reddit_items),
            "major_news": len(news_items),
            "total": len(x_items) + len(reddit_items) + len(news_items),
        },
        "media_trend": {
            "label": "unknown",
            "score": None,
            "reason": "Retrieval-only phase; scoring logic pending.",
        },
        "debug": {
            "ai_enabled": _MEDIA_AI_ENABLED,
            "key_detected": key_detected,
            "openai_reachable": any(
                score.get("openai_reachable") is True
                for score in (x_scores, reddit_scores, news_scores)
            ),
            "source_methods": {
                "x": x_scores.get("method"),
                "reddit": reddit_scores.get("method"),
                "major_news": news_scores.get("method"),
            },
            "source_openai_reachable": {
                "x": x_scores.get("openai_reachable"),
                "reddit": reddit_scores.get("openai_reachable"),
                "major_news": news_scores.get("openai_reachable"),
            },
        },
    }


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
    try:
        media_analysis = generate_media_analysis(snapshot["symbol"], company=snapshot.get("name"))
    except Exception as exc:
        media_analysis = {
            "symbol": snapshot["symbol"],
            "company": snapshot.get("name", snapshot["symbol"]),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "retrieval_only": True,
            "summary": "Social analysis unavailable.",
            "error": str(exc),
            "sources": {"x": {}, "reddit": {}, "major_news": {}},
            "sub_scores": {
                "x": {"positive_score": 0, "negative_score": 0},
                "reddit": {"positive_score": 0, "negative_score": 0},
                "major_news": {"positive_score": 0, "negative_score": 0},
            },
            "counts": {"x": 0, "reddit": 0, "major_news": 0, "total": 0},
            "media_trend": {"label": "unknown", "score": None, "reason": "Media retrieval failed."},
            "debug": {"ai_enabled": _MEDIA_AI_ENABLED, "openai_reachable": False},
        }

    stock_summary = {
        "summary": f"${snapshot['price']:.2f} • Trend: {snapshot['trend']} • 1D change: {snapshot['change_pct']:.2f}%",
        "details": snapshot,
    }

    return {
        "symbol": snapshot["symbol"],
        "company": snapshot["name"],
        "stock_analysis": stock_summary,
        "sentiment_analysis": sentiment,
        "social_analysis": media_analysis,
        "media_analysis": media_analysis,
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
