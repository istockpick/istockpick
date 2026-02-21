"""Crypto snapshot and scoring -- uses yfinance with SYMBOL-USD tickers."""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_CRYPTO_PATTERN = re.compile(r"^[A-Z0-9]+-USD$", re.IGNORECASE)
_KNOWN_CRYPTOS = {
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC",
    "LINK", "UNI", "SHIB", "LTC", "BCH", "ATOM", "XLM", "NEAR", "APT",
    "ARB", "OP", "FIL", "HBAR", "ICP", "VET", "ALGO", "FTM", "SAND",
    "MANA", "AXS", "AAVE", "MKR", "CRV", "SNX", "COMP", "SUSHI",
    "BNB", "TRX", "TON", "PEPE", "WIF", "BONK", "RENDER", "SUI",
}

CRYPTO_WEIGHT_DEFAULTS = {
    "base_score": 50.0,
    "trend_bullish": 18.0,
    "trend_bearish": 18.0,
    "high_volume_bonus": 10.0,
    "ma_bullish_bonus": 7.0,
    "ma_bearish_penalty": 7.0,
    "price_above_ma_bonus": 5.0,
    "price_below_ma_penalty": 5.0,
    "volume_ratio_threshold": 2.0,
    "sentiment_buy_threshold": 65.0,
    "sentiment_sell_threshold": 35.0,
    "action_buy_threshold": 65.0,
    "action_sell_threshold": 35.0,
}


def is_crypto_symbol(symbol: str) -> bool:
    s = (symbol or "").strip().upper()
    if _CRYPTO_PATTERN.fullmatch(s):
        return True
    return s in _KNOWN_CRYPTOS


def normalize_crypto_symbol(symbol: str) -> str:
    """Ensure crypto symbols are in SYMBOL-USD format."""
    s = (symbol or "").strip().upper()
    if _CRYPTO_PATTERN.fullmatch(s):
        return s
    if s in _KNOWN_CRYPTOS:
        return f"{s}-USD"
    return s


def get_crypto_snapshot(symbol: str) -> dict:
    """Fetch crypto price data via yfinance."""
    import yfinance as yf

    yf_symbol = normalize_crypto_symbol(symbol)
    ticker = yf.Ticker(yf_symbol)
    hist = ticker.history(period="1y", interval="1d", auto_adjust=False)
    if hist is None or hist.empty:
        raise ValueError(f"No historical data for {yf_symbol}")

    closes = [float(v) for v in hist["Close"].dropna().tolist()]
    volumes = [float(v) for v in hist["Volume"].fillna(0).tolist()]
    if not closes:
        raise ValueError(f"No close data for {yf_symbol}")

    price = closes[-1]
    prev_close = closes[-2] if len(closes) > 1 else price
    change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
    ma50 = sum(closes[-50:]) / min(len(closes), 50)
    ma200 = sum(closes[-200:]) / min(len(closes), 200)
    avg_vol_20 = sum(volumes[-20:]) / min(len(volumes), 20) if volumes else 0
    current_vol = volumes[-1] if volumes else 0

    trend = "NEUTRAL"
    if change_pct > 3:
        trend = "BULLISH"
    elif change_pct < -3:
        trend = "BEARISH"

    open_price = float(hist["Open"].dropna().iloc[-1]) if not hist["Open"].dropna().empty else price
    high_price = float(hist["High"].dropna().iloc[-1]) if not hist["High"].dropna().empty else price
    low_price = float(hist["Low"].dropna().iloc[-1]) if not hist["Low"].dropna().empty else price

    base_symbol = yf_symbol.replace("-USD", "") if "-USD" in yf_symbol else yf_symbol

    return {
        "symbol": yf_symbol,
        "name": base_symbol,
        "asset_type": "crypto",
        "price": price,
        "change_pct": change_pct,
        "volume_ratio": (current_vol / avg_vol_20) if avg_vol_20 else 1,
        "trend": trend,
        "fifty_day_avg": ma50,
        "two_hundred_day_avg": ma200,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "volume": current_vol,
    }


def get_crypto_sentiment(snapshot: dict, weights: Optional[dict] = None) -> dict:
    """Compute sentiment for a crypto asset."""
    w = dict(CRYPTO_WEIGHT_DEFAULTS)
    if weights:
        for k, v in weights.items():
            if k in w:
                w[k] = float(v)

    score = w["base_score"]
    drivers = []

    if snapshot["trend"] == "BULLISH":
        score += w["trend_bullish"]
        drivers.append("positive price momentum")
    elif snapshot["trend"] == "BEARISH":
        score -= w["trend_bearish"]
        drivers.append("negative price momentum")

    if snapshot.get("volume_ratio", 1) > w["volume_ratio_threshold"]:
        score += w["high_volume_bonus"]
        drivers.append("elevated trading volume")

    if snapshot["fifty_day_avg"] > snapshot["two_hundred_day_avg"]:
        score += w["ma_bullish_bonus"]
        drivers.append("50D MA above 200D MA")
    else:
        score -= w["ma_bearish_penalty"]
        drivers.append("50D MA below 200D MA")

    score = max(0, min(100, score))
    label = "neutral"
    if score >= w["sentiment_buy_threshold"]:
        label = "bullish"
    elif score <= w["sentiment_sell_threshold"]:
        label = "bearish"

    return {
        "score": score,
        "label": label,
        "summary": f"Crypto sentiment appears {label} (score {score}/100).",
        "drivers": drivers,
        "weights_used": w,
    }


def get_crypto_recommendation(snapshot: dict, sentiment: dict, weights: Optional[dict] = None) -> dict:
    """BUY/SELL/HOLD recommendation for crypto."""
    w = dict(CRYPTO_WEIGHT_DEFAULTS)
    if weights:
        for k, v in weights.items():
            if k in w:
                w[k] = float(v)

    score = sentiment["score"]
    if snapshot["price"] > snapshot["fifty_day_avg"]:
        score += w["price_above_ma_bonus"]
    else:
        score -= w["price_below_ma_penalty"]
    score = max(0, min(100, score))

    action = "HOLD"
    if score >= w["action_buy_threshold"]:
        action = "BUY"
    elif score <= w["action_sell_threshold"]:
        action = "SELL"

    return {
        "action": action,
        "confidence": score,
        "summary": f"AI recommendation: {action} based on crypto trend, volume, and momentum.",
        "disclaimer": "For informational purposes only, not financial advice. Crypto is highly volatile.",
        "weights_used": w,
    }
