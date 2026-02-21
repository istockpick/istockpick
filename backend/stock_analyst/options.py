"""Options chain, snapshot, and scoring -- uses yfinance option_chain()."""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

OPTIONS_WEIGHT_DEFAULTS = {
    "base_score": 50.0,
    "trend_bullish": 12.0,
    "trend_bearish": 12.0,
    "high_volume_bonus": 6.0,
    "ma_bullish_bonus": 5.0,
    "ma_bearish_penalty": 5.0,
    "price_above_ma_bonus": 4.0,
    "price_below_ma_penalty": 4.0,
    "volume_ratio_threshold": 1.5,
    "put_call_ratio_bullish_bonus": 8.0,
    "put_call_ratio_bearish_penalty": 8.0,
    "put_call_ratio_threshold": 0.7,
    "high_iv_penalty": 5.0,
    "iv_threshold": 0.5,
    "sentiment_buy_threshold": 65.0,
    "sentiment_sell_threshold": 35.0,
    "action_buy_threshold": 65.0,
    "action_sell_threshold": 35.0,
}


def get_options_chain(symbol: str, expiry: Optional[str] = None) -> dict:
    """Fetch options chain for a symbol. If expiry is None, use the nearest
    available expiration date."""
    import yfinance as yf

    symbol = (symbol or "").strip().upper()
    ticker = yf.Ticker(symbol)

    expirations = ticker.options
    if not expirations:
        raise ValueError(f"No options available for {symbol}")

    target_expiry = expiry
    if not target_expiry:
        target_expiry = expirations[0]
    elif target_expiry not in expirations:
        raise ValueError(f"Expiry {target_expiry} not available. Options: {list(expirations[:10])}")

    chain = ticker.option_chain(target_expiry)

    calls_data = []
    if chain.calls is not None and not chain.calls.empty:
        for _, row in chain.calls.iterrows():
            calls_data.append({
                "strike": float(row.get("strike", 0)),
                "lastPrice": float(row.get("lastPrice", 0)),
                "bid": float(row.get("bid", 0)),
                "ask": float(row.get("ask", 0)),
                "volume": int(row.get("volume", 0)) if row.get("volume") and str(row.get("volume")) != "nan" else 0,
                "openInterest": int(row.get("openInterest", 0)) if row.get("openInterest") and str(row.get("openInterest")) != "nan" else 0,
                "impliedVolatility": float(row.get("impliedVolatility", 0)),
                "inTheMoney": bool(row.get("inTheMoney", False)),
            })

    puts_data = []
    if chain.puts is not None and not chain.puts.empty:
        for _, row in chain.puts.iterrows():
            puts_data.append({
                "strike": float(row.get("strike", 0)),
                "lastPrice": float(row.get("lastPrice", 0)),
                "bid": float(row.get("bid", 0)),
                "ask": float(row.get("ask", 0)),
                "volume": int(row.get("volume", 0)) if row.get("volume") and str(row.get("volume")) != "nan" else 0,
                "openInterest": int(row.get("openInterest", 0)) if row.get("openInterest") and str(row.get("openInterest")) != "nan" else 0,
                "impliedVolatility": float(row.get("impliedVolatility", 0)),
                "inTheMoney": bool(row.get("inTheMoney", False)),
            })

    total_call_oi = sum(c["openInterest"] for c in calls_data)
    total_put_oi = sum(p["openInterest"] for p in puts_data)
    put_call_ratio = (total_put_oi / total_call_oi) if total_call_oi > 0 else 0.0

    total_call_vol = sum(c["volume"] for c in calls_data)
    total_put_vol = sum(p["volume"] for p in puts_data)

    all_iv = [c["impliedVolatility"] for c in calls_data + puts_data if c["impliedVolatility"] > 0]
    avg_iv = sum(all_iv) / len(all_iv) if all_iv else 0.0

    # Max pain: strike where total option value is minimized for holders
    max_pain = _compute_max_pain(calls_data, puts_data)

    return {
        "symbol": symbol,
        "expiry": target_expiry,
        "available_expiries": list(expirations[:20]),
        "calls_count": len(calls_data),
        "puts_count": len(puts_data),
        "calls": calls_data,
        "puts": puts_data,
        "summary": {
            "put_call_ratio": round(put_call_ratio, 4),
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "total_call_volume": total_call_vol,
            "total_put_volume": total_put_vol,
            "avg_implied_volatility": round(avg_iv, 4),
            "max_pain": max_pain,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _compute_max_pain(calls: list[dict], puts: list[dict]) -> Optional[float]:
    """Find the strike where total losses for option holders is maximized
    (i.e., maximum pain for buyers)."""
    strikes = sorted({c["strike"] for c in calls} | {p["strike"] for p in puts})
    if not strikes:
        return None

    min_pain = float("inf")
    max_pain_strike = None

    for strike in strikes:
        call_pain = sum(max(0, strike - c["strike"]) * c["openInterest"] for c in calls)
        put_pain = sum(max(0, p["strike"] - strike) * p["openInterest"] for p in puts)
        total = call_pain + put_pain
        if total < min_pain:
            min_pain = total
            max_pain_strike = strike

    return max_pain_strike


def get_options_snapshot(symbol: str) -> dict:
    """Underlying stock snapshot enriched with nearest-expiry options summary."""
    import yfinance as yf

    symbol = (symbol or "").strip().upper()
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1y", interval="1d", auto_adjust=False)
    if hist is None or hist.empty:
        raise ValueError(f"No historical data for {symbol}")

    closes = [float(v) for v in hist["Close"].dropna().tolist()]
    volumes = [float(v) for v in hist["Volume"].fillna(0).tolist()]
    if not closes:
        raise ValueError(f"No close data for {symbol}")

    price = closes[-1]
    prev_close = closes[-2] if len(closes) > 1 else price
    change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
    ma50 = sum(closes[-50:]) / min(len(closes), 50)
    ma200 = sum(closes[-200:]) / min(len(closes), 200)
    avg_vol_20 = sum(volumes[-20:]) / min(len(volumes), 20) if volumes else 0
    current_vol = volumes[-1] if volumes else 0

    trend = "NEUTRAL"
    if change_pct > 2:
        trend = "BULLISH"
    elif change_pct < -2:
        trend = "BEARISH"

    open_price = float(hist["Open"].dropna().iloc[-1]) if not hist["Open"].dropna().empty else price
    high_price = float(hist["High"].dropna().iloc[-1]) if not hist["High"].dropna().empty else price
    low_price = float(hist["Low"].dropna().iloc[-1]) if not hist["Low"].dropna().empty else price

    # Fetch options summary for nearest expiry
    options_summary = {}
    try:
        chain = get_options_chain(symbol)
        options_summary = chain.get("summary", {})
        options_summary["expiry"] = chain.get("expiry")
    except Exception as exc:
        logger.warning("Options chain fetch failed for %s: %s", symbol, exc)

    return {
        "symbol": symbol,
        "name": symbol,
        "asset_type": "option",
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
        "options_summary": options_summary,
    }


def get_options_sentiment(snapshot: dict, weights: Optional[dict] = None) -> dict:
    """Compute sentiment for an options-focused analysis using put/call ratio and IV."""
    w = dict(OPTIONS_WEIGHT_DEFAULTS)
    if weights:
        for k, v in weights.items():
            if k in w:
                w[k] = float(v)

    score = w["base_score"]
    drivers = []

    if snapshot["trend"] == "BULLISH":
        score += w["trend_bullish"]
        drivers.append("positive underlying momentum")
    elif snapshot["trend"] == "BEARISH":
        score -= w["trend_bearish"]
        drivers.append("negative underlying momentum")

    if snapshot.get("volume_ratio", 1) > w["volume_ratio_threshold"]:
        score += w["high_volume_bonus"]
        drivers.append("elevated trading volume")

    if snapshot["fifty_day_avg"] > snapshot["two_hundred_day_avg"]:
        score += w["ma_bullish_bonus"]
        drivers.append("50D MA above 200D MA")
    else:
        score -= w["ma_bearish_penalty"]
        drivers.append("50D MA below 200D MA")

    # Put/call ratio signal
    opts = snapshot.get("options_summary", {})
    pcr = opts.get("put_call_ratio", 0)
    if pcr > 0:
        if pcr < w["put_call_ratio_threshold"]:
            score += w["put_call_ratio_bullish_bonus"]
            drivers.append(f"low put/call ratio ({pcr:.2f}) — bullish signal")
        elif pcr > 1.0:
            score -= w["put_call_ratio_bearish_penalty"]
            drivers.append(f"high put/call ratio ({pcr:.2f}) — bearish signal")

    # High IV penalty
    avg_iv = opts.get("avg_implied_volatility", 0)
    if avg_iv > w["iv_threshold"]:
        score -= w["high_iv_penalty"]
        drivers.append(f"elevated implied volatility ({avg_iv:.2%})")

    score = max(0, min(100, score))
    label = "neutral"
    if score >= w["sentiment_buy_threshold"]:
        label = "bullish"
    elif score <= w["sentiment_sell_threshold"]:
        label = "bearish"

    return {
        "score": score,
        "label": label,
        "summary": f"Options sentiment appears {label} (score {score}/100).",
        "drivers": drivers,
        "weights_used": w,
    }


def get_options_recommendation(snapshot: dict, sentiment: dict, weights: Optional[dict] = None) -> dict:
    """BUY/SELL/HOLD recommendation for options."""
    w = dict(OPTIONS_WEIGHT_DEFAULTS)
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
        "summary": f"AI recommendation: {action} based on options flow, IV, and underlying trend.",
        "disclaimer": "For informational purposes only, not financial advice. Options involve significant risk.",
        "weights_used": w,
    }
