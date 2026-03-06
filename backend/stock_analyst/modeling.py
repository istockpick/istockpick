from datetime import datetime, timezone
from typing import Optional

from .market_data import get_company_profile, get_price_history
from .qlib_engine import (
    bollinger_bands,
    engine_name,
    macd,
    moving_average,
    rsi,
    support_resistance,
    volatility_summary,
    volume_summary,
)


def generate_modeling_payload(symbol: str, asset_type: str = "stock", period: str = "1y") -> dict:
    resolved_symbol = (symbol or "").strip().upper()
    at = (asset_type or "stock").lower()
    history = get_price_history(resolved_symbol, period=period, asset_type=at)
    if history is None or history.empty:
        raise ValueError(f"No history available for {resolved_symbol}")

    profile = get_company_profile(resolved_symbol)
    closes = history["Close"].dropna()
    latest_close = float(closes.iloc[-1])
    ma20 = moving_average(closes, 20)
    ma50 = moving_average(closes, 50)
    ma200 = moving_average(closes, 200)
    rsi14 = rsi(closes, 14)
    macd_values = macd(closes)
    bands = bollinger_bands(closes, 20)
    volume = volume_summary(history)
    levels = support_resistance(history, 20)
    volatility = volatility_summary(history, 20)

    latest_features = {
        "close": latest_close,
        "ma_20": ma20,
        "ma_50": ma50,
        "ma_200": ma200,
        "rsi_14": rsi14,
        "macd": macd_values["macd"],
        "macd_signal": macd_values["signal"],
        "macd_histogram": macd_values["histogram"],
        "bollinger_upper": bands["upper_band"],
        "bollinger_middle": bands["middle_band"],
        "bollinger_lower": bands["lower_band"],
        "volume_ratio": volume["volume_ratio"],
        "atr_20": volatility["atr"],
        "historical_volatility_20": volatility["historical_volatility"],
        "distance_to_support": levels["distance_to_support"],
        "distance_to_resistance": levels["distance_to_resistance"],
    }

    target = None
    if len(closes) >= 6:
        target = float(((closes.iloc[-1] / closes.iloc[-6]) - 1.0) * 100.0)

    preview_rows = []
    tail = history.tail(5)
    for idx, row in tail.iterrows():
        preview_rows.append(
            {
                "date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                "open": float(row.get("Open", 0) or 0),
                "high": float(row.get("High", 0) or 0),
                "low": float(row.get("Low", 0) or 0),
                "close": float(row.get("Close", 0) or 0),
                "volume": float(row.get("Volume", 0) or 0),
            }
        )

    return {
        "symbol": resolved_symbol,
        "company": profile.get("name") or profile.get("company_name") or resolved_symbol,
        "asset_type": at,
        "data_source": "openbb",
        "feature_engine": engine_name(),
        "history_rows": int(len(history)),
        "period": period,
        "latest_features": latest_features,
        "label_preview": {
            "definition": "5-day trailing return percent on the latest window",
            "value": target,
        },
        "training_segments": {
            "train": "first 70% of rows",
            "valid": "next 15% of rows",
            "test": "final 15% of rows",
        },
        "preview_rows": preview_rows,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
