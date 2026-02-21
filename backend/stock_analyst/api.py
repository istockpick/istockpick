import re
import json
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError

from .web_analyzer import generate_full_analysis, generate_scoring_data, AssetType

_DB_LOCK = threading.Lock()
_MAX_STOCK_INPUT_LEN = 120
_MAX_AGENT_NAME_LEN = 64
_MAX_AGENT_TOKEN_LEN = 256
_AGENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-\.]{0,63}$")
_MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-\.]{0,63}$")
_DEFAULT_MODEL_NAME = "default"


class RecommendationRequest(BaseModel):
    stock: str = Field(..., description="Ticker symbol or company name")
    agent_name: str = Field(..., description="Registered agent name")
    agent_token: str = Field(..., description="Registered agent token")
    asset_type: str = Field(
        default="stock",
        description="Asset type: stock, crypto, option, or future",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Optional personalized model name",
    )
    weights: Optional["ScoringWeights"] = Field(
        default=None,
        description="Optional scoring weight overrides",
    )
    verbose: bool = Field(
        default=False,
        description="If true, return detailed recommendation sections",
    )
    verborse: Optional[bool] = Field(
        default=None,
        description="Deprecated alias for verbose",
    )


class AgentRegistrationRequest(BaseModel):
    name: str = Field(..., description="Unique agent name")


class BatchRecommendationRequest(BaseModel):
    stocks: list[str] = Field(..., description="List of ticker symbols or company names")
    agent_name: str = Field(..., description="Registered agent name")
    agent_token: str = Field(..., description="Registered agent token")
    asset_type: str = Field(
        default="stock",
        description="Asset type: stock, crypto, option, or future",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Optional personalized model name",
    )
    weights: Optional["ScoringWeights"] = Field(
        default=None,
        description="Optional scoring weight overrides",
    )
    verbose: bool = Field(
        default=False,
        description="If true, return detailed recommendation sections per stock",
    )
    verborse: Optional[bool] = Field(
        default=None,
        description="Deprecated alias for verbose",
    )


class ScoringDataRequest(BaseModel):
    stock: str = Field(..., description="Ticker symbol or company name")
    agent_name: str = Field(..., description="Registered agent name")
    agent_token: str = Field(..., description="Registered agent token")
    asset_type: str = Field(
        default="stock",
        description="Asset type: stock, crypto, option, or future",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Optional personalized model name",
    )
    weights: Optional["ScoringWeights"] = Field(
        default=None,
        description="Optional scoring weight overrides",
    )


class ScoringWeights(BaseModel):
    base_score: Optional[float] = Field(default=None, ge=0, le=100)
    trend_bullish: Optional[float] = Field(default=None, ge=0, le=100)
    trend_bearish: Optional[float] = Field(default=None, ge=0, le=100)
    high_volume_bonus: Optional[float] = Field(default=None, ge=0, le=100)
    ma_bullish_bonus: Optional[float] = Field(default=None, ge=0, le=100)
    ma_bearish_penalty: Optional[float] = Field(default=None, ge=0, le=100)
    price_above_ma_bonus: Optional[float] = Field(default=None, ge=0, le=100)
    price_below_ma_penalty: Optional[float] = Field(default=None, ge=0, le=100)
    volume_ratio_threshold: Optional[float] = Field(default=None, ge=0.1, le=10)
    sentiment_buy_threshold: Optional[float] = Field(default=None, ge=0, le=100)
    sentiment_sell_threshold: Optional[float] = Field(default=None, ge=0, le=100)
    action_buy_threshold: Optional[float] = Field(default=None, ge=0, le=100)
    action_sell_threshold: Optional[float] = Field(default=None, ge=0, le=100)


def _rebuild_forward_refs() -> None:
    for model in (RecommendationRequest, BatchRecommendationRequest, ScoringDataRequest):
        if hasattr(model, "model_rebuild"):
            model.model_rebuild()
        elif hasattr(model, "update_forward_refs"):
            model.update_forward_refs()


def _weights_to_payload(weights: Optional[ScoringWeights]) -> Optional[dict]:
    if weights is None:
        return None
    if hasattr(weights, "model_dump"):
        payload = weights.model_dump(exclude_none=True)
    else:
        payload = weights.dict(exclude_none=True)
    if (
        "sentiment_buy_threshold" in payload
        and "sentiment_sell_threshold" in payload
        and payload["sentiment_sell_threshold"] >= payload["sentiment_buy_threshold"]
    ):
        raise HTTPException(
            status_code=400,
            detail="sentiment_sell_threshold must be less than sentiment_buy_threshold",
        )
    if (
        "action_buy_threshold" in payload
        and "action_sell_threshold" in payload
        and payload["action_sell_threshold"] >= payload["action_buy_threshold"]
    ):
        raise HTTPException(
            status_code=400,
            detail="action_sell_threshold must be less than action_buy_threshold",
        )
    return payload


def _parse_weights_query(weights: Optional[str]) -> Optional[dict]:
    if weights is None or not weights.strip():
        return None
    try:
        payload = json.loads(weights)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid weights JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="weights must be a JSON object")
    try:
        parsed = ScoringWeights(**payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid weights payload: {exc.errors()}") from exc
    return _weights_to_payload(parsed)


def _resolve_verbose_flag(verbose: bool = False, verborse: Optional[bool] = None) -> bool:
    if verborse is not None:
        return bool(verborse)
    return bool(verbose)


def _normalize_model_name(model_name: Optional[str]) -> Optional[str]:
    if model_name is None:
        return None
    cleaned = model_name.strip()
    if not cleaned:
        return None
    if len(cleaned) > 64 or not _MODEL_NAME_PATTERN.fullmatch(cleaned):
        raise HTTPException(status_code=400, detail="Invalid model_name format")
    return cleaned


def _looks_like_ticker(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9\.\-]{0,9}", value.strip()))


def _looks_like_multi_asset_ticker(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\.\-=]{0,14}", value.strip()))


def _detect_asset_type(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    if re.fullmatch(r"[A-Z0-9]+=F", s):
        return "future"
    if re.fullmatch(r"[A-Z0-9]+-USD", s):
        return "crypto"
    return "stock"


def _resolve_symbol(stock: str, asset_type: str = "stock") -> Optional[str]:
    candidate = stock.strip()
    if not candidate or len(candidate) > _MAX_STOCK_INPUT_LEN:
        return None

    at = (asset_type or "stock").lower()

    if at == "crypto":
        from .crypto import normalize_crypto_symbol, is_crypto_symbol
        if is_crypto_symbol(candidate):
            return normalize_crypto_symbol(candidate)
        return normalize_crypto_symbol(candidate.upper())

    if at == "future":
        from .futures import normalize_futures_symbol, is_futures_symbol
        if is_futures_symbol(candidate):
            return normalize_futures_symbol(candidate)
        return normalize_futures_symbol(candidate.upper())

    detected = _detect_asset_type(candidate)
    if detected in ("crypto", "future"):
        return candidate.upper()

    if _looks_like_ticker(candidate):
        return candidate.upper()

    try:
        import yfinance as yf

        search = yf.Search(query=candidate, max_results=5, news_count=0)
        quotes = search.quotes or []
        for quote in quotes:
            symbol = quote.get("symbol")
            if symbol and _looks_like_ticker(symbol):
                return symbol.upper()
    except Exception:
        return None

    return None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Stock Analyst API",
        version="1.0.0",
        description="API endpoint for stock trading recommendations",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/v1/agents/register")
    def register_agent(request: AgentRegistrationRequest) -> dict:
        return _register_agent(request.name)

    @app.get("/api/v1/recommendation")
    def get_recommendation(
        stock: str = Query(..., description="Ticker symbol or company name"),
        agent_name: str = Query(..., description="Registered agent name"),
        agent_token: str = Query(..., description="Registered agent token"),
        asset_type: str = Query(
            default="stock",
            description="Asset type: stock, crypto, option, or future",
        ),
        model_name: Optional[str] = Query(
            default=None,
            description="Optional personalized model name",
        ),
        verbose: bool = Query(
            default=False,
            description="If true, return detailed recommendation sections",
        ),
        verborse: Optional[bool] = Query(
            default=None,
            description="Deprecated alias for verbose",
        ),
        weights: Optional[str] = Query(
            default=None,
            description="Optional JSON-encoded scoring weights",
        ),
    ) -> dict:
        _validate_agent(agent_name, agent_token)
        normalized_model = _normalize_model_name(model_name)
        weights_payload = _parse_weights_query(weights)
        verbose_flag = _resolve_verbose_flag(verbose=verbose, verborse=verborse)
        effective_weights, active_model = _resolve_effective_weights(
            agent_name,
            agent_token,
            weights_payload,
            normalized_model,
        )
        response = _build_recommendation_response(
            stock,
            effective_weights,
            verbose=verbose_flag,
            model_name=active_model,
            asset_type=asset_type,
        )
        if weights_payload is not None:
            _save_agent_weights(agent_name, agent_token, weights_payload, model_name=active_model)
        return response

    @app.post("/api/v1/recommendation")
    def post_recommendation(request: RecommendationRequest) -> dict:
        _validate_agent(request.agent_name, request.agent_token)
        normalized_model = _normalize_model_name(request.model_name)
        verbose_flag = _resolve_verbose_flag(verbose=request.verbose, verborse=request.verborse)
        requested_weights = _weights_to_payload(request.weights)
        effective_weights, active_model = _resolve_effective_weights(
            request.agent_name,
            request.agent_token,
            requested_weights,
            normalized_model,
        )
        response = _build_recommendation_response(
            request.stock,
            effective_weights,
            verbose=verbose_flag,
            model_name=active_model,
            asset_type=request.asset_type,
        )
        if requested_weights is not None:
            _save_agent_weights(
                request.agent_name,
                request.agent_token,
                requested_weights,
                model_name=active_model,
            )
        return response

    @app.get("/api/v1/recommendations")
    def get_recommendations(
        stocks: str = Query(..., description="Comma-separated ticker symbols or company names"),
        agent_name: str = Query(..., description="Registered agent name"),
        agent_token: str = Query(..., description="Registered agent token"),
        asset_type: str = Query(
            default="stock",
            description="Asset type: stock, crypto, option, or future",
        ),
        model_name: Optional[str] = Query(
            default=None,
            description="Optional personalized model name",
        ),
        verbose: bool = Query(
            default=False,
            description="If true, return detailed recommendation sections per stock",
        ),
        verborse: Optional[bool] = Query(
            default=None,
            description="Deprecated alias for verbose",
        ),
        weights: Optional[str] = Query(
            default=None,
            description="Optional JSON-encoded scoring weights",
        ),
    ) -> dict:
        _validate_agent(agent_name, agent_token)
        normalized_model = _normalize_model_name(model_name)
        parsed_stocks = _parse_batch_stocks(stocks)
        weights_payload = _parse_weights_query(weights)
        verbose_flag = _resolve_verbose_flag(verbose=verbose, verborse=verborse)
        effective_weights, active_model = _resolve_effective_weights(
            agent_name,
            agent_token,
            weights_payload,
            normalized_model,
        )
        response = _build_batch_recommendation_response(
            parsed_stocks,
            effective_weights,
            verbose=verbose_flag,
            model_name=active_model,
            asset_type=asset_type,
        )
        if weights_payload is not None:
            _save_agent_weights(agent_name, agent_token, weights_payload, model_name=active_model)
        return response

    @app.post("/api/v1/recommendations")
    def post_recommendations(request: BatchRecommendationRequest) -> dict:
        _validate_agent(request.agent_name, request.agent_token)
        normalized_model = _normalize_model_name(request.model_name)
        verbose_flag = _resolve_verbose_flag(verbose=request.verbose, verborse=request.verborse)
        requested_weights = _weights_to_payload(request.weights)
        effective_weights, active_model = _resolve_effective_weights(
            request.agent_name,
            request.agent_token,
            requested_weights,
            normalized_model,
        )
        response = _build_batch_recommendation_response(
            request.stocks,
            effective_weights,
            verbose=verbose_flag,
            model_name=active_model,
            asset_type=request.asset_type,
        )
        if requested_weights is not None:
            _save_agent_weights(
                request.agent_name,
                request.agent_token,
                requested_weights,
                model_name=active_model,
            )
        return response

    @app.get("/api/v1/scoring-data")
    def get_scoring_data(
        stock: str = Query(..., description="Ticker symbol or company name"),
        agent_name: str = Query(..., description="Registered agent name"),
        agent_token: str = Query(..., description="Registered agent token"),
        asset_type: str = Query(
            default="stock",
            description="Asset type: stock, crypto, option, or future",
        ),
        model_name: Optional[str] = Query(
            default=None,
            description="Optional personalized model name",
        ),
        weights: Optional[str] = Query(
            default=None,
            description="Optional JSON-encoded scoring weights",
        ),
    ) -> dict:
        _validate_agent(agent_name, agent_token)
        normalized_model = _normalize_model_name(model_name)
        weights_payload = _parse_weights_query(weights)
        effective_weights, active_model = _resolve_effective_weights(
            agent_name,
            agent_token,
            weights_payload,
            normalized_model,
        )
        response = _build_scoring_data_response(stock, effective_weights, model_name=active_model, asset_type=asset_type)
        if weights_payload is not None:
            _save_agent_weights(agent_name, agent_token, weights_payload, model_name=active_model)
        return response

    @app.post("/api/v1/scoring-data")
    def post_scoring_data(request: ScoringDataRequest) -> dict:
        _validate_agent(request.agent_name, request.agent_token)
        normalized_model = _normalize_model_name(request.model_name)
        requested_weights = _weights_to_payload(request.weights)
        effective_weights, active_model = _resolve_effective_weights(
            request.agent_name,
            request.agent_token,
            requested_weights,
            normalized_model,
        )
        response = _build_scoring_data_response(
            request.stock,
            effective_weights,
            model_name=active_model,
            asset_type=request.asset_type,
        )
        if requested_weights is not None:
            _save_agent_weights(
                request.agent_name,
                request.agent_token,
                requested_weights,
                model_name=active_model,
            )
        return response

    @app.get("/api/v1/congress/trades")
    def get_congress_trades(
        year: Optional[int] = Query(default=None, description="Year to fetch trades for"),
        chamber: str = Query(default="all", description="senate, house, or all"),
        symbol: Optional[str] = Query(default=None, description="Filter by ticker symbol"),
        politician: Optional[str] = Query(default=None, description="Filter by politician name"),
    ) -> dict:
        from .congress import fetch_trades
        import datetime as _dt

        effective_year = year or _dt.datetime.now().year
        trades = fetch_trades(year=effective_year, chamber=chamber)

        if symbol:
            sym_upper = symbol.strip().upper()
            trades = [t for t in trades if t.get("symbol") == sym_upper]
        if politician:
            pol_lower = politician.strip().lower()
            trades = [t for t in trades if pol_lower in (t.get("politician") or "").lower()]

        return {
            "year": effective_year,
            "chamber": chamber,
            "total": len(trades),
            "trades": trades[:200],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/v1/congress/roi")
    def get_congress_roi(
        year: Optional[int] = Query(default=None, description="Year for ROI report"),
        chamber: str = Query(default="all", description="senate, house, or all"),
        top_n: int = Query(default=10, ge=1, le=50, description="Number of top performers"),
    ) -> dict:
        from .congress import yearly_report
        import datetime as _dt

        effective_year = year or _dt.datetime.now().year
        return yearly_report(year=effective_year, chamber=chamber, top_n=top_n)

    @app.get("/api/v1/congress/seasonal")
    def get_congress_seasonal(
        year: Optional[int] = Query(default=None, description="Year for seasonal analysis"),
        chamber: str = Query(default="all", description="senate, house, or all"),
    ) -> dict:
        from .congress import fetch_trades, compute_trade_roi, seasonal_summary
        import datetime as _dt

        effective_year = year or _dt.datetime.now().year
        trades = fetch_trades(year=effective_year, chamber=chamber)
        trades = compute_trade_roi(trades)
        return seasonal_summary(trades, year=effective_year)

    @app.get("/api/v1/options/chain")
    def get_options_chain_endpoint(
        symbol: str = Query(..., description="Underlying stock symbol"),
        expiry: Optional[str] = Query(default=None, description="Expiration date (YYYY-MM-DD)"),
    ) -> dict:
        from .options import get_options_chain
        return get_options_chain(symbol, expiry=expiry)

    return app


def _db_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "data" / "agents_db.txt"


def _weights_db_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "data" / "weights.txt"


def _ensure_db_exists() -> None:
    db_path = _db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        db_path.write_text('{"agents": {}}', encoding="utf-8")


def _ensure_weights_db_exists() -> None:
    db_path = _weights_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        db_path.write_text('{"agents": {}}', encoding="utf-8")


def _load_agents() -> Dict[str, Dict[str, str]]:
    _ensure_db_exists()
    raw = _db_path().read_text(encoding="utf-8").strip()
    if not raw:
        return {}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Agent DB is corrupted: {exc}") from exc

    agents = payload.get("agents", {})
    if not isinstance(agents, dict):
        raise HTTPException(status_code=500, detail="Agent DB has invalid format")
    return agents


def _save_agents(agents: Dict[str, Dict[str, str]]) -> None:
    db_path = _db_path()
    payload = {"agents": agents}
    tmp_path = db_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(db_path)


def _load_weights_agents() -> Dict[str, Dict[str, object]]:
    _ensure_weights_db_exists()
    raw = _weights_db_path().read_text(encoding="utf-8").strip()
    if not raw:
        return {}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Weights DB is corrupted: {exc}") from exc

    agents = payload.get("agents", {})
    if not isinstance(agents, dict):
        raise HTTPException(status_code=500, detail="Weights DB has invalid format")
    return agents


def _save_weights_agents(agents: Dict[str, Dict[str, object]]) -> None:
    db_path = _weights_db_path()
    payload = {"agents": agents}
    tmp_path = db_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(db_path)


def _sanitize_weights(weights: object) -> Optional[dict]:
    if not isinstance(weights, dict):
        return None
    try:
        parsed = ScoringWeights(**weights)
    except ValidationError:
        return None
    try:
        return _weights_to_payload(parsed)
    except HTTPException:
        return None


def _normalize_agent_weights_record(record: object, fallback_agent_name: str) -> Optional[dict]:
    if not isinstance(record, dict):
        return None

    agent_name = str(record.get("agent_name") or fallback_agent_name).strip()
    agent_token = str(record.get("agent_token") or "").strip()
    default_model = str(record.get("default_model") or _DEFAULT_MODEL_NAME).strip() or _DEFAULT_MODEL_NAME

    models: Dict[str, Dict[str, object]] = {}
    raw_models = record.get("models")
    if isinstance(raw_models, dict):
        for model_key, model_data in raw_models.items():
            model_name = _normalize_model_name(str(model_key))
            if model_name is None or not isinstance(model_data, dict):
                continue
            sanitized = _sanitize_weights(model_data.get("weights"))
            if sanitized is None:
                continue
            models[model_name] = {
                "weights": sanitized,
                "updated_at": model_data.get("updated_at"),
            }

    # Backward compatibility for legacy one-model schema.
    legacy_weights = _sanitize_weights(record.get("weights"))
    if legacy_weights is not None and default_model not in models:
        models[default_model] = {
            "weights": legacy_weights,
            "updated_at": record.get("updated_at"),
        }

    if default_model not in models:
        if models:
            default_model = next(iter(models.keys()))
        else:
            default_model = _DEFAULT_MODEL_NAME

    return {
        "agent_name": agent_name or fallback_agent_name,
        "agent_token": agent_token,
        "default_model": default_model,
        "models": models,
    }


def _get_default_model_name(agent_name: str, agent_token: str) -> str:
    cleaned_name = agent_name.strip()
    cleaned_token = agent_token.strip()
    if not cleaned_name or not cleaned_token:
        return _DEFAULT_MODEL_NAME

    with _DB_LOCK:
        agents = _load_weights_agents()
        record = agents.get(cleaned_name)

    normalized = _normalize_agent_weights_record(record, cleaned_name)
    if not normalized:
        return _DEFAULT_MODEL_NAME
    expected_token = normalized.get("agent_token", "")
    if not expected_token or not secrets.compare_digest(str(expected_token), cleaned_token):
        return _DEFAULT_MODEL_NAME
    return str(normalized.get("default_model") or _DEFAULT_MODEL_NAME)


def _get_saved_agent_weights(
    agent_name: str,
    agent_token: str,
    model_name: Optional[str] = None,
) -> tuple[Optional[dict], str]:
    cleaned_name = agent_name.strip()
    cleaned_token = agent_token.strip()
    active_model = model_name or _DEFAULT_MODEL_NAME
    if not cleaned_name or not cleaned_token:
        return None, active_model

    with _DB_LOCK:
        agents = _load_weights_agents()
        record = agents.get(cleaned_name)

    normalized = _normalize_agent_weights_record(record, cleaned_name)
    if not normalized:
        return None, active_model

    expected_token = str(normalized.get("agent_token", ""))
    if not expected_token or not secrets.compare_digest(expected_token, cleaned_token):
        return None, active_model

    active_model = model_name or str(normalized.get("default_model") or _DEFAULT_MODEL_NAME)
    model_entry = (normalized.get("models") or {}).get(active_model)
    if not isinstance(model_entry, dict):
        return None, active_model
    weights = _sanitize_weights(model_entry.get("weights"))
    return weights, active_model


def _save_agent_weights(
    agent_name: str,
    agent_token: str,
    weights: dict,
    model_name: Optional[str] = None,
) -> None:
    cleaned_name = agent_name.strip()
    cleaned_token = agent_token.strip()
    sanitized = _sanitize_weights(weights)
    normalized_model = _normalize_model_name(model_name) if model_name else None
    if not cleaned_name or not cleaned_token or sanitized is None:
        return

    with _DB_LOCK:
        agents = _load_weights_agents()
        existing = _normalize_agent_weights_record(agents.get(cleaned_name), cleaned_name) or {
            "agent_name": cleaned_name,
            "agent_token": cleaned_token,
            "default_model": _DEFAULT_MODEL_NAME,
            "models": {},
        }
        active_model = normalized_model or str(existing.get("default_model") or _DEFAULT_MODEL_NAME)
        existing["agent_name"] = cleaned_name
        existing["agent_token"] = cleaned_token
        existing["default_model"] = str(existing.get("default_model") or _DEFAULT_MODEL_NAME)
        existing.setdefault("models", {})
        existing["models"][active_model] = {
            "weights": sanitized,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        agents[cleaned_name] = existing
        _save_weights_agents(agents)


def _resolve_effective_weights(
    agent_name: str,
    agent_token: str,
    requested_weights: Optional[dict],
    model_name: Optional[str] = None,
) -> tuple[Optional[dict], str]:
    active_model = model_name or _get_default_model_name(agent_name, agent_token)
    if requested_weights is not None:
        return requested_weights, active_model

    saved, resolved_model = _get_saved_agent_weights(
        agent_name,
        agent_token,
        model_name=active_model,
    )
    if model_name and saved is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found for agent.")
    return saved, resolved_model


def _generate_unique_token(agents: Dict[str, Dict[str, str]]) -> str:
    existing_tokens = {record.get("token", "") for record in agents.values()}
    while True:
        token = secrets.token_urlsafe(24)
        if token not in existing_tokens:
            return token


def _register_agent(name: str) -> dict:
    cleaned_name = name.strip()
    if not cleaned_name:
        raise HTTPException(status_code=400, detail="Agent name cannot be empty")
    if len(cleaned_name) > _MAX_AGENT_NAME_LEN:
        raise HTTPException(status_code=400, detail="Agent name is too long")
    if not _AGENT_NAME_PATTERN.fullmatch(cleaned_name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Agent name must start with a letter/number and use only "
                "letters, numbers, '.', '_', or '-'."
            ),
        )

    with _DB_LOCK:
        agents = _load_agents()
        if cleaned_name in agents:
            raise HTTPException(
                status_code=409,
                detail=f"Agent '{cleaned_name}' is already registered",
            )

        token = _generate_unique_token(agents)
        created_at = datetime.now(timezone.utc).isoformat()
        agents[cleaned_name] = {"token": token, "created_at": created_at}
        _save_agents(agents)

    return {"name": cleaned_name, "token": token, "created_at": created_at}


def _validate_agent(name: str, token: str) -> None:
    cleaned_name = name.strip()
    cleaned_token = token.strip()
    if not cleaned_name or not cleaned_token:
        raise HTTPException(status_code=401, detail="Agent name and token are required")
    if len(cleaned_name) > _MAX_AGENT_NAME_LEN or len(cleaned_token) > _MAX_AGENT_TOKEN_LEN:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not _AGENT_NAME_PATTERN.fullmatch(cleaned_name):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    with _DB_LOCK:
        agents = _load_agents()
        record = agents.get(cleaned_name)

    if not record:
        raise HTTPException(status_code=401, detail="Unknown agent")

    expected_token = record.get("token", "")
    if not expected_token or not secrets.compare_digest(expected_token, cleaned_token):
        raise HTTPException(status_code=401, detail="Invalid agent token")


def _build_recommendation_response(
    stock: str,
    weights: Optional[dict] = None,
    verbose: bool = False,
    model_name: Optional[str] = None,
    asset_type: str = "stock",
) -> dict:
    stock = stock.strip()
    if not stock:
        raise HTTPException(status_code=400, detail="Stock input is required")
    if len(stock) > _MAX_STOCK_INPUT_LEN:
        raise HTTPException(status_code=400, detail="Stock input is too long")

    at = (asset_type or "stock").lower()
    resolved_symbol = _resolve_symbol(stock, asset_type=at)
    if not resolved_symbol:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not resolve input to a ticker. "
                "Provide a valid ticker (for example, AAPL, BTC-USD, ES=F) or a company name."
            ),
        )

    try:
        analysis = generate_full_analysis(resolved_symbol, weights=weights, asset_type=at)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to generate recommendation for {resolved_symbol}",
        ) from exc

    if analysis.get("ai_recommendation") is None:
        raise HTTPException(
            status_code=502,
            detail=f"Recommendation unavailable for {resolved_symbol}",
        )

    action = analysis["ai_recommendation"].get("action")
    if not action:
        raise HTTPException(
            status_code=502,
            detail=f"Recommendation action unavailable for {resolved_symbol}",
        )

    if not verbose:
        return {"recommendation": action}

    return {
        "input": stock,
        "resolved_symbol": resolved_symbol,
        "asset_type": at,
        "company": analysis.get("company"),
        "stock_analysis": analysis.get("stock_analysis"),
        "sentiment_analysis": analysis.get("sentiment_analysis"),
        "social_analysis": analysis.get("social_analysis"),
        "media_analysis": analysis.get("media_analysis"),
        "ai_recommendation": analysis.get("ai_recommendation"),
        "scoring_weights": analysis.get("scoring_weights"),
        "model_name": model_name or _DEFAULT_MODEL_NAME,
        "generated_at": analysis.get("generated_at"),
    }


def _parse_batch_stocks(stocks: str) -> list[str]:
    # Accept comma-separated values from GET query and normalize whitespace.
    entries = [item.strip() for item in stocks.split(",")]
    values = [item for item in entries if item]
    if not values:
        raise HTTPException(status_code=400, detail="At least one stock is required")
    if len(values) > 25:
        raise HTTPException(status_code=400, detail="Maximum 25 stocks per request")
    return values


def _build_batch_recommendation_response(
    stocks: list[str],
    weights: Optional[dict] = None,
    verbose: bool = False,
    model_name: Optional[str] = None,
    asset_type: str = "stock",
) -> dict:
    if not stocks:
        raise HTTPException(status_code=400, detail="At least one stock is required")
    if len(stocks) > 25:
        raise HTTPException(status_code=400, detail="Maximum 25 stocks per request")

    results = []
    for stock in stocks:
        try:
            result = _build_recommendation_response(
                stock,
                weights=weights,
                verbose=verbose,
                model_name=model_name,
                asset_type=asset_type,
            )
            result["status"] = "ok"
            results.append(result)
        except HTTPException as exc:
            results.append(
                {
                    "input": stock,
                    "status": "error",
                    "error": {
                        "code": exc.status_code,
                        "message": exc.detail,
                    },
                }
            )

    success_count = sum(1 for item in results if item.get("status") == "ok")
    error_count = len(results) - success_count
    return {
        "total": len(results),
        "success": success_count,
        "errors": error_count,
        "results": results,
    }


def _build_scoring_data_response(
    stock: str,
    weights: Optional[dict] = None,
    model_name: Optional[str] = None,
    asset_type: str = "stock",
) -> dict:
    stock = stock.strip()
    if not stock:
        raise HTTPException(status_code=400, detail="Stock input is required")
    if len(stock) > _MAX_STOCK_INPUT_LEN:
        raise HTTPException(status_code=400, detail="Stock input is too long")

    at = (asset_type or "stock").lower()
    resolved_symbol = _resolve_symbol(stock, asset_type=at)
    if not resolved_symbol:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not resolve input to a ticker. "
                "Provide a valid ticker (for example, AAPL, BTC-USD, ES=F) or a company name."
            ),
        )

    try:
        data = generate_scoring_data(resolved_symbol, weights=weights, asset_type=at)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to load scoring data for {resolved_symbol}",
        ) from exc

    return {
        "input": stock,
        "resolved_symbol": resolved_symbol,
        "asset_type": at,
        "company": data.get("company"),
        "price": data.get("price"),
        "snapshot": data.get("snapshot"),
        "scoring_inputs": data.get("scoring_inputs"),
        "scoring_weights": data.get("scoring_weights"),
        "model_name": model_name or _DEFAULT_MODEL_NAME,
        "generated_at": data.get("generated_at"),
    }


_rebuild_forward_refs()
app = create_app()
