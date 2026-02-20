import re
import json
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError

from .web_analyzer import generate_full_analysis, generate_scoring_data

_DB_LOCK = threading.Lock()
_MAX_STOCK_INPUT_LEN = 120
_MAX_AGENT_NAME_LEN = 64
_MAX_AGENT_TOKEN_LEN = 256
_AGENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-\.]{0,63}$")


class RecommendationRequest(BaseModel):
    stock: str = Field(..., description="Ticker symbol or company name")
    agent_name: str = Field(..., description="Registered agent name")
    agent_token: str = Field(..., description="Registered agent token")
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


def _looks_like_ticker(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9\.\-]{0,9}", value.strip()))


def _resolve_symbol(stock: str) -> Optional[str]:
    candidate = stock.strip()
    if not candidate or len(candidate) > _MAX_STOCK_INPUT_LEN:
        return None

    if _looks_like_ticker(candidate):
        return candidate.upper()

    try:
        import yfinance as yf  # Optional runtime dependency for name lookup.

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
        weights_payload = _parse_weights_query(weights)
        verbose_flag = _resolve_verbose_flag(verbose=verbose, verborse=verborse)
        effective_weights = _resolve_effective_weights(agent_name, agent_token, weights_payload)
        response = _build_recommendation_response(
            stock,
            effective_weights,
            verbose=verbose_flag,
        )
        if weights_payload is not None:
            _save_agent_weights(agent_name, agent_token, weights_payload)
        return response

    @app.post("/api/v1/recommendation")
    def post_recommendation(request: RecommendationRequest) -> dict:
        _validate_agent(request.agent_name, request.agent_token)
        verbose_flag = _resolve_verbose_flag(verbose=request.verbose, verborse=request.verborse)
        requested_weights = _weights_to_payload(request.weights)
        effective_weights = _resolve_effective_weights(
            request.agent_name,
            request.agent_token,
            requested_weights,
        )
        response = _build_recommendation_response(
            request.stock,
            effective_weights,
            verbose=verbose_flag,
        )
        if requested_weights is not None:
            _save_agent_weights(request.agent_name, request.agent_token, requested_weights)
        return response

    @app.get("/api/v1/recommendations")
    def get_recommendations(
        stocks: str = Query(..., description="Comma-separated ticker symbols or company names"),
        agent_name: str = Query(..., description="Registered agent name"),
        agent_token: str = Query(..., description="Registered agent token"),
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
        parsed_stocks = _parse_batch_stocks(stocks)
        weights_payload = _parse_weights_query(weights)
        verbose_flag = _resolve_verbose_flag(verbose=verbose, verborse=verborse)
        effective_weights = _resolve_effective_weights(agent_name, agent_token, weights_payload)
        response = _build_batch_recommendation_response(
            parsed_stocks,
            effective_weights,
            verbose=verbose_flag,
        )
        if weights_payload is not None:
            _save_agent_weights(agent_name, agent_token, weights_payload)
        return response

    @app.post("/api/v1/recommendations")
    def post_recommendations(request: BatchRecommendationRequest) -> dict:
        _validate_agent(request.agent_name, request.agent_token)
        verbose_flag = _resolve_verbose_flag(verbose=request.verbose, verborse=request.verborse)
        requested_weights = _weights_to_payload(request.weights)
        effective_weights = _resolve_effective_weights(
            request.agent_name,
            request.agent_token,
            requested_weights,
        )
        response = _build_batch_recommendation_response(
            request.stocks,
            effective_weights,
            verbose=verbose_flag,
        )
        if requested_weights is not None:
            _save_agent_weights(request.agent_name, request.agent_token, requested_weights)
        return response

    @app.get("/api/v1/scoring-data")
    def get_scoring_data(
        stock: str = Query(..., description="Ticker symbol or company name"),
        agent_name: str = Query(..., description="Registered agent name"),
        agent_token: str = Query(..., description="Registered agent token"),
        weights: Optional[str] = Query(
            default=None,
            description="Optional JSON-encoded scoring weights",
        ),
    ) -> dict:
        _validate_agent(agent_name, agent_token)
        weights_payload = _parse_weights_query(weights)
        effective_weights = _resolve_effective_weights(agent_name, agent_token, weights_payload)
        response = _build_scoring_data_response(stock, effective_weights)
        if weights_payload is not None:
            _save_agent_weights(agent_name, agent_token, weights_payload)
        return response

    @app.post("/api/v1/scoring-data")
    def post_scoring_data(request: ScoringDataRequest) -> dict:
        _validate_agent(request.agent_name, request.agent_token)
        requested_weights = _weights_to_payload(request.weights)
        effective_weights = _resolve_effective_weights(
            request.agent_name,
            request.agent_token,
            requested_weights,
        )
        response = _build_scoring_data_response(
            request.stock,
            effective_weights,
        )
        if requested_weights is not None:
            _save_agent_weights(request.agent_name, request.agent_token, requested_weights)
        return response

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


def _get_saved_agent_weights(agent_name: str, agent_token: str) -> Optional[dict]:
    cleaned_name = agent_name.strip()
    cleaned_token = agent_token.strip()
    if not cleaned_name or not cleaned_token:
        return None

    with _DB_LOCK:
        agents = _load_weights_agents()
        record = agents.get(cleaned_name)

    if not isinstance(record, dict):
        return None
    expected_token = str(record.get("agent_token", ""))
    if not expected_token or not secrets.compare_digest(expected_token, cleaned_token):
        return None
    return _sanitize_weights(record.get("weights"))


def _save_agent_weights(agent_name: str, agent_token: str, weights: dict) -> None:
    cleaned_name = agent_name.strip()
    cleaned_token = agent_token.strip()
    sanitized = _sanitize_weights(weights)
    if not cleaned_name or not cleaned_token or sanitized is None:
        return

    with _DB_LOCK:
        agents = _load_weights_agents()
        agents[cleaned_name] = {
            "agent_name": cleaned_name,
            "agent_token": cleaned_token,
            "weights": sanitized,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_weights_agents(agents)


def _resolve_effective_weights(
    agent_name: str,
    agent_token: str,
    requested_weights: Optional[dict],
) -> Optional[dict]:
    if requested_weights is not None:
        return requested_weights
    return _get_saved_agent_weights(agent_name, agent_token)


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
) -> dict:
    stock = stock.strip()
    if not stock:
        raise HTTPException(status_code=400, detail="Stock input is required")
    if len(stock) > _MAX_STOCK_INPUT_LEN:
        raise HTTPException(status_code=400, detail="Stock input is too long")

    resolved_symbol = _resolve_symbol(stock)
    if not resolved_symbol:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not resolve stock input to a ticker. "
                "Provide a valid ticker (for example, AAPL) or a company name."
            ),
        )

    try:
        analysis = generate_full_analysis(resolved_symbol, weights=weights)
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
        "company": analysis.get("company"),
        "stock_analysis": analysis.get("stock_analysis"),
        "sentiment_analysis": analysis.get("sentiment_analysis"),
        "ai_recommendation": analysis.get("ai_recommendation"),
        "scoring_weights": analysis.get("scoring_weights"),
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


def _build_scoring_data_response(stock: str, weights: Optional[dict] = None) -> dict:
    stock = stock.strip()
    if not stock:
        raise HTTPException(status_code=400, detail="Stock input is required")
    if len(stock) > _MAX_STOCK_INPUT_LEN:
        raise HTTPException(status_code=400, detail="Stock input is too long")

    resolved_symbol = _resolve_symbol(stock)
    if not resolved_symbol:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not resolve stock input to a ticker. "
                "Provide a valid ticker (for example, AAPL) or a company name."
            ),
        )

    try:
        data = generate_scoring_data(resolved_symbol, weights=weights)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to load scoring data for {resolved_symbol}",
        ) from exc

    return {
        "input": stock,
        "resolved_symbol": resolved_symbol,
        "company": data.get("company"),
        "price": data.get("price"),
        "snapshot": data.get("snapshot"),
        "scoring_inputs": data.get("scoring_inputs"),
        "scoring_weights": data.get("scoring_weights"),
        "generated_at": data.get("generated_at"),
    }


_rebuild_forward_refs()
app = create_app()
