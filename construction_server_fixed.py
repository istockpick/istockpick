#!/usr/bin/env python3
"""
Enhanced Web Server with In Construction Page - Fixed for External Access
Binds to all interfaces (0.0.0.0) instead of just localhost
"""

import http.server
import socketserver
import datetime
import json
import re
import os
import sys
import subprocess
import threading
import secrets
import importlib.util
import urllib.error
import urllib.request
import urllib.parse
from urllib.parse import urlparse, parse_qs

DEFAULT_STOCK_ANALYST_PATH = "/Users/richliu/projects/private/istockpick/stock-analyst"
STOCK_ANALYST_PATH = os.path.realpath(
    os.environ.get("STOCK_ANALYST_PATH", DEFAULT_STOCK_ANALYST_PATH)
)
LEGACY_STOCK_ANALYST_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "stock-analyst")
)
ANALYSIS_RUNTIME_PYTHON = os.getenv(
    "ANALYSIS_RUNTIME_PYTHON",
    "/tmp/stock-analyst-venv/bin/python3",
)

for p in (STOCK_ANALYST_PATH, LEGACY_STOCK_ANALYST_PATH):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from stock_analyst.web_analyzer import generate_full_analysis, generate_scoring_data
    ANALYSIS_IMPORT_ERROR = None
except Exception as exc:
    generate_full_analysis = None
    generate_scoring_data = None
    ANALYSIS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    for root in (STOCK_ANALYST_PATH, LEGACY_STOCK_ANALYST_PATH):
        module_path = os.path.join(root, "stock_analyst", "web_analyzer.py")
        if not os.path.isfile(module_path):
            continue
        try:
            spec = importlib.util.spec_from_file_location("stock_analyst_web_analyzer", module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                generate_full_analysis = module.generate_full_analysis
                generate_scoring_data = getattr(module, "generate_scoring_data", None)
                ANALYSIS_IMPORT_ERROR = None
                break
        except Exception as load_exc:
            ANALYSIS_IMPORT_ERROR = f"{type(load_exc).__name__}: {load_exc}"


def _looks_like_ticker(value):
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9\.\-]{0,9}", (value or "").strip()))


def _resolve_symbol_from_input(stock):
    candidate = (stock or "").strip()
    if not candidate or len(candidate) > 120:
        return None

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


def _parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _parse_weights_payload(raw_weights):
    if raw_weights in (None, ""):
        return None
    if isinstance(raw_weights, dict):
        return raw_weights
    if isinstance(raw_weights, str):
        try:
            payload = json.loads(raw_weights)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


class ConstructionHandler(http.server.SimpleHTTPRequestHandler):
    SEC_TICKERS_CACHE = None
    SEC_TICKERS_CACHE_LOCK = threading.Lock()
    AGENTS_DB_LOCK = threading.Lock()
    WEIGHTS_DB_LOCK = threading.Lock()
    MAX_PROXY_BODY_BYTES = 1_000_000
    MAX_LOOKUP_QUERY_LEN = 120
    STOCK_QUERY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9\.\-]{0,9}$")
    AGENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-\.]{0,63}$")

    def end_headers(self):
        # Baseline hardening headers for every response.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        super().end_headers()

    def do_GET(self):
        parsed_path = urlparse(self.path)
        query = parse_qs(parsed_path.query)

        if parsed_path.path == '/':
            self.serve_construction_page()
        elif parsed_path.path == '/health':
            self.serve_health_check()
        elif parsed_path.path == '/api/v1/recommendation':
            self.serve_api_recommendation_get(query)
        elif parsed_path.path == '/api/v1/scoring-data':
            self.serve_api_scoring_data_get(query)
        elif parsed_path.path == '/status':
            self.serve_status()
        elif parsed_path.path == '/lookup':
            lookup_value = query.get('q', [''])[0]
            self.serve_stock_lookup(lookup_value)
        elif parsed_path.path == '/analyze':
            lookup_value = query.get('q', [''])[0]
            self.serve_stock_analysis(lookup_value)
        elif parsed_path.path == '/SKILL.md':
            self.serve_skill_markdown()
        else:
            self.serve_404()

    def do_POST(self):
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/api/v1/agents/register':
            self.serve_api_register_agent()
            return
        if parsed_path.path == '/api/v1/recommendation':
            self.serve_api_recommendation_post()
            return
        if parsed_path.path == '/api/v1/scoring-data':
            self.serve_api_scoring_data_post()
            return

        self.send_json(404, {"error": "Unknown endpoint."})

    def _read_json_body(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            self.send_json(400, {"error": "Invalid Content-Length header."})
            return None
        if content_length < 0:
            self.send_json(400, {"error": "Invalid Content-Length header."})
            return None
        if content_length > self.MAX_PROXY_BODY_BYTES:
            self.send_json(413, {"error": "Request body too large."})
            return None

        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json(400, {"error": "Invalid JSON payload."})
            return None

        if not isinstance(payload, dict):
            self.send_json(400, {"error": "JSON object payload is required."})
            return None
        return payload

    def _agents_db_path(self):
        for root in (STOCK_ANALYST_PATH, LEGACY_STOCK_ANALYST_PATH):
            if os.path.isdir(root):
                return os.path.join(root, "data", "agents_db.txt")
        return os.path.join(LEGACY_STOCK_ANALYST_PATH, "data", "agents_db.txt")

    def _weights_db_path(self):
        return os.path.join(os.path.dirname(__file__), "stock-analyst", "data", "weights.txt")

    def _load_agents(self):
        db_path = self._agents_db_path()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        if not os.path.exists(db_path):
            with open(db_path, "w", encoding="utf-8") as f:
                f.write('{"agents": {}}')

        with open(db_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError("Agent DB is corrupted")
        agents = payload.get("agents", {})
        if not isinstance(agents, dict):
            raise ValueError("Agent DB has invalid format")
        return agents

    def _save_agents(self, agents):
        db_path = self._agents_db_path()
        payload = {"agents": agents}
        tmp_path = f"{db_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2))
        os.replace(tmp_path, db_path)

    def _load_weights_db(self):
        db_path = self._weights_db_path()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        if not os.path.exists(db_path):
            with open(db_path, "w", encoding="utf-8") as f:
                f.write('{"agents": {}}')

        with open(db_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError("Weights DB is corrupted")
        agents = payload.get("agents", {})
        if not isinstance(agents, dict):
            raise ValueError("Weights DB has invalid format")
        return agents

    def _save_weights_db(self, agents):
        db_path = self._weights_db_path()
        payload = {"agents": agents}
        tmp_path = f"{db_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2))
        os.replace(tmp_path, db_path)

    def _get_saved_weights(self, agent_name, agent_token):
        with ConstructionHandler.WEIGHTS_DB_LOCK:
            try:
                agents = self._load_weights_db()
            except ValueError:
                return None
            record = agents.get((agent_name or "").strip())
            if not isinstance(record, dict):
                return None
            expected_token = str(record.get("agent_token", ""))
            if not expected_token or not secrets.compare_digest(expected_token, (agent_token or "").strip()):
                return None
            weights = record.get("weights")
            return weights if isinstance(weights, dict) else None

    def _save_agent_weights(self, agent_name, agent_token, weights):
        if not isinstance(weights, dict):
            return
        cleaned_name = (agent_name or "").strip()
        cleaned_token = (agent_token or "").strip()
        if not cleaned_name or not cleaned_token:
            return

        with ConstructionHandler.WEIGHTS_DB_LOCK:
            try:
                agents = self._load_weights_db()
            except ValueError:
                agents = {}

            agents[cleaned_name] = {
                "agent_name": cleaned_name,
                "agent_token": cleaned_token,
                "weights": weights,
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            self._save_weights_db(agents)

    def _register_agent(self, name):
        cleaned_name = (name or "").strip()
        if not cleaned_name:
            self.send_json(400, {"error": "Agent name cannot be empty"})
            return
        if len(cleaned_name) > 64:
            self.send_json(400, {"error": "Agent name is too long"})
            return
        if not self.AGENT_NAME_PATTERN.fullmatch(cleaned_name):
            self.send_json(400, {"error": "Invalid agent name format"})
            return

        with ConstructionHandler.AGENTS_DB_LOCK:
            try:
                agents = self._load_agents()
            except ValueError:
                self.send_json(500, {"error": "Agent database unavailable"})
                return

            if cleaned_name in agents:
                self.send_json(409, {"error": f"Agent '{cleaned_name}' is already registered"})
                return

            existing_tokens = {record.get("token", "") for record in agents.values()}
            token = ""
            while not token or token in existing_tokens:
                token = secrets.token_urlsafe(24)

            created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            agents[cleaned_name] = {"token": token, "created_at": created_at}
            self._save_agents(agents)

        self.send_json(200, {"name": cleaned_name, "token": token, "created_at": created_at})

    def _validate_agent(self, name, token):
        cleaned_name = (name or "").strip()
        cleaned_token = (token or "").strip()
        if not cleaned_name or not cleaned_token:
            self.send_json(401, {"error": "Agent name and token are required"})
            return False
        if len(cleaned_name) > 64 or len(cleaned_token) > 256:
            self.send_json(401, {"error": "Invalid credentials"})
            return False
        if not self.AGENT_NAME_PATTERN.fullmatch(cleaned_name):
            self.send_json(401, {"error": "Invalid credentials"})
            return False

        with ConstructionHandler.AGENTS_DB_LOCK:
            try:
                agents = self._load_agents()
            except ValueError:
                self.send_json(500, {"error": "Agent database unavailable"})
                return False
            record = agents.get(cleaned_name)

        if not record:
            self.send_json(401, {"error": "Unknown agent"})
            return False
        expected_token = record.get("token", "")
        if not expected_token or not secrets.compare_digest(expected_token, cleaned_token):
            self.send_json(401, {"error": "Invalid agent token"})
            return False
        return True

    def _build_recommendation_response(
        self,
        stock,
        weights=None,
        verbose=False,
        agent_name=None,
        agent_token=None,
        persist_weights=False,
    ):
        stock = (stock or "").strip()
        if not stock:
            self.send_json(400, {"error": "Stock input is required"})
            return
        if len(stock) > 120:
            self.send_json(400, {"error": "Stock input is too long"})
            return

        resolved_symbol = _resolve_symbol_from_input(stock)
        if not resolved_symbol:
            self.send_json(
                400,
                {
                    "error": (
                        "Could not resolve stock input to a ticker. "
                        "Provide a valid ticker (for example, AAPL) or a company name."
                    )
                },
            )
            return

        try:
            analysis = generate_full_analysis(resolved_symbol, weights=weights)
        except Exception:
            self.send_json(502, {"error": f"Failed to generate recommendation for {resolved_symbol}"})
            return
        if analysis.get("ai_recommendation") is None:
            self.send_json(502, {"error": f"Recommendation unavailable for {resolved_symbol}"})
            return

        action = analysis["ai_recommendation"].get("action")
        if not action:
            self.send_json(502, {"error": f"Recommendation action unavailable for {resolved_symbol}"})
            return

        if persist_weights:
            self._save_agent_weights(agent_name, agent_token, weights)

        if not verbose:
            self.send_json(200, {"recommendation": action})
            return

        self.send_json(
            200,
            {
                "input": stock,
                "resolved_symbol": resolved_symbol,
                "company": analysis.get("company"),
                "stock_analysis": analysis.get("stock_analysis"),
                "sentiment_analysis": analysis.get("sentiment_analysis"),
                "ai_recommendation": analysis.get("ai_recommendation"),
                "scoring_weights": analysis.get("scoring_weights"),
                "generated_at": analysis.get("generated_at"),
            },
        )

    def serve_api_register_agent(self):
        payload = self._read_json_body()
        if payload is None:
            return
        self._register_agent(payload.get("name", ""))

    def serve_api_recommendation_get(self, query):
        stock = query.get("stock", [""])[0]
        agent_name = query.get("agent_name", [""])[0]
        agent_token = query.get("agent_token", [""])[0]
        raw_weights = query.get("weights", [""])[0]
        requested_weights = _parse_weights_payload(raw_weights)
        if raw_weights not in ("", None) and requested_weights is None:
            self.send_json(400, {"error": "Invalid weights payload. Expected a JSON object."})
            return
        verbose = _parse_bool(query.get("verbose", [""])[0], default=False)
        if "verborse" in query:
            verbose = _parse_bool(query.get("verborse", [""])[0], default=verbose)

        if not self._validate_agent(agent_name, agent_token):
            return
        weights = requested_weights if requested_weights is not None else self._get_saved_weights(agent_name, agent_token)
        self._build_recommendation_response(
            stock,
            weights=weights,
            verbose=verbose,
            agent_name=agent_name,
            agent_token=agent_token,
            persist_weights=requested_weights is not None,
        )

    def serve_api_recommendation_post(self):
        payload = self._read_json_body()
        if payload is None:
            return

        if not self._validate_agent(payload.get("agent_name", ""), payload.get("agent_token", "")):
            return
        requested_weights = _parse_weights_payload(payload.get("weights"))
        if "weights" in payload and requested_weights is None:
            self.send_json(400, {"error": "Invalid weights payload. Expected a JSON object."})
            return
        verbose = _parse_bool(payload.get("verbose"), default=False)
        if "verborse" in payload:
            verbose = _parse_bool(payload.get("verborse"), default=verbose)
        agent_name = payload.get("agent_name", "")
        agent_token = payload.get("agent_token", "")
        weights = requested_weights if requested_weights is not None else self._get_saved_weights(agent_name, agent_token)
        self._build_recommendation_response(
            payload.get("stock", ""),
            weights=weights,
            verbose=verbose,
            agent_name=agent_name,
            agent_token=agent_token,
            persist_weights=requested_weights is not None,
        )

    def _build_scoring_data_response(
        self,
        stock,
        weights=None,
        agent_name=None,
        agent_token=None,
        persist_weights=False,
    ):
        stock = (stock or "").strip()
        if not stock:
            self.send_json(400, {"error": "Stock input is required"})
            return
        if len(stock) > 120:
            self.send_json(400, {"error": "Stock input is too long"})
            return

        resolved_symbol = _resolve_symbol_from_input(stock)
        if not resolved_symbol:
            self.send_json(
                400,
                {
                    "error": (
                        "Could not resolve stock input to a ticker. "
                        "Provide a valid ticker (for example, AAPL) or a company name."
                    )
                },
            )
            return

        if generate_scoring_data is None:
            self.send_json(503, {"error": "Scoring-data endpoint is unavailable on this runtime"})
            return

        try:
            data = generate_scoring_data(resolved_symbol, weights=weights)
        except Exception:
            self.send_json(502, {"error": f"Failed to load scoring data for {resolved_symbol}"})
            return

        if persist_weights:
            self._save_agent_weights(agent_name, agent_token, weights)

        self.send_json(
            200,
            {
                "input": stock,
                "resolved_symbol": resolved_symbol,
                "company": data.get("company"),
                "price": data.get("price"),
                "snapshot": data.get("snapshot"),
                "scoring_inputs": data.get("scoring_inputs"),
                "scoring_weights": data.get("scoring_weights"),
                "generated_at": data.get("generated_at"),
            },
        )

    def serve_api_scoring_data_get(self, query):
        stock = query.get("stock", [""])[0]
        agent_name = query.get("agent_name", [""])[0]
        agent_token = query.get("agent_token", [""])[0]
        raw_weights = query.get("weights", [""])[0]
        requested_weights = _parse_weights_payload(raw_weights)
        if raw_weights not in ("", None) and requested_weights is None:
            self.send_json(400, {"error": "Invalid weights payload. Expected a JSON object."})
            return

        if not self._validate_agent(agent_name, agent_token):
            return
        weights = requested_weights if requested_weights is not None else self._get_saved_weights(agent_name, agent_token)
        self._build_scoring_data_response(
            stock,
            weights=weights,
            agent_name=agent_name,
            agent_token=agent_token,
            persist_weights=requested_weights is not None,
        )

    def serve_api_scoring_data_post(self):
        payload = self._read_json_body()
        if payload is None:
            return

        if not self._validate_agent(payload.get("agent_name", ""), payload.get("agent_token", "")):
            return
        requested_weights = _parse_weights_payload(payload.get("weights"))
        if "weights" in payload and requested_weights is None:
            self.send_json(400, {"error": "Invalid weights payload. Expected a JSON object."})
            return
        agent_name = payload.get("agent_name", "")
        agent_token = payload.get("agent_token", "")
        weights = requested_weights if requested_weights is not None else self._get_saved_weights(agent_name, agent_token)
        self._build_scoring_data_response(
            payload.get("stock", ""),
            weights=weights,
            agent_name=agent_name,
            agent_token=agent_token,
            persist_weights=requested_weights is not None,
        )
    
    def serve_construction_page(self):
        """Serve the in construction page"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>iStockPick.ai</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #001f3f;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            text-align: center;
        }}
        
        .container {{
            max-width: 800px;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }}
        
        .construction-icon {{
            font-size: 4rem;
            margin-bottom: 1rem;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.1); }}
            100% {{ transform: scale(1); }}
        }}
        
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 1rem;
            font-weight: 300;
        }}
        
        .subtitle {{
            font-size: 1.2rem;
            margin-bottom: 2rem;
            opacity: 0.9;
        }}
        
        .features {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }}
        
        .feature {{
            background: rgba(255, 255, 255, 0.1);
            padding: 1.5rem;
            border-radius: 15px;
            backdrop-filter: blur(5px);
        }}
        
        .feature-icon {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}
        
        .feature h3 {{
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }}
        
        .feature p {{
            font-size: 0.9rem;
            opacity: 0.8;
        }}
        
        .lookup-card {{
            margin-top: 2rem;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            text-align: left;
        }}

        .lookup-title {{
            font-weight: 600;
            margin-bottom: 0.75rem;
        }}

        .lookup-form {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}

        .lookup-input {{
            flex: 1;
            min-width: 220px;
            padding: 0.7rem 0.9rem;
            border-radius: 8px;
            border: none;
            outline: none;
            font-size: 0.95rem;
        }}

        .lookup-button {{
            padding: 0.7rem 1rem;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-weight: 600;
            background: #00d4ff;
            color: #1f2a44;
        }}

        .lookup-result {{
            margin-top: 0.75rem;
            font-size: 0.92rem;
            min-height: 1.2em;
        }}

        .analysis-card {{
            margin-top: 1rem;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            display: none;
        }}

        .analysis-meta {{
            margin-top: 0.5rem;
            font-size: 0.82rem;
            opacity: 0.8;
        }}

        .spinner {{
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-top: 2px solid #ffffff;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            display: inline-block;
            vertical-align: middle;
            margin-right: 8px;
        }}

        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        .analysis-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.75rem;
            margin-top: 0.75rem;
        }}

        .analysis-item {{
            background: rgba(255, 255, 255, 0.12);
            border-radius: 10px;
            padding: 0.75rem;
            font-size: 0.9rem;
        }}

        .status {{
            margin-top: 2rem;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }}
        
        .footer {{
            margin-top: 2rem;
            opacity: 0.7;
            font-size: 0.9rem;
        }}

        .command-card {{
            margin-top: 1rem;
            padding: 0.85rem;
            background: rgba(255, 255, 255, 0.12);
            border-radius: 10px;
            text-align: left;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            word-break: break-all;
        }}
        
        .loading-bar {{
            width: 100%;
            height: 4px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 2px;
            overflow: hidden;
            margin: 1rem 0;
        }}
        
        .loading-progress {{
            height: 100%;
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            border-radius: 2px;
            animation: loading 3s ease-in-out infinite;
        }}
        
        @keyframes loading {{
            0% {{ width: 0%; }}
            50% {{ width: 70%; }}
            100% {{ width: 100%; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="construction-icon">📈</div>
        <h1>iStockPick.ai</h1>
        <p class="subtitle">AI-powered stock analysis, sentiment, and recommendations</p>

        <div class="lookup-card">
            <div class="lookup-title">Try ticker/company lookup</div>
            <form class="lookup-form" id="lookupForm">
                <input id="lookupInput" class="lookup-input" type="text" placeholder="Enter ticker (AAPL) or company name (Apple)" required />
                <button class="lookup-button" type="submit">Check</button>
            </form>
            <div class="lookup-result" id="lookupResult"></div>
            <div class="analysis-card" id="analysisCard">
                <div id="analysisTitle" class="lookup-title">AI analysis</div>
                <div class="analysis-grid" id="analysisGrid"></div>
                <div class="analysis-meta" id="analysisMeta"></div>
            </div>
        </div>

        <div class="command-card">curl -L "https://api.istockpick.ai/SKILL.md" -o SKILL.md</div>

        <div class="status">
            <strong>System Status:</strong> Live<br>
            <strong>Server Time:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}<br>
            <strong>Mode:</strong> Production
        </div>

        <div class="footer">
            <p>Powered by 🦞 • Built with ❤️ in California</p>
        </div>
    </div>

    <script>
        const lookupForm = document.getElementById('lookupForm');
        const lookupInput = document.getElementById('lookupInput');
        const lookupResult = document.getElementById('lookupResult');
        const analysisCard = document.getElementById('analysisCard');
        const analysisTitle = document.getElementById('analysisTitle');
        const analysisGrid = document.getElementById('analysisGrid');
        const analysisMeta = document.getElementById('analysisMeta');

        lookupForm.addEventListener('submit', async (e) => {{
            e.preventDefault();
            const q = lookupInput.value.trim();
            if (!q) return;

            analysisCard.style.display = 'none';
            analysisGrid.innerHTML = '';
            analysisMeta.textContent = '';
            lookupResult.textContent = 'Checking symbol...';
            lookupResult.style.color = '#ffffff';

            try {{
                const res = await fetch(`/lookup?q=${{encodeURIComponent(q)}}`);
                const data = await res.json();

                if (!res.ok) {{
                    lookupResult.style.color = '#ffd6d6';
                    lookupResult.textContent = data.error || 'Lookup failed.';
                    return;
                }}

                lookupResult.style.color = '#d7ffd7';
                lookupResult.textContent = `✅ ${{data.input}} → ${{data.symbol}} (${{data.name}})`;

                analysisTitle.textContent = `AI analysis for ${{data.symbol}}`;
                analysisGrid.innerHTML = '<div class="analysis-item"><span class="spinner"></span>Running stock + sentiment + AI recommendation analysis...</div>';
                analysisMeta.textContent = 'Generating analysis...';
                analysisCard.style.display = 'block';

                const analysisRes = await fetch(`/analyze?q=${{encodeURIComponent(data.symbol)}}`);
                const analysis = await analysisRes.json();

                if (!analysisRes.ok) {{
                    analysisGrid.innerHTML = `<div class="analysis-item">${{analysis.error || 'Analysis failed.'}}</div>`;
                    analysisMeta.textContent = '';
                    return;
                }}

                analysisGrid.innerHTML = `
                    <div class="analysis-item"><strong>Stock analysis</strong><br>${{analysis.stock_analysis.summary}}</div>
                    <div class="analysis-item"><strong>Sentiment analysis</strong><br>${{analysis.sentiment_analysis.summary}}</div>
                    <div class="analysis-item"><strong>AI recommendation</strong><br>${{analysis.ai_recommendation.summary}}</div>
                    <div class="analysis-item"><strong>Action</strong><br>${{analysis.ai_recommendation.action}} (confidence: ${{analysis.ai_recommendation.confidence}}%)</div>
                `;

                const generatedAt = analysis.generated_at ? new Date(analysis.generated_at) : null;
                analysisMeta.textContent = generatedAt && !isNaN(generatedAt)
                    ? `Last updated: ${{generatedAt.toLocaleString()}}`
                    : '';
            }} catch (err) {{
                lookupResult.style.color = '#ffd6d6';
                lookupResult.textContent = 'Network error. Please try again.';
                analysisMeta.textContent = '';
            }}
        }});
    </script>
</body>
</html>'''
        
        self.wfile.write(html.encode())

    def serve_skill_markdown(self):
        candidates = [
            os.path.expanduser('~/projects/public/stock-analyst/SKILL.md'),
            os.path.join(os.path.dirname(__file__), 'SKILL.md'),
            os.path.join(os.path.dirname(__file__), 'stock-analyst', 'SKILL.md'),
        ]

        for skill_path in candidates:
            if os.path.isfile(skill_path):
                self.send_response(200)
                self.send_header('Content-Type', 'text/markdown; charset=utf-8')
                self.end_headers()
                with open(skill_path, 'r', encoding='utf-8') as f:
                    self.wfile.write(f.read().encode('utf-8'))
                return

        self.send_json(404, {'error': 'SKILL.md not found'})

    def serve_stock_lookup(self, raw_query):
        """Lookup ticker/company and verify it maps to a public stock."""
        query = (raw_query or '').strip()

        if not query:
            self.send_json(400, {"error": "Please provide a stock ticker or company name."})
            return
        if len(query) > self.MAX_LOOKUP_QUERY_LEN:
            self.send_json(400, {"error": "Query is too long."})
            return

        try:
            result = self.lookup_public_stock(query)
            if not result:
                self.send_json(404, {"error": f"'{query}' is not recognized as a public stock."})
                return

            self.send_json(200, {
                "input": query,
                "symbol": result["symbol"],
                "name": result["name"]
            })
        except Exception:
            self.send_json(502, {"error": "Unable to validate symbol right now. Please try again."})

    def serve_stock_analysis(self, raw_query):
        query = (raw_query or '').strip().upper()
        if not query:
            self.send_json(400, {"error": "Please provide a stock symbol to analyze."})
            return
        if len(query) > 12 or not self.STOCK_QUERY_PATTERN.fullmatch(query):
            self.send_json(400, {"error": "Please provide a valid stock ticker symbol (e.g., AAPL)."})
            return

        try:
            public_stock = self.lookup_ticker(query)
            if not public_stock:
                self.send_json(404, {"error": f"'{query}' is not recognized as a public stock."})
                return

            if generate_full_analysis is None:
                try:
                    analysis = self.generate_full_analysis_subprocess(query)
                except Exception:
                    self.send_json(500, {
                        "error": "Analysis engine is temporarily unavailable.",
                    })
                    return
            else:
                analysis = generate_full_analysis(query)

            analysis["company"] = public_stock.get("name", analysis.get("company", query))
            self.send_json(200, analysis)
        except Exception as exc:
            self.log_error("analyze failed for %s: %s", query, exc)
            self.send_json(502, {"error": "Unable to generate analysis right now. Please try again."})

    def generate_full_analysis_subprocess(self, symbol):
        script = (
            "import json, sys\n"
            f"sys.path.insert(0, {os.path.join(STOCK_ANALYST_PATH, 'stock_analyst')!r})\n"
            f"sys.path.insert(0, {os.path.join(LEGACY_STOCK_ANALYST_PATH, 'stock_analyst')!r})\n"
            "import web_analyzer\n"
            "result = web_analyzer.generate_full_analysis(sys.argv[1])\n"
            "print(json.dumps(result))\n"
        )
        runtime_python = ANALYSIS_RUNTIME_PYTHON
        if not os.path.isfile(runtime_python):
            runtime_python = sys.executable
        result = subprocess.run(
            [runtime_python, "-c", script, symbol],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(details or "analysis subprocess failed")
        stdout = (result.stdout or "").strip()
        if not stdout:
            raise RuntimeError("analysis subprocess returned empty output")
        return json.loads(stdout.splitlines()[-1])

    def lookup_public_stock(self, query):
        """Return a dict with symbol/name when query resolves to a listed equity."""
        is_ticker_like = re.fullmatch(r'[A-Za-z.\-]{1,10}', query) is not None

        # Treat as ticker only when user input looks explicitly ticker-like.
        if is_ticker_like and (query.isupper() or len(query) <= 4):
            ticker_match = self.lookup_ticker(query.upper())
            if ticker_match:
                return ticker_match

        return self.lookup_company_name(query)

    def get_sec_tickers(self):
        with ConstructionHandler.SEC_TICKERS_CACHE_LOCK:
            if ConstructionHandler.SEC_TICKERS_CACHE is not None:
                return ConstructionHandler.SEC_TICKERS_CACHE

            req = urllib.request.Request(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": "iStockPick/1.0 (admin@istockpick.ai)"}
            )

            with urllib.request.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode('utf-8'))

            tickers = []
            for item in data.values():
                ticker = (item.get('ticker') or '').strip().upper()
                title = (item.get('title') or '').strip()
                if ticker and title:
                    tickers.append({"symbol": ticker, "name": title})

            ConstructionHandler.SEC_TICKERS_CACHE = tickers
            return tickers

    def lookup_ticker(self, symbol):
        for item in self.get_sec_tickers():
            if item['symbol'] == symbol:
                return item
        return None

    def lookup_company_name(self, company_name):
        company_name_lower = company_name.lower()
        tickers = self.get_sec_tickers()

        exact = [i for i in tickers if i['name'].lower() == company_name_lower]
        if exact:
            return exact[0]

        contains = [i for i in tickers if company_name_lower in i['name'].lower()]
        if contains:
            return contains[0]

        return None

    def send_json(self, status_code, payload):
        body = json.dumps(payload).encode()
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    
    def serve_health_check(self):
        """Serve FastAPI-compatible health payload."""
        self.send_json(200, {"status": "ok"})
    
    def serve_status(self):
        """Serve detailed status information"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        status_data = {
            "server_status": "operational",
            "mode": "live",
            "domain_configured": True,
            "public_ip": "107.3.167.1",
            "nginx_proxy": "active",
            "python_backend": "running",
            "server_time": datetime.datetime.now().isoformat(),
            "endpoints": {
                "root": "/",
                "health": "/health",
                "status": "/status",
                "lookup": "/lookup?q=<ticker-or-company-name>",
                "analyze": "/analyze?q=<ticker-symbol>"
            }
        }
        
        self.wfile.write(json.dumps(status_data, indent=2).encode())
    
    def serve_404(self):
        """Serve 404 page"""
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>404 - Page Not Found</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        h1 { color: #ff6b6b; }
    </style>
</head>
<body>
    <h1>404 - Page Not Found</h1>
    <p>The page you're looking for doesn't exist yet.</p>
    <p><a href="/">← Back to Home</a></p>
</body>
</html>'''
        
        self.wfile.write(html.encode())

def main():
    """Main function to run the server"""
    PORT = 8001
    Handler = ConstructionHandler
    
    print(f"🚀 Starting In Construction Web Server on port {PORT}")
    print(f"🌐 Server accessible at: http://0.0.0.0:{PORT}/")
    print(f"🏗️  In Construction page: http://0.0.0.0:{PORT}/")
    print(f"💚 Health check: http://0.0.0.0:{PORT}/health")
    print(f"📊 Status: http://0.0.0.0:{PORT}/status")
    print("="*60)
    
    try:
        # Bind to all interfaces (0.0.0.0) instead of localhost
        class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            daemon_threads = True
            allow_reuse_address = True

        with ThreadingHTTPServer(("0.0.0.0", PORT), Handler) as httpd:
            print("✅ Server started successfully on all interfaces!")
            print("External access available at: http://107.3.167.1:8080")
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except OSError as e:
        print(f"❌ Port {PORT} is already in use. Try a different port.")
        print(f"Error: {e}")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()
