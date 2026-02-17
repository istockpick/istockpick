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
import urllib.error
import urllib.request
import urllib.parse
from urllib.parse import urlparse, parse_qs

DEFAULT_STOCK_ANALYST_PATH = "/Users/richliu/projects/public/stock-analyst"
STOCK_ANALYST_PATH = os.path.realpath(
    os.environ.get("STOCK_ANALYST_PATH", DEFAULT_STOCK_ANALYST_PATH)
)
LEGACY_STOCK_ANALYST_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "stock-analyst")
)
FASTAPI_UPSTREAM = os.getenv("FASTAPI_UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
ANALYSIS_RUNTIME_PYTHON = os.getenv(
    "ANALYSIS_RUNTIME_PYTHON",
    "/tmp/stock-analyst-venv/bin/python3",
)

for p in (STOCK_ANALYST_PATH, LEGACY_STOCK_ANALYST_PATH):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from stock_analyst.web_analyzer import generate_full_analysis
    ANALYSIS_IMPORT_ERROR = None
except Exception as exc:
    generate_full_analysis = None
    ANALYSIS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


class ConstructionHandler(http.server.SimpleHTTPRequestHandler):
    SEC_TICKERS_CACHE = None
    FASTAPI_PROXY_PATH_PREFIXES = ("/api/", "/docs", "/openapi.json")
    FASTAPI_PROXY_EXACT_PATHS = {"/health"}

    def should_proxy_to_fastapi(self, path):
        if path in self.FASTAPI_PROXY_EXACT_PATHS:
            return True
        return any(path.startswith(prefix) for prefix in self.FASTAPI_PROXY_PATH_PREFIXES)

    def proxy_to_fastapi(self):
        target_url = f"{FASTAPI_UPSTREAM}{self.path}"
        payload = None
        if self.command in {"POST", "PUT", "PATCH"}:
            content_length = int(self.headers.get("Content-Length", "0") or 0)
            payload = self.rfile.read(content_length) if content_length > 0 else b""

        headers = {}
        for name, value in self.headers.items():
            if name.lower() in {"host", "connection", "content-length", "accept-encoding"}:
                continue
            headers[name] = value

        headers["X-Forwarded-For"] = self.client_address[0]
        headers["X-Forwarded-Proto"] = "https"
        headers["X-Forwarded-Host"] = self.headers.get("Host", "api.istockpick.ai")

        request = urllib.request.Request(target_url, data=payload, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                response_body = response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
        except urllib.error.HTTPError as err:
            error_body = err.read()
            self.send_response(err.code)
            self.send_header("Content-Type", err.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(error_body)))
            self.end_headers()
            self.wfile.write(error_body)
        except Exception:
            self.send_json(502, {"error": "FastAPI upstream is unavailable."})

    def do_GET(self):
        parsed_path = urlparse(self.path)
        query = parse_qs(parsed_path.query)

        if self.should_proxy_to_fastapi(parsed_path.path):
            self.proxy_to_fastapi()
        elif parsed_path.path == '/':
            self.serve_construction_page()
        elif parsed_path.path == '/health':
            self.serve_health_check()
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

        if self.should_proxy_to_fastapi(parsed_path.path):
            self.proxy_to_fastapi()
            return

        self.send_json(404, {"error": "Unknown endpoint."})
    
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

        try:
            public_stock = self.lookup_ticker(query)
            if not public_stock:
                self.send_json(404, {"error": f"'{query}' is not recognized as a public stock."})
                return

            if generate_full_analysis is None:
                try:
                    analysis = self.generate_full_analysis_subprocess(query)
                except Exception as exc:
                    self.send_json(500, {
                        "error": "stock_analyst analysis functions are unavailable.",
                        "import_error": ANALYSIS_IMPORT_ERROR,
                        "fallback_error": f"{type(exc).__name__}: {exc}",
                        "search_paths": [STOCK_ANALYST_PATH, LEGACY_STOCK_ANALYST_PATH],
                        "analysis_runtime_python": ANALYSIS_RUNTIME_PYTHON,
                    })
                    return
            else:
                analysis = generate_full_analysis(query)

            analysis["company"] = public_stock.get("name", analysis.get("company", query))
            self.send_json(200, analysis)
        except Exception:
            self.send_json(502, {"error": "Unable to generate analysis right now. Please try again."})

    def generate_full_analysis_subprocess(self, symbol):
        script = (
            "import json, sys\n"
            f"sys.path.insert(0, {STOCK_ANALYST_PATH!r})\n"
            "from stock_analyst.web_analyzer import generate_full_analysis\n"
            "result = generate_full_analysis(sys.argv[1])\n"
            "print(json.dumps(result))\n"
        )
        result = subprocess.run(
            [ANALYSIS_RUNTIME_PYTHON, "-c", script, symbol],
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
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())
    
    def serve_health_check(self):
        """Serve health check endpoint"""
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "server": "OpenClaw Web Server",
            "version": "1.0.0",
            "port": 8001,
            "mode": "live"
        }
        
        self.wfile.write(json.dumps(health_data, indent=2).encode())
    
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
        with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
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
