---
name: Frontend Backend Refactor
overview: Reorganize the codebase into top-level `frontend/` and `backend/` directories, update the server to serve the static frontend from disk instead of embedded HTML, add a requirements.txt, and update all docs and CI config for the new layout.
todos:
  - id: move-files
    content: Create frontend/ and backend/ dirs; move docs/index.html -> frontend/, construction_server_fixed.py -> backend/server.py, stock-analyst/* contents -> backend/
    status: completed
  - id: refactor-server
    content: "Update backend/server.py: fix path constants, replace embedded HTML with file read from frontend/, update data paths"
    status: completed
  - id: add-requirements
    content: Create backend/requirements.txt with project dependencies
    status: completed
  - id: update-workflow
    content: Update .github/workflows/jekyll-gh-pages.yml to deploy from ./frontend
    status: completed
  - id: update-gitignore
    content: Update .gitignore for new paths (backend/data/weights.txt)
    status: completed
  - id: update-readme
    content: Rewrite README.md sections for new frontend/backend layout and local dev instructions
    status: completed
  - id: update-skill
    content: Update SKILL.md paths from stock-analyst/ to backend/ and construction_server_fixed.py to backend/server.py
    status: completed
  - id: cleanup
    content: "Delete old files/dirs: docs/, stock-analyst/, construction_server.py, construction_server_fixed.py"
    status: completed
isProject: false
---

# Frontend / Backend Refactor

## Current vs Target Layout

```
BEFORE                                   AFTER
─────────────────────                    ─────────────────────
construction_server.py      (legacy)     frontend/
construction_server_fixed.py             ├── index.html
docs/                                    backend/
├── index.html                           ├── server.py
stock-analyst/                           ├── requirements.txt
├── stock_analyst/                       ├── stock_analyst/
│   ├── __init__.py                      │   ├── __init__.py
│   ├── api.py                           │   ├── api.py
│   ├── config.py                        │   ├── config.py
│   ├── fundamental.py                   │   ├── fundamental.py
│   ├── technical.py                     │   ├── technical.py
│   └── web_analyzer.py                  │   └── web_analyzer.py
├── scripts/                             ├── scripts/
├── samples/                             ├── samples/
└── data/                                └── data/
```

## File Moves

- `docs/index.html` -> `frontend/index.html`
- `construction_server_fixed.py` -> `backend/server.py`
- `stock-analyst/stock_analyst/` -> `backend/stock_analyst/`
- `stock-analyst/scripts/` -> `backend/scripts/`
- `stock-analyst/samples/` -> `backend/samples/`
- `stock-analyst/data/` -> `backend/data/`
- Delete `docs/`, `stock-analyst/`, `construction_server.py`, `construction_server_fixed.py` after moves

## Key Code Changes

### 1. `backend/server.py` (was `construction_server_fixed.py`)

Path constants at the top (lines 23-38) change because `stock_analyst/` is now a sibling, not nested under `stock-analyst/`:

```python
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_FRONTEND_DIR = os.path.realpath(os.path.join(_BACKEND_DIR, "..", "frontend"))

DEFAULT_STOCK_ANALYST_PATH = os.path.realpath(
    os.getenv("DEFAULT_STOCK_ANALYST_PATH", _BACKEND_DIR)
)
STOCK_ANALYST_PATH = os.path.realpath(
    os.environ.get("STOCK_ANALYST_PATH", DEFAULT_STOCK_ANALYST_PATH)
)
LEGACY_STOCK_ANALYST_PATH = _BACKEND_DIR
```

`_weights_db_path()` and `_portfolio_db_path()` (lines 256-259) change from `os.path.join(os.path.dirname(__file__), "stock-analyst", "data", ...)` to `os.path.join(_BACKEND_DIR, "data", ...)`.

`serve_construction_page()` (lines 1144-1778) -- replace the ~630-line embedded HTML f-string with reading `frontend/index.html` from disk:

```python
def serve_construction_page(self):
    index_path = os.path.join(_FRONTEND_DIR, "index.html")
    if not os.path.isfile(index_path):
        self.send_json(404, {"error": "Frontend index.html not found"})
        return
    self.send_response(200)
    self.send_header('Content-type', 'text/html')
    self.end_headers()
    with open(index_path, 'r', encoding='utf-8') as f:
        self.wfile.write(f.read().encode())
```

### 2. `backend/requirements.txt` (new)

```
fastapi
uvicorn
yfinance
pydantic
python-dotenv
pandas
numpy
```

### 3. `.github/workflows/jekyll-gh-pages.yml`

Change `path: ./docs` to `path: ./frontend`.

### 4. `.gitignore`

Update `stock-analyst/data/weights.txt` to `backend/data/weights.txt`.

### 5. `README.md`

Rewrite the following sections for the new layout:

- **Project Structure** -- reflect `frontend/` and `backend/`
- **Architecture** -- update diagram to reference `backend/server.py`
- **Getting Started** -- new commands (`cd backend`, `pip install -r requirements.txt`, `python server.py`)
- **Deployment** -- update compile-check and run paths

### 6. `SKILL.md`

Update all `stock-analyst/` references to `backend/` and `construction_server_fixed.py` to `backend/server.py`.

## Local Development (single command)

After the refactor, local development is:

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python server.py
# Serves frontend at http://localhost:8001/ and API at the same origin
```

The server reads `frontend/index.html` from disk, so editing the HTML takes effect on the next page refresh -- no rebuild needed.