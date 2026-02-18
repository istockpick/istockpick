#!/usr/bin/env python3
"""
Scan S&P 500 tickers through iStockPick recommendation API and
return symbols matching a target recommendation action.

Example:
  python3 istockpick_reco_scan.py --rec sell --agent Geoffrey_US --token YOUR_TOKEN
  python3 istockpick_reco_scan.py --rec buy --agent Geoffrey_US --token YOUR_TOKEN --limit 50 --output buys.json
  python3 istockpick_reco_scan.py --rec hold --agent Geoffrey_US --token YOUR_TOKEN --list AAPL,MSFT,NVDA
  python3 istockpick_reco_scan.py --rec buy --index dow --agent Geoffrey_US --token YOUR_TOKEN
"""

import argparse
import time
import csv
import io
import json
import urllib.parse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
DEFAULT_API_URL = "https://api.istockpick.ai/api/v1/recommendation"

DOW30_SYMBOLS = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
    "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT", "AMZN",
]

NASDAQ100_SYMBOLS = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "ANSS", "APP", "ARM", "ASML", "AVGO", "AXON", "AZN", "BIIB", "BKNG",
    "CDNS", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO", "CSGP", "CSX",
    "CTAS", "CTSH", "DASH", "DDOG", "DXCM", "EA", "EXC", "FANG", "FAST", "FTNT",
    "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "INTC", "INTU", "ISRG",
    "KDP", "KHC", "KLAC", "LRCX", "LULU", "MAR", "MCHP", "MDB", "MDLZ", "MELI",
    "META", "MNST", "MRNA", "MRVL", "MSFT", "MU", "NFLX", "NVDA", "NXPI", "ODFL",
    "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP", "PYPL", "QCOM", "REGN",
    "ROP", "ROST", "SBUX", "SHOP", "SNPS", "TEAM", "TMUS", "TSLA", "TTD", "TTWO",
    "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS",
]


def fetch_sp500_symbols() -> list[str]:
    text = urllib.request.urlopen(SP500_CSV_URL, timeout=20).read().decode("utf-8", "ignore")
    rows = csv.DictReader(io.StringIO(text))
    symbols = []
    for row in rows:
        sym = (row.get("Symbol") or "").strip()
        if sym:
            symbols.append(sym.replace(".", "-"))
    return symbols


def get_symbols_for_index(index_name: str) -> list[str]:
    normalized = index_name.strip().lower()
    if normalized in {"sp", "sp500", "s&p", "s&p500"}:
        return fetch_sp500_symbols()
    if normalized in {"dow", "dow30", "djia"}:
        return DOW30_SYMBOLS.copy()
    if normalized in {"qqq", "nasdaq100", "ndx", "nasdaq-100"}:
        return NASDAQ100_SYMBOLS.copy()
    raise ValueError(f"Unsupported index: {index_name}. Use sp/sp500, dow, or qqq.")


def call_reco(symbol: str, agent_name: str, agent_token: str, timeout: int, api_url: str, method: str) -> dict:
    payload = {
        "stock": symbol,
        "agent_name": agent_name,
        "agent_token": agent_token,
    }
    method = method.upper()

    if method == "POST":
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            api_url,
            data=data,
            method="POST",
            headers={
                "User-Agent": "iStockPickRecoScan/1.0",
                "Content-Type": "application/json",
            },
        )
    else:
        params = urllib.parse.urlencode(payload)
        req = urllib.request.Request(
            f"{api_url}?{params}",
            method="GET",
            headers={"User-Agent": "iStockPickRecoScan/1.0"},
        )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "ignore")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        detail = body
        try:
            decoded = json.loads(body)
            detail = decoded.get("detail") or decoded
        except Exception:
            pass
        raise RuntimeError(f"http_{exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"network_error: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan S&P 500 recommendations from iStockPick")
    parser.add_argument(
        "--rec",
        required=False,
        default="",
        choices=["", "buy", "hold", "sell"],
        help="Recommendation to filter (optional). If omitted, returns BUY/HOLD/SELL all together.",
    )
    parser.add_argument("--agent", required=True, help="Registered agent name")
    parser.add_argument("--token", required=True, help="Registered agent token")
    parser.add_argument(
        "--index",
        default="sp500",
        help="Index universe when --list is omitted: sp/sp500, dow, qqq (default: sp500)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional number of symbols to scan (0 = all)")
    parser.add_argument(
        "--list",
        default="",
        help="Comma-separated tickers to scan instead of full S&P 500 list (e.g. AAPL,MSFT,NVDA)",
    )
    parser.add_argument("--workers", type=int, default=12, help="Parallel requests")
    parser.add_argument("--timeout", type=int, default=15, help="Per-request timeout seconds")
    parser.add_argument(
        "--method",
        default="GET",
        choices=["GET", "POST", "get", "post"],
        help="HTTP method for recommendation API (default: GET)",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Recommendation API endpoint (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N symbols (default: 50, 0 = disable)",
    )
    parser.add_argument("--output", default="", help="Optional output JSON path")
    args = parser.parse_args()

    target = args.rec.upper()
    method = args.method.upper()
    started_at = time.time()

    if args.list.strip():
        symbols = [s.strip().upper().replace('.', '-') for s in args.list.split(',') if s.strip()]
    else:
        symbols = get_symbols_for_index(args.index)
        if args.limit and args.limit > 0:
            symbols = symbols[: args.limit]

    matches = []
    errors = []
    ok = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(call_reco, sym, args.agent, args.token, args.timeout, args.api_url, method): sym
            for sym in symbols
        }
        for i, fut in enumerate(as_completed(futures), 1):
            sym = futures[fut]
            try:
                payload = fut.result()
                rec = ((payload.get("recommendation") or {}).get("action") or "").upper()
                conf = (payload.get("recommendation") or {}).get("confidence")
                ok += 1
                if (target and rec == target) or (not target and rec in {"BUY", "HOLD", "SELL"}):
                    matches.append(
                        {
                            "symbol": payload.get("resolved_symbol") or sym,
                            "company": payload.get("company") or sym,
                            "action": rec,
                            "confidence": conf,
                            "summary": (payload.get("recommendation") or {}).get("summary"),
                            "generated_at": payload.get("generated_at"),
                        }
                    )
            except Exception as exc:
                errors.append({"symbol": sym, "error": str(exc)})

            if args.progress_every > 0 and i % args.progress_every == 0:
                print(f"progress {i}/{len(symbols)} ok={ok} match={len(matches)} err={len(errors)}", flush=True)

    matches.sort(key=lambda x: (x["action"], x["confidence"] is None, -(x["confidence"] or 0), x["symbol"]))

    by_action = {"BUY": [], "HOLD": [], "SELL": []}
    for item in matches:
        by_action.setdefault(item["action"], []).append(item)

    elapsed_s = round(time.time() - started_at, 2)
    throughput = round((len(symbols) / elapsed_s), 2) if elapsed_s > 0 else None

    result = {
        "target": target or "ALL",
        "index": args.index.lower(),
        "method": method,
        "api_url": args.api_url,
        "scanned": len(symbols),
        "ok": ok,
        "errors": len(errors),
        "elapsed_seconds": elapsed_s,
        "throughput_per_sec": throughput,
        "matches": matches,
        "by_action": by_action,
        "error_details": errors,
    }

    for item in matches:
        generated_at = (item.get("generated_at") or "").strip()
        date_part = generated_at.split("T", 1)[0] if generated_at else ""
        print(f"{item['symbol']}:{item['action']}:{date_part}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved full output to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
