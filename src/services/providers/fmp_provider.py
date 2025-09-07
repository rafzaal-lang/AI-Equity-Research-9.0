# src/services/providers/fmp_provider.py
"""
Financial Modeling Prep (FMP) provider helpers.

Env:
  FMP_API_KEY=...   (required)
"""

import os
from typing import Any, Dict, List, Optional

# ------------------------- Core HTTP helpers -------------------------------

BASE_FMP = "https://financialmodelingprep.com/api/v3"
FMP_KEY = os.getenv("FMP_API_KEY")


def _require_key() -> None:
    if not FMP_KEY:
        raise RuntimeError("FMP_API_KEY not set")


def _get(path: str, **params) -> Any:
    """
    Synchronous GET to FMP. Returns parsed JSON (dict or list).
      _get("profile/AAPL")
      _get("historical-price-full/AAPL", serietype="line", **{"from":"2023-01-01","to":"2024-12-31"})
    """
    import requests

    _require_key()
    params = dict(params or {})
    params["apikey"] = FMP_KEY
    url = f"{BASE_FMP}/{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


async def _aget(path: str, **params) -> Any:
    """
    Async GET to FMP (httpx). Use this only from async code.
    """
    import httpx

    _require_key()
    params = dict(params or {})
    params["apikey"] = FMP_KEY
    url = f"{BASE_FMP}/{path}"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()


# ------------------------- Convenience wrappers ---------------------------

def historical_prices(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: Optional[int] = None,
    serietype: str = "line",
) -> List[Dict[str, Any]]:
    """
    Returns a list of {date, close} sorted ascending by date.
    """
    params: Dict[str, Any] = {"serietype": serietype}
    if start:
        params["from"] = start
    if end:
        params["to"] = end
    if limit:
        # FMP accepts 'timeseries' as a count (last N points)
        params["timeseries"] = limit

    data = _get(f"historical-price-full/{symbol}", **params) or {}
    hist = data.get("historical") or []
    rows = [{"date": h.get("date"), "close": h.get("close")} for h in hist if h.get("close") is not None]
    rows.sort(key=lambda r: r.get("date") or "")
    return rows


def profile(symbol: str) -> Dict[str, Any]:
    """
    Company profile. Returns {} if missing.
    """
    arr = _get(f"profile/{symbol}") or []
    return (arr or [{}])[0]


def earnings_surprises(symbol: str) -> List[Dict[str, Any]]:
    return _get(f"earnings-surprises/{symbol}") or []


def transcripts(symbol: str) -> List[Dict[str, Any]]:
    # Each item typically has {date, content, ...}
    return _get(f"earning_call_transcript/{symbol}") or []


def key_metrics(symbol: str, period: str = "annual", limit: int = 40) -> List[Dict[str, Any]]:
    return _get(f"key-metrics/{symbol}", period=period, limit=limit) or []


def key_metrics_ttm(symbol: str) -> List[Dict[str, Any]]:
    data = _get(f"key-metrics-ttm/{symbol}") or []
    return data if isinstance(data, list) else [data]


def financial_ratios(symbol: str, period: str = "annual", limit: int = 40) -> List[Dict[str, Any]]:
    return _get(f"ratios/{symbol}", period=period, limit=limit) or []


def enterprise_values(symbol: str, period: str = "annual", limit: int = 40) -> List[Dict[str, Any]]:
    data = _get(f"enterprise-values/{symbol}", period=period, limit=limit) or {}
    return data.get("enterpriseValues") or []


def income_statement(symbol: str, period: str = "annual", limit: int = 40) -> List[Dict[str, Any]]:
    return _get(f"income-statement/{symbol}", period=period, limit=limit) or []


def balance_sheet(symbol: str, period: str = "annual", limit: int = 40) -> List[Dict[str, Any]]:
    return _get(f"balance-sheet-statement/{symbol}", period=period, limit=limit) or []


def cash_flow(symbol: str, period: str = "annual", limit: int = 40) -> List[Dict[str, Any]]:
    return _get(f"cash-flow-statement/{symbol}", period=period, limit=limit) or []


def screener_by_industry(industry: str, exchange: str = "NASDAQ,NYSE", limit: int = 200) -> List[str]:
    """
    Returns list of peer symbols for a given industry.
    """
    arr = _get("stock-screener", industry=industry, exchange=exchange, limit=limit) or []
    return [x.get("symbol") for x in arr if x.get("symbol")]

# --- Peer universe helper (robust signature) --------------------------------
from typing import Optional

def peers_by_screener(symbol: str, *args, **kwargs) -> list[str]:
    """
    Build a peer list using FMP's stock-screener by industry.
    Flexible signature to match various older call sites:

        peers_by_screener("AAPL")                          -> default industry via profile, ~25 names
        peers_by_screener("AAPL", 30)                      -> ~30 names
        peers_by_screener("AAPL", "Consumer Electronics")  -> override industry
        peers_by_screener("AAPL", "Consumer Electronics", 50)
        peers_by_screener("AAPL", max_peers=40, exchange="NASDAQ,NYSE")

    Returns a list of tickers (uppercase), excluding the input symbol, de-duped.
    """
    # Defaults
    max_peers: int = kwargs.get("max_peers", kwargs.get("size", 25))
    exchange: str = kwargs.get("exchange", "NASDAQ,NYSE")
    industry: Optional[str] = kwargs.get("industry")

    # Positional arg parsing for backward compatibility
    if args:
        # 1st arg could be industry (str) OR max_peers (int)
        if isinstance(args[0], str) and not industry:
            industry = args[0]
        elif isinstance(args[0], int):
            max_peers = args[0]
        # 2nd arg could be max_peers (int) OR exchange (str)
        if len(args) > 1:
            if isinstance(args[1], int):
                max_peers = args[1]
            elif isinstance(args[1], str):
                exchange = args[1]

    sym = (symbol or "").upper().strip()

    # Determine industry if not provided
    if not industry:
        try:
            prof = profile(sym)  # uses FMP /profile/{symbol}
        except Exception:
            prof = {}
        industry = (prof.get("industry") or prof.get("sector") or "").strip()

    if not industry:
        # No way to screen peers without an industry
        return []

    # Pull candidates via stock screener
    try:
        candidates = screener_by_industry(industry=industry, exchange=exchange, limit=200)
    except Exception:
        candidates = []

    # De-dup, uppercase, remove self, cap to max_peers
    out: list[str] = []
    for t in candidates or []:
        u = (t or "").upper().strip()
        if not u or u == sym:
            continue
        if u not in out:
            out.append(u)
        if len(out) >= max_peers:
            break
    return out
