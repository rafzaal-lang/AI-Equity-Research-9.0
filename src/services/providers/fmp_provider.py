# src/services/providers/fmp_provider.py
"""
Financial Modeling Prep (FMP) provider helpers.

Env:
  FMP_API_KEY=...   (required)

This module exposes a stable API used throughout the app:
- historical_prices(symbol, start=None, end=None, limit=None, serietype="line") -> list[{date, close}]
- latest_price(symbol) -> float | None         (alias: price)
- profile(symbol) -> dict | None
- income_statement(symbol, period="annual", limit=40) -> list[dict]
- balance_sheet(symbol, period="annual", limit=40) -> list[dict]   (alias: balance_sheet_statement)
- cash_flow(symbol, period="annual", limit=40) -> list[dict]       (alias: cash_flow_statement)
- key_metrics_ttm(symbol) -> list[dict]
- enterprise_values(symbol, period="quarter", limit=16) -> list[dict]
- peers_by_screener(sector, industry, limit=20, exchange="NASDAQ,NYSE") -> list[str]

Additional helpers (not strictly required by all callers but handy):
- financial_ratios(symbol, period="annual", limit=40) -> list[dict]
- quote(symbol) -> dict | None
- market_cap(symbol) -> float | None
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

import requests

# ------------------------- Core HTTP helpers -------------------------------

BASE_FMP = "https://financialmodelingprep.com/api/v3"
FMP_KEY = os.getenv("FMP_API_KEY")


def _require_key() -> None:
    if not FMP_KEY:
        raise RuntimeError("FMP_API_KEY not set")


def _url(path: str) -> str:
    return f"{BASE_FMP}/{path}"


def _get(path: str, **params) -> Any:
    """
    Synchronous GET to FMP. Returns parsed JSON (dict or list).
      _get("profile/AAPL")
      _get("historical-price-full/AAPL", serietype="line", **{"from":"2023-01-01","to":"2024-12-31"})
    """
    _require_key()
    params = dict(params or {})
    params["apikey"] = FMP_KEY
    url = _url(path)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _first_dict(obj: Any) -> Optional[Dict[str, Any]]:
    """Return the first dict if obj is a list; if dict return it; else None."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                return x
        return None
    return None


def _ensure_list_of_dicts(obj: Any) -> List[Dict[str, Any]]:
    """Normalize response into a list[dict]."""
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []

# ------------------------- Quotes & Prices ---------------------------------


def quote(symbol: str) -> Optional[Dict[str, Any]]:
    return _first_dict(_get(f"quote/{symbol.upper()}"))


def latest_price(symbol: str) -> Optional[float]:
    """Try quote-short first; fall back to full quote."""
    head = _first_dict(_get(f"quote-short/{symbol.upper()}"))
    if head and isinstance(head, dict) and "price" in head:
        try:
            return float(head["price"])
        except Exception:
            pass
    q = quote(symbol)
    if q is not None:
        p = q.get("price") or q.get("previousClose")
        try:
            return float(p) if p is not None else None
        except Exception:
            return None
    return None


# Alias expected in some codepaths
price = latest_price


def market_cap(symbol: str) -> Optional[float]:
    q = quote(symbol)
    if not q:
        return None
    mc = q.get("marketCap")
    try:
        return float(mc) if mc is not None else None
    except Exception:
        return None

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
    hist = []
    if isinstance(data, dict):
        hist = data.get("historical") or []
    elif isinstance(data, list):
        hist = data
    rows = [{"date": h.get("date"), "close": h.get("close")} for h in hist if isinstance(h, dict) and h.get("close") is not None]
    rows.sort(key=lambda r: r.get("date") or "")
    return rows

# ------------------------- Fundamentals -----------------------------------


def profile(symbol: str) -> Optional[Dict[str, Any]]:
    return _first_dict(_get(f"profile/{symbol.upper()}"))


def key_metrics_ttm(symbol: str) -> List[Dict[str, Any]]:
    """
    FMP: /key-metrics-ttm/{symbol} returns a list.
    """
    data = _get(f"key-metrics-ttm/{symbol.upper()}") or []
    return _ensure_list_of_dicts(data)


def financial_ratios(symbol: str, period: str = "annual", limit: int = 40) -> List[Dict[str, Any]]:
    return _ensure_list_of_dicts(_get(f"ratios/{symbol.upper()}", period=period, limit=limit) or [])


def enterprise_values(symbol: str, period: str = "quarter", limit: int = 16) -> List[Dict[str, Any]]:
    """
    FMP sometimes returns:
      {"symbol":"AAPL","enterpriseValues":[...]}
    and sometimes returns a list directly. Normalize to list[dict].
    """
    data = _get(f"enterprise-values/{symbol.upper()}", period=period, limit=limit) or []
    if isinstance(data, dict):
        evs = data.get("enterpriseValues")
        return _ensure_list_of_dicts(evs)
    return _ensure_list_of_dicts(data)


def income_statement(symbol: str, period: str = "annual", limit: int = 40) -> List[Dict[str, Any]]:
    return _ensure_list_of_dicts(_get(f"income-statement/{symbol.upper()}", period=period, limit=limit) or [])


def balance_sheet(symbol: str, period: str = "annual", limit: int = 40) -> List[Dict[str, Any]]:
    return _ensure_list_of_dicts(_get(f"balance-sheet-statement/{symbol.upper()}", period=period, limit=limit) or [])


def cash_flow(symbol: str, period: str = "annual", limit: int = 40) -> List[Dict[str, Any]]:
    return _ensure_list_of_dicts(_get(f"cash-flow-statement/{symbol.upper()}", period=period, limit=limit) or [])


# Aliases some parts of the code may expect
balance_sheet_statement = balance_sheet
cash_flow_statement = cash_flow

# ------------------------- Peers / Screener -------------------------------


def peers_by_screener(
    sector: str,
    industry: str,
    limit: int = 20,
    exchange: str = "NASDAQ,NYSE"
) -> List[str]:
    """
    Build a peer list using FMP's stock-screener by sector+industry.
    Returns a list of uppercase tickers, deduped.
    """
    if not sector and not industry:
        return []

    params = {
        "limit": max(1, int(limit)),
        "exchange": exchange,
    }
    if sector:
        params["sector"] = sector
    if industry:
        params["industry"] = industry

    data = _get("stock-screener", **params) or []
    tickers: List[str] = []
    for row in data if isinstance(data, list) else []:
        sym = (row.get("symbol") or "").upper().strip() if isinstance(row, dict) else ""
        if sym and sym not in tickers:
            tickers.append(sym)
        if len(tickers) >= limit:
            break
    return tickers
