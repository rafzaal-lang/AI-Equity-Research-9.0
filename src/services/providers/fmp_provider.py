"""
Financial Modeling Prep (FMP) provider — drop-in module.

Adds robust wrappers for common endpoints and normalizes list responses
so callers can safely do .get(...) without "list has no attribute 'get'".

Env:
    FMP_API_KEY  (required)

Provided functions (with common aliases):
- latest_price(ticker) -> float | None        (alias: price)
- quote(ticker) -> dict | None
- profile(ticker) -> dict | None
- enterprise_values(ticker) -> dict | None    (alias: get_enterprise_values)
- market_cap(ticker) -> float | None
- key_metrics(ticker) -> dict | None

Financial statements (lists newest-first from FMP):
- income_statement(ticker, period="annual", limit=120) -> list[dict]
  (aliases: get_income_statement, income_stmt)
- balance_sheet_statement(ticker, period="annual", limit=120) -> list[dict]
  (aliases: get_balance_sheet_statement, balance_sheet)
- cash_flow_statement(ticker, period="annual", limit=120) -> list[dict]
  (aliases: get_cash_flow_statement, cashflow_statement, cash_flow, cashflow)

Bundle:
- financials_annual(ticker) -> dict with income_statement/balance_sheet/cashflow_statement
- ev_and_marketcap(ticker) -> {"enterpriseValue": float|None, "marketCap": float|None}
"""
from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

FMP_API_KEY = os.getenv("FMP_API_KEY")
BASE_URL = "https://financialmodelingprep.com/api/v3"


class FMPError(RuntimeError):
    pass


def _require_key():
    if not FMP_API_KEY:
        raise FMPError("FMP_API_KEY is not set in environment.")


def _url(path: str, **params) -> str:
    q = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
    if q:
        return f"{BASE_URL}{path}?{q}&apikey={FMP_API_KEY}"
    return f"{BASE_URL}{path}?apikey={FMP_API_KEY}"


def _get(path: str, **params) -> Any:
    """GET helper that returns parsed JSON or None."""
    _require_key()
    url = _url(path, **params)
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning("FMP GET failed for %s: %s", url, e)
        return None


def _head_or_none(obj: Any) -> Optional[Dict[str, Any]]:
    """Normalize FMP list responses to the first dict element, else None."""
    if obj is None:
        return None
    if isinstance(obj, list):
        return obj[0] if obj else None
    if isinstance(obj, dict):
        return obj
    return None


# --------------------------- Quotes & Company ---------------------------

@lru_cache(maxsize=1024)
def quote(ticker: str) -> Optional[Dict[str, Any]]:
    return _head_or_none(_get(f"/quote/{ticker.upper()}"))


@lru_cache(maxsize=4096)
def latest_price(ticker: str) -> Optional[float]:
    # Fast path
    head = _head_or_none(_get(f"/quote-short/{ticker.upper()}"))
    if head and isinstance(head, dict) and "price" in head:
        try:
            return float(head["price"])
        except Exception:
            pass
    # Fallback
    q = quote(ticker)
    if q:
        p = q.get("price") or q.get("previousClose")
        try:
            return float(p) if p is not None else None
        except Exception:
            return None
    return None


# Alias commonly used by existing code
price = latest_price


@lru_cache(maxsize=1024)
def profile(ticker: str) -> Optional[Dict[str, Any]]:
    return _head_or_none(_get(f"/profile/{ticker.upper()}"))


@lru_cache(maxsize=2048)
def enterprise_values(ticker: str) -> Optional[Dict[str, Any]]:
    """Return the most recent enterprise value row as a dict."""
    return _head_or_none(_get(f"/enterprise-values/{ticker.upper()}"))


# Backward-compat alias
get_enterprise_values = enterprise_values


@lru_cache(maxsize=1024)
def market_cap(ticker: str) -> Optional[float]:
    q = quote(ticker)
    if q:
        mc = q.get("marketCap")
        try:
            return float(mc) if mc is not None else None
        except Exception:
            return None
    p = profile(ticker)
    if p:
        mc = p.get("mktCap") or p.get("marketCap")
        try:
            return float(mc) if mc is not None else None
        except Exception:
            return None
    return None


@lru_cache(maxsize=1024)
def key_metrics(ticker: str) -> Optional[Dict[str, Any]]:
    return _head_or_none(_get(f"/key-metrics/{ticker.upper()}"))


# --------------------------- Financials ---------------------------

def _normalize_list(obj: Any) -> List[Dict[str, Any]]:
    """Ensure a list[dict] (or empty list) for statement endpoints."""
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


@lru_cache(maxsize=1024)
def income_statement(ticker: str, period: str = "annual", limit: int = 120) -> List[Dict[str, Any]]:
    return _normalize_list(_get(f"/income-statement/{ticker.upper()}", period=period, limit=limit))


# common aliases some codebases use
get_income_statement = income_stmt = income_statement


@lru_cache(maxsize=1024)
def balance_sheet_statement(ticker: str, period: str = "annual", limit: int = 120) -> List[Dict[str, Any]]:
    return _normalize_list(_get(f"/balance-sheet-statement/{ticker.upper()}", period=period, limit=limit))


get_balance_sheet_statement = balance_sheet = balance_sheet_statement


@lru_cache(maxsize=1024)
def cash_flow_statement(ticker: str, period: str = "annual", limit: int = 120) -> List[Dict[str, Any]]:
    return _normalize_list(_get(f"/cash-flow-statement/{ticker.upper()}", period=period, limit=limit))


# more aliases for safety
get_cash_flow_statement = cashflow_statement = cash_flow = cashflow = cash_flow_statement


@lru_cache(maxsize=256)
def financials_annual(ticker: str) -> Dict[str, List[Dict[str, Any]]]:
    """Convenience bundle of annual statements."""
    return {
        "income_statement": income_statement(ticker, period="annual"),
        "balance_sheet": balance_sheet_statement(ticker, period="annual"),
        "cashflow_statement": cash_flow_statement(ticker, period="annual"),
    }


# --------------------------- Convenience ---------------------------

@lru_cache(maxsize=1024)
def ev_and_marketcap(ticker: str) -> Dict[str, Optional[float]]:
    ev_row = enterprise_values(ticker) or {}
    ev = ev_row.get("enterpriseValue") if isinstance(ev_row, dict) else None
    try:
        ev_val = float(ev) if ev is not None else None
    except Exception:
        ev_val = None
    mc_val = market_cap(ticker)
    return {"enterpriseValue": ev_val, "marketCap": mc_val}
