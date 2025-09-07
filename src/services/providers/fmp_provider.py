"""
Financial Modeling Prep (FMP) provider — drop-in module.

This module wraps commonly used FMP endpoints and normalizes responses so
callers can safely access dictionaries (e.g., .get(...)) without hitting
"list has no attribute 'get'" errors.

Environment:
    FMP_API_KEY: required

Public functions:
    latest_price(ticker) -> float | None
    price(ticker) -> float | None                # alias
    quote(ticker) -> dict | None
    profile(ticker) -> dict | None
    enterprise_values(ticker) -> dict | None
    get_enterprise_values(ticker) -> dict | None # alias
    market_cap(ticker) -> float | None
    key_metrics(ticker) -> dict | None
    financials_annual(ticker) -> dict[str, list[dict]]  # income/balance/cashflow
    ev_and_marketcap(ticker) -> dict[str, float | None]
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


def _url(path: str) -> str:
    return f"{BASE_URL}{path}?apikey={FMP_API_KEY}"


def _get(path: str) -> Any:
    """GET helper that returns parsed JSON or None.

    FMP typically returns a list; we don't coerce here except in public fns.
    """
    _require_key()
    url = _url(path)
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning("FMP GET failed for %s: %s", path, e)
        return None


def _head_or_none(obj: Any) -> Optional[Dict[str, Any]]:
    """Normalize FMP list responses to the first dict element, else None."""
    if obj is None:
        return None
    if isinstance(obj, list):
        return obj[0] if obj else None
    if isinstance(obj, dict):
        return obj
    # Unexpected type; return None to keep callers safe
    return None


# --------------------------- Public API ---------------------------

@lru_cache(maxsize=1024)
def quote(ticker: str) -> Optional[Dict[str, Any]]:
    """Full quote with marketCap, price, volume, etc."""
    data = _get(f"/quote/{ticker.upper()}")
    return _head_or_none(data)


@lru_cache(maxsize=4096)
def latest_price(ticker: str) -> Optional[float]:
    """Lightweight latest price; falls back to full quote if needed."""
    data = _get(f"/quote-short/{ticker.upper()}")
    head = _head_or_none(data)
    if head and isinstance(head, dict) and "price" in head:
        try:
            return float(head["price"])
        except Exception:
            pass
    # Fallback to full quote
    q = quote(ticker)
    if q and isinstance(q, dict):
        p = q.get("price") or q.get("previousClose")
        try:
            return float(p) if p is not None else None
        except Exception:
            return None
    return None


# Alias commonly used by callers
price = latest_price


@lru_cache(maxsize=1024)
def profile(ticker: str) -> Optional[Dict[str, Any]]:
    return _head_or_none(_get(f"/profile/{ticker.upper()}"))


@lru_cache(maxsize=2048)
def enterprise_values(ticker: str) -> Optional[Dict[str, Any]]:
    """Return the most recent enterprise value row as a dict.

    Many FMP endpoints return a list sorted from most recent to oldest.
    We always normalize to the first item, so callers can safely do .get(...).
    """
    return _head_or_none(_get(f"/enterprise-values/{ticker.upper()}"))


# Backward compatible alias
get_enterprise_values = enterprise_values


@lru_cache(maxsize=1024)
def market_cap(ticker: str) -> Optional[float]:
    q = quote(ticker)
    if q and isinstance(q, dict):
        mc = q.get("marketCap")
        try:
            return float(mc) if mc is not None else None
        except Exception:
            return None
    # Fallback: some profiles include mktCap
    p = profile(ticker)
    if p and isinstance(p, dict):
        mc = p.get("mktCap") or p.get("marketCap")
        try:
            return float(mc) if mc is not None else None
        except Exception:
            return None
    return None


@lru_cache(maxsize=1024)
def key_metrics(ticker: str) -> Optional[Dict[str, Any]]:
    return _head_or_none(_get(f"/key-metrics/{ticker.upper()}"))


@lru_cache(maxsize=256)
def financials_annual(ticker: str) -> Dict[str, List[Dict[str, Any]]]:
    """Return standardized annual financial statements.

    Each list is newest-first, as returned by FMP.
    """
    income = _get(f"/income-statement/{ticker.upper()}") or []
    balance = _get(f"/balance-sheet-statement/{ticker.upper()}") or []
    cash = _get(f"/cash-flow-statement/{ticker.upper()}") or []
    # Ensure lists
    income = income if isinstance(income, list) else []
    balance = balance if isinstance(balance, list) else []
    cash = cash if isinstance(cash, list) else []
    return {
        "income_statement": income,
        "balance_sheet": balance,
        "cashflow_statement": cash,
    }


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
