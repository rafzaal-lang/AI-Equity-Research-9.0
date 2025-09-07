# src/services/providers/fmp_provider.py
from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import httpx

log = logging.getLogger(__name__)

FMP_API_KEY = os.getenv("FMP_API_KEY") or os.getenv("FMP_KEY") or os.getenv("FINANCIAL_MODELING_PREP_KEY")
BASE_URL = "https://financialmodelingprep.com/api/v3"


def _apikey() -> str:
    if not FMP_API_KEY:
        log.warning("FMP_API_KEY not set; requests will likely fail.")
    return FMP_API_KEY or ""


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], List[Any], None]:
    """Small HTTP helper with sane defaults."""
    p = {"apikey": _apikey()}
    if params:
        p.update(params)
    url = f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    try:
        with httpx.Client(timeout=20.0) as client:
            r = client.get(url, params=p)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        log.error("FMP GET failed: %s %s :: %s", url, p, e)
        return None


def _first_row(obj: Any) -> Dict[str, Any]:
    return obj[0] if isinstance(obj, list) and obj else (obj if isinstance(obj, dict) else {})


# -------- Core endpoints we rely on -------------------------------------------------------------

@lru_cache(maxsize=256)
def key_metrics_ttm(symbol: str) -> List[Dict[str, Any]]:
    """https://financialmodelingprep.com/developer/docs#key-metrics-ttm"""
    data = _get(f"key-metrics-ttm/{symbol.upper()}")
    return data if isinstance(data, list) else []

@lru_cache(maxsize=256)
def income_statement(symbol: str, period: str = "annual", limit: int = 4) -> List[Dict[str, Any]]:
    """https://financialmodelingprep.com/developer/docs#income-statement"""
    data = _get(f"income-statement/{symbol.upper()}", {"period": period, "limit": limit})
    return data if isinstance(data, list) else []

@lru_cache(maxsize=256)
def cash_flow_statement(symbol: str, period: str = "annual", limit: int = 4) -> List[Dict[str, Any]]:
    """https://financialmodelingprep.com/developer/docs#cash-flow-statement"""
    data = _get(f"cash-flow-statement/{symbol.upper()}", {"period": period, "limit": limit})
    return data if isinstance(data, list) else []

@lru_cache(maxsize=256)
def market_cap(symbol: str) -> Optional[float]:
    """https://financialmodelingprep.com/developer/docs#market-capitalization"""
    data = _get(f"market-capitalization/{symbol.upper()}", {"limit": 1})
    row = _first_row(data)
    mc = row.get("marketCap")
    return float(mc) if isinstance(mc, (int, float)) else None

@lru_cache(maxsize=256)
def enterprise_values(symbol: str, period: str = "quarter", limit: int = 1) -> Dict[str, Any]:
    """https://financialmodelingprep.com/developer/docs#enterprise-values"""
    data = _get(f"enterprise-values/{symbol.upper()}", {"period": period, "limit": limit})
    return _first_row(data)

# -------- Useful extras used elsewhere in your project ------------------------------------------

@lru_cache(maxsize=256)
def latest_price(symbol: str) -> Optional[float]:
    """https://financialmodelingprep.com/developer/docs#quote"""
    data = _get(f"quote/{symbol.upper()}", {"limit": 1})
    row = _first_row(data)
    px = row.get("price")
    return float(px) if isinstance(px, (int, float)) else None

@lru_cache(maxsize=256)
def quote(symbol: str) -> Dict[str, Any]:
    data = _get(f"quote/{symbol.upper()}", {"limit": 1})
    return _first_row(data)

@lru_cache(maxsize=256)
def profile(symbol: str) -> Dict[str, Any]:
    data = _get(f"profile/{symbol.upper()}")
    return _first_row(data)
