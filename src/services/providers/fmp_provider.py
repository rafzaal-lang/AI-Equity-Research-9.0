import os
from typing import Any, Dict, List, Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.services.cache.redis_client import get_json, set_json

FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/api")
SESSION = requests.Session()

class FMPError(Exception): pass

@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=0.2, max=2.0),
       retry=retry_if_exception_type((requests.RequestException, FMPError)))
def _get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Any:
    if params is None: params = {}
    if not FMP_API_KEY: raise FMPError("Missing FMP_API_KEY")
    params = {**params, "apikey": FMP_API_KEY}
    url = f"{FMP_BASE_URL}/v3/{path.lstrip('/')}"
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _cached(path: str, params: Dict[str, Any], ttl: int, force_refresh: bool = False):
    key = f"fmp:{path}:{sorted(params.items())}"
    if not force_refresh:
        cached = get_json(key)
        if cached is not None:
            return cached
    data = _get(path, params)
    set_json(key, data, ttl=ttl)
    return data

def historical_prices(symbol: str, start: Optional[str] = None, end: Optional[str] = None, limit: int = 1000, force_refresh: bool=False) -> List[Dict[str, Any]]:
    data = _cached(f"historical-price-full/{symbol}", {"serietype": "line"}, ttl=3600, force_refresh=force_refresh)
    hist = data.get("historical", []) if isinstance(data, dict) else data
    if start:
        hist = [h for h in hist if h.get("date") >= start]
    if end:
        hist = [h for h in hist if h.get("date") <= end]
    return hist[:limit]

def latest_price(symbol: str) -> Optional[float]:
    q = _cached(f"quote/{symbol}", {}, ttl=300)
    try:
        if isinstance(q, list) and q:
            return float(q[0].get("price"))
    except Exception:
        return None
    return None

def profile(symbol: str) -> Dict[str, Any]:
    p = _cached(f"profile/{symbol}", {}, ttl=21600)
    return p[0] if isinstance(p, list) and p else {}

def income_statement(symbol: str, period: str = "annual", limit: int = 8) -> List[Dict[str, Any]]:
    return _cached(f"income-statement/{symbol}", {"period": period, "limit": limit}, ttl=21600)

def balance_sheet(symbol: str, period: str = "annual", limit: int = 8) -> List[Dict[str, Any]]:
    return _cached(f"balance-sheet-statement/{symbol}", {"period": period, "limit": limit}, ttl=21600)

def cash_flow(symbol: str, period: str = "annual", limit: int = 8) -> List[Dict[str, Any]]:
    return _cached(f"cash-flow-statement/{symbol}", {"period": period, "limit": limit}, ttl=21600)

def key_metrics_ttm(symbol: str) -> List[Dict[str, Any]]:
    return _cached(f"key-metrics-ttm/{symbol}", {}, ttl=7200)

def enterprise_values(symbol: str, period: str = "quarter", limit: int = 16) -> List[Dict[str, Any]]:
    return _cached(f"enterprise-values/{symbol}", {"period": period, "limit": limit}, ttl=3600)

def peers_by_screener(sector: str, industry: str, limit: int = 20) -> List[str]:
    res = _cached("stock-screener", {"sector": sector, "industry": industry, "limit": limit}, ttl=21600)
    tickers = [r.get("symbol") for r in res if r.get("symbol")]
    return [t for t in tickers if t]

