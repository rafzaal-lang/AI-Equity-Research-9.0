# CORRECT src/services/comps/engine.py file
# This file should ONLY contain business logic, no FastAPI routes!

from __future__ import annotations
from typing import Dict, Any, List, Optional
import statistics as stats
from src.services.providers import fmp_provider as fmp

def _num(x):
    try: return float(x) if x is not None else None
    except: return None

def peer_universe(symbol: str, max_peers: int = 15) -> List[str]:
    prof = fmp.profile(symbol) or {}
    sector, industry = prof.get("sector"), prof.get("industry")
    if not sector or not industry: return []
    peers = fmp.peers_by_screener(sector, industry, limit=60)
    peers = [p for p in peers if p.upper() != symbol.upper()]
    return peers[:max_peers]

def latest_metrics(symbol: str, as_of: Optional[str] = None) -> Dict[str, Optional[float]]:
    """Get latest financial metrics for a symbol with robust error handling."""
    try:
        # Get data with safe fallbacks
        inc_data = fmp.income_statement(symbol, period="annual", limit=1)
        km_data = fmp.key_metrics_ttm(symbol)
        evs_data = fmp.enterprise_values(symbol, period="quarter", limit=1)
        
        # Safely extract first element or use empty dict
        inc = {}
        if inc_data and isinstance(inc_data, list) and len(inc_data) > 0:
            inc = inc_data[0] if isinstance(inc_data[0], dict) else {}
        
        km = {}
        if km_data and isinstance(km_data, list) and len(km_data) > 0:
            km = km_data[0] if isinstance(km_data[0], dict) else {}
        
        evs = {}
        if evs_data and isinstance(evs_data, list) and len(evs_data) > 0:
            evs = evs_data[0] if isinstance(evs_data[0], dict) else {}
        
        # Get price with error handling
        try:
            price = fmp.latest_price(symbol)
        except Exception:
            price = None
        
        # Extract values safely
        sales = _num(inc.get("revenue")) if isinstance(inc, dict) else None
        ebitda = (_num(inc.get("ebitda")) or _num(inc.get("operatingIncome"))) if isinstance(inc, dict) else None
        eps = _num(km.get("netIncomePerShareTTM")) if isinstance(km, dict) else None
        market_cap = _num(evs.get("marketCapitalization")) if isinstance(evs, dict) else None
        ev = _num(evs.get("enterpriseValue")) if isinstance(evs, dict) else None
        shares = _num(inc.get("weightedAverageShsOut")) if isinstance(inc, dict) else None
        
        return {
            "price": price,
            "sales": sales,
            "ebitda": ebitda,
            "eps_ttm": eps,
            "market_cap": market_cap,
            "enterprise_value": ev,
            "shares_out": shares,
            "ps": (market_cap / sales) if market_cap and sales and sales != 0 else None,
            "pe": (price / eps) if price and eps and eps != 0 else None,
            "ev_ebitda": (ev / ebitda) if ev and ebitda and ebitda != 0 else None,
        }
        
    except Exception as e:
        # Log the error and return safe defaults
        print(f"Error in latest_metrics for {symbol}: {e}")
        return {
            "price": None, "sales": None, "ebitda": None, "eps_ttm": None,
            "market_cap": None, "enterprise_value": None, "shares_out": None,
            "ps": None, "pe": None, "ev_ebitda": None,
        }

def comps_table(symbol: str, max_peers: int = 15, as_of: Optional[str] = None) -> Dict[str, Any]:
    tickers = [symbol.upper()] + peer_universe(symbol, max_peers=max_peers)
    rows = []
    for t in tickers:
        metrics = latest_metrics(t, as_of=as_of)
        rows.append({"ticker": t, **metrics})
    
    fields = ["pe", "ps", "ev_ebitda", "market_cap"]
    for f in fields:
        vals = [r[f] for r in rows if r.get(f) is not None]
        if len(vals) >= 4:
            q = stats.quantiles(vals, n=4)
            p25, p50, p75 = q[0], q[1], q[2]
        else:
            p25 = p50 = p75 = None
        for r in rows:
            r[f+"_p25"] = p25; r[f+"_p50"] = p50; r[f+"_p75"] = p75
    
    return {"symbol": symbol.upper(), "peers": rows}
