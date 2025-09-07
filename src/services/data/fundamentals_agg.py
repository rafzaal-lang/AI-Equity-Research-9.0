# src/services/data/fundamentals_agg.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import math
import logging

from src.services.providers import fmp_provider as fmp

log = logging.getLogger(__name__)

# ---- small utils -----------------------------------------------------------

def _first(lst: Any) -> Dict[str, Any]:
    if isinstance(lst, list) and lst:
        x = lst[0]
        return x if isinstance(x, dict) else {}
    return {} if lst is None else (lst if isinstance(lst, dict) else {})

def _get(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None

def _div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        if a is None or b in (None, 0):
            return None
        return float(a) / float(b)
    except Exception:
        return None

def _is_pos_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0

# ---- robust peer stats -----------------------------------------------------

def robust_peer_pe(peers: List[Dict[str, Any]], min_n: int = 3) -> Optional[float]:
    """
    Compute a trimmed mean of positive, sane P/E values to avoid outliers.
    Keep 25th–75th percentile and average.
    """
    vals = [
        float(p.get("pe"))
        for p in peers or []
        if isinstance(p, dict) and _is_pos_num(p.get("pe")) and 0 < float(p.get("pe")) < 100
    ]
    if len(vals) < min_n:
        return None
    vals.sort()
    k1, k2 = int(0.25 * len(vals)), int(0.75 * len(vals))
    mid = vals[k1:k2] or vals
    return sum(mid) / len(mid)

# ---- fundamentals aggregator -----------------------------------------------

def fetch_fundamentals(symbol: str) -> Dict[str, Any]:
    """
    Pulls TTM-ish fundamentals from FMP with resilient key mapping.
    Returns a dict with the keys your pipeline expects.
    """
    sym = symbol.upper()

    metrics_ttm = fmp.key_metrics_ttm(sym) or []
    km = _first(metrics_ttm)

    # Try quote/profile for market cap as a fallback
    mc = fmp.market_cap(sym)
    ev_row = fmp.enterprise_values(sym) or {}
    ev = ev_row.get("enterpriseValue") if isinstance(ev_row, dict) else None
    if not isinstance(ev, (int, float)):
        # some endpoints return list; grab first row if needed
        ev = _first(fmp.enterprise_values(sym)).get("enterpriseValue")

    # Try to obtain revenue/income/FCF from key metrics TTM; fallback to statements if needed
    revenue = _get(km, ["revenueTTM", "revenue_ttm", "revenue"])
    net_income = _get(km, ["netIncomeTTM", "net_income_ttm", "netIncome"])
    fcf = _get(km, ["freeCashFlowTTM", "fcfTTM", "freeCashFlow"])

    if revenue is None or net_income is None or fcf is None:
        # fallback to latest annual statements as a last resort
        inc = _first(fmp.income_statement(sym, period="annual", limit=1))
        cfs = _first(fmp.cash_flow_statement(sym, period="annual", limit=1))
        if revenue is None:
            revenue = _get(inc, ["revenue", "totalRevenue", "sales"])
        if net_income is None:
            net_income = _get(inc, ["netIncome", "netIncomeApplicableToCommonShares", "netincome"])
        if fcf is None:
            # FCF ~ operatingCashFlow - capitalExpenditure
            ocf = _get(cfs, ["netCashProvidedByOperatingActivities", "operatingCashFlow", "cashFlowFromOperations"])
            capex = _get(cfs, ["capitalExpenditure", "capex"])
            fcf = ocf - capex if (ocf is not None and capex is not None) else None

    # margins/returns from key metrics if available
    gross_margin = _get(km, ["grossProfitMarginTTM", "grossMarginTTM", "grossMargin"])
    op_margin    = _get(km, ["operatingMarginTTM", "opMarginTTM", "operatingMargin"])
    net_margin   = _get(km, ["netProfitMarginTTM", "netMarginTTM", "netMargin"])
    roe          = _get(km, ["returnOnEquityTTM", "roeTTM", "roe"])
    roic         = _get(km, ["returnOnInvestedCapitalTTM", "roicTTM", "roic"])
    pe           = _get(km, ["peTTM", "priceEarningsRatioTTM", "peRatioTTM", "pe"])

    fundamentals: Dict[str, Any] = {
        "revenue": revenue,
        "net_income": net_income,
        "fcf": fcf,
        "gross_margin": gross_margin,
        "op_margin": op_margin,
        "net_margin": net_margin,
        "roe": roe,
        "roic": roic,
        "market_cap": mc,
        "enterprise_value": ev,
        "pe": pe,
    }

    # FCF yield if we can compute it
    fundamentals["fcf_yield"] = _div(fcf, mc)

    return fundamentals
