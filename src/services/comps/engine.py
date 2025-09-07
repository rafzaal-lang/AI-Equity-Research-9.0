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

def latest_metrics(symbol: str) -> Dict[str, Optional[float]]:
    inc = (fmp.income_statement(symbol, period="annual", limit=1) or [{}])[0]
    km  = (fmp.key_metrics_ttm(symbol) or [{}])[0]
    evs = (fmp.enterprise_values(symbol, period="quarter", limit=1) or [{}])[0]
    price = fmp.latest_price(symbol)
    sales = _num(inc.get("revenue"))
    ebitda = _num(inc.get("ebitda")) or _num(inc.get("operatingIncome"))
    eps = _num(km.get("netIncomePerShareTTM"))
    market_cap = _num(evs.get("marketCapitalization"))
    ev = _num(evs.get("enterpriseValue"))
    shares = _num(inc.get("weightedAverageShsOut"))
    return {
        "price": price, "sales": sales, "ebitda": ebitda, "eps_ttm": eps,
        "market_cap": market_cap, "enterprise_value": ev, "shares_out": shares,
        "ps": (market_cap / sales) if market_cap and sales else None,
        "pe": (price / eps) if price and eps else None,
        "ev_ebitda": (ev / ebitda) if ev and ebitda else None,
    }

def comps_table(symbol: str, max_peers: int = 15) -> Dict[str, Any]:
    tickers = [symbol.upper()] + peer_universe(symbol, max_peers=max_peers)
    rows = []
    for t in tickers:
        rows.append({"ticker": t, **latest_metrics(t)})
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

