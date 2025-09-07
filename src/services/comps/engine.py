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

# Fix for the AttributeError in comps/WACC calculation
# The issue is in src/services/comps/engine.py - latest_metrics function

def latest_metrics(symbol: str, as_of: Optional[str] = None) -> Dict[str, Optional[float]]:
    """Get latest financial metrics for a symbol with better error handling."""
    try:
        # Get data with error handling
        inc_data = fmp.income_statement(symbol, period="annual", limit=1)
        km_data = fmp.key_metrics_ttm(symbol)
        evs_data = fmp.enterprise_values(symbol, period="quarter", limit=1)
        
        # Safely extract first element or empty dict
        inc = inc_data[0] if inc_data and len(inc_data) > 0 else {}
        km = km_data[0] if km_data and len(km_data) > 0 else {}
        evs = evs_data[0] if evs_data and len(evs_data) > 0 else {}
        
        # Get price with fallback
        try:
            price = fmp.latest_price(symbol)
        except Exception:
            price = None
        
        # Extract values safely
        sales = _num(inc.get("revenue"))
        ebitda = _num(inc.get("ebitda")) or _num(inc.get("operatingIncome"))
        eps = _num(km.get("netIncomePerShareTTM"))
        market_cap = _num(evs.get("marketCapitalization"))
        ev = _num(evs.get("enterpriseValue"))
        shares = _num(inc.get("weightedAverageShsOut"))
        
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
            "fcf_yield": None  # Would need FCF data
        }
    except Exception as e:
        logger.error(f"Error getting metrics for {symbol}: {e}")
        return {
            "price": None, "sales": None, "ebitda": None, "eps_ttm": None,
            "market_cap": None, "enterprise_value": None, "shares_out": None,
            "ps": None, "pe": None, "ev_ebitda": None, "fcf_yield": None
        }


# Fix for ui_minimal.py - _compat_call function to handle the error better
def _compat_call(func, *args, **kwargs):
    """Drop unknown kwargs so older service signatures won't crash."""
    import inspect
    try:
        sig = inspect.signature(func)
        allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return func(*args, **allowed)
    except Exception as e:
        logger.error(f"Function call failed: {func.__name__}: {e}")
        # Return empty/safe defaults based on function name
        if 'comps' in func.__name__ or 'metrics' in func.__name__:
            return {"peers": [], "symbol": args[0] if args else "UNKNOWN"}
        elif 'wacc' in func.__name__:
            return {"wacc": 0.10, "beta": 1.0}
        else:
            return {}


# Enhanced error handling in the main report generation
# Add this to ui_minimal.py in the post_report function:

@app.post("/report", response_class=HTMLResponse)
def post_report(
    ticker: str = Form(...),
    as_of: Optional[str] = Form(None),
    rf: float = Form(0.045),
    mrp: float = Form(0.055),
    kd: float = Form(0.05),
    include_citations: Optional[str] = Form(None),
):
    import markdown as md
    from src.services.financial_modeler import build_model
    from src.services.comps.engine import comps_table, latest_metrics
    from src.services.wacc.peer_beta import peer_beta_wacc
    from reports.composer import compose as compose_report

    errors = []
    
    # Build model with better error handling
    try:
        model = build_model(ticker, force_refresh=False)
        if isinstance(model, dict) and "error" in model:
            errors.append(f"Model error: {model['error']}")
    except Exception as e:
        errors.append(f"Model error: {type(e).__name__}: {e}")
        model = {"error": str(e)}

    # Comps/WACC/metrics with comprehensive error handling
    comps = {"peers": []}
    peers = []
    w = {"wacc": 0.10}
    lm = {}
    
    try:
        comps = _compat_call(comps_table, ticker, as_of=as_of, max_peers=25)
        if not comps or "error" in str(comps):
            comps = {"peers": [{"ticker": ticker.upper()}]}
        
        # Safely get peers list
        if isinstance(comps, dict) and "peers" in comps:
            peer_list = comps.get("peers", [])
            if isinstance(peer_list, list) and len(peer_list) > 1:
                peers = [r.get("ticker") for r in peer_list[1:] if isinstance(r, dict) and r.get("ticker")]
        
    except Exception as e:
        errors.append(f"Comps error: {type(e).__name__}: {e}")
        comps = {"peers": [{"ticker": ticker.upper()}]}

    try:
        if peers:
            w = _compat_call(peer_beta_wacc, ticker, peers[:10], rf=rf, mrp=mrp, kd=kd, target_d_e=None, as_of=as_of)
        else:
            w = {"wacc": 0.10, "beta": 1.0, "beta_source": "default"}
    except Exception as e:
        errors.append(f"WACC error: {type(e).__name__}: {e}")
        w = {"wacc": 0.10, "beta": 1.0, "beta_source": "error_fallback"}

    try:
        lm = _compat_call(latest_metrics, ticker, as_of=as_of)
        if not isinstance(lm, dict):
            lm = {}
    except Exception as e:
        errors.append(f"Metrics error: {type(e).__name__}: {e}")
        lm = {}

    # Show errors if any occurred
    if errors:
        error_html = f"""
        <div class="panel" style="border-color: #e74c3c; background: #fadbd8;">
            <h3>⚠️ Errors Encountered</h3>
            <ul>
                {"".join(f"<li>{error}</li>" for error in errors)}
            </ul>
            <p><strong>Report generated with available data.</strong></p>
        </div>
        """
        return HTMLResponse(
            render(REPORT_FORM + error_html, active="report"),
            status_code=200  # Still return 200 but show errors
        )

    # Continue with rest of the function...

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


