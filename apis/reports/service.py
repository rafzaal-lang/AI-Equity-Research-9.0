# apis/reports/service.py
import sys, os, re, json, logging
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from datetime import datetime, date
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Response, Request, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.services.report.composer import compose
from src.services.financial_modeler import build_model
from src.services.macro.snapshot import macro_snapshot
from src.services.quant.signals import momentum, rsi, sma_cross
from src.services.comps.engine import comps_table
from src.services.providers import fmp_provider as fmp

# Optional AI analyzers (safe import)
try:
    from src.services.ai.financial_analyst import ai_financial_analyst
except Exception:
    ai_financial_analyst = None  # type: ignore

try:
    from src.services.ai.risk_analyzer import ai_risk_analyzer
except Exception:
    ai_risk_analyzer = None  # type: ignore

log = logging.getLogger(__name__)

app = FastAPI(title="Reports Service")

# Permissive CORS (tighten allow_origins if you have a fixed UI origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STRICT_INPUTS = os.getenv("STRICT_REPORT_INPUTS", "false").lower() == "true"
DEFAULT_TICKER = os.getenv("DEFAULT_TICKER", "AAPL").strip() or "AAPL"

class ReportResponse(BaseModel):
    symbol: str
    markdown: str

class ReportRequest(BaseModel):
    ticker: Optional[str] = Field(default=None)
    symbol: Optional[str] = Field(default=None)

    def get_symbol(self) -> str:
        return (self.ticker or self.symbol or "").strip()

# ----------------------------- Utilities ------------------------------------
_TICKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")  # allows BRK.B, RDS-A, SHOP, MSFT

def _looks_like_ticker(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s or len(s) > 10:
        return False
    return bool(_TICKER_RE.match(s))

def _deep_find_ticker(obj: Any) -> Optional[str]:
    """Recursively search any JSON-like structure for a plausible ticker."""
    try:
        if isinstance(obj, str) and _looks_like_ticker(obj):
            return obj.strip()
        if isinstance(obj, dict):
            # common keys first
            for k in ("ticker", "symbol", "t", "s", "security", "equity", "name", "q"):
                if k in obj and _looks_like_ticker(str(obj[k])):
                    return str(obj[k]).strip()
            # any value
            for v in obj.values():
                found = _deep_find_ticker(v)
                if found:
                    return found
        if isinstance(obj, list):
            for v in obj:
                found = _deep_find_ticker(v)
                if found:
                    return found
    except Exception:
        pass
    return None

async def _extract_symbol(request: Request,
                          ticker_body: Optional[str],
                          symbol_body: Optional[str],
                          ticker_q: Optional[str],
                          symbol_q: Optional[str]) -> Optional[str]:
    # 1) direct body fields from Body(...)
    sym = (ticker_body or symbol_body or "").strip()

    # 2) headers
    if not sym:
        headers = request.headers
        for hk in ("x-ticker", "x-symbol", "ticker", "symbol"):
            hv = headers.get(hk)
            if hv and _looks_like_ticker(hv):
                sym = hv.strip()
                break

    # 3) JSON body
    if not sym:
        try:
            data = await request.json()
            if isinstance(data, dict):
                # quick keys
                for k in ("ticker", "symbol", "t", "s", "q", "equity", "security"):
                    if k in data and _looks_like_ticker(str(data[k])):
                        sym = str(data[k]).strip()
                        break
                # deep search
                if not sym:
                    found = _deep_find_ticker(data)
                    if found:
                        sym = found.strip()
            elif isinstance(data, str) and _looks_like_ticker(data):
                sym = data.strip()
        except Exception:
            pass

    # 4) form-data
    if not sym:
        try:
            form = await request.form()
            # case-insensitive scan
            for k, v in form.multi_items():
                if _looks_like_ticker(str(v)):
                    sym = str(v).strip()
                    break
        except Exception:
            pass

    # 5) querystring
    if not sym:
        for v in (ticker_q, symbol_q, request.query_params.get("t"), request.query_params.get("s"), request.query_params.get("q")):
            if v and _looks_like_ticker(v):
                sym = v.strip()
                break

    # 6) raw text fallback
    if not sym:
        try:
            raw = (await request.body()).decode(errors="ignore").strip()
            if raw:
                # try raw as JSON first
                try:
                    obj = json.loads(raw)
                    found = _deep_find_ticker(obj)
                    if found:
                        sym = found.strip()
                except Exception:
                    if _looks_like_ticker(raw):
                        sym = raw
        except Exception:
            pass

    # 7) default
    if not sym and not STRICT_INPUTS:
        sym = DEFAULT_TICKER

    return sym

# ----------------------------- Root & Health --------------------------------
@app.get("/")
def root():
    return {
        "service": "AI Equity Research - Reports Service",
        "version": "9.2",
        "status": "healthy",
        "endpoints": {
            "health": "/v1/health",
            "report_get": "/v1/report/{ticker}",
            "report_post": "/report (POST), /v1/report (POST), /report (GET)",
            "ai_analysis": "/v1/ai-analysis/{ticker}",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_simple():
    return {"status": "healthy"}

@app.get("/v1/health")
def health():
    return {"status": "healthy"}

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

# ----------------------------- Metrics --------------------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ----------------------------- Debug ----------------------------------------
@app.get("/debug/apis")
def debug_apis():
    return {
        "redis_url": "SET" if os.getenv("REDIS_URL") else "MISSING",
        "fmp_key": "SET" if os.getenv("FMP_API_KEY") else "MISSING",
        "fred_key": "SET" if os.getenv("FRED_API_KEY") else "MISSING",
        "openai_key": "SET" if os.getenv("OPENAI_API_KEY") else "MISSING",
        "sec_agent": os.getenv("SEC_USER_AGENT", "MISSING"),
    }

@app.get("/debug/test-apis/{ticker}")
def test_individual_apis(ticker: str):
    results: Dict[str, Any] = {}
    try:
        hist = fmp.historical_prices(ticker, limit=5)
        results["fmp_historical"] = "SUCCESS" if hist else "NO_DATA"
    except Exception as e:
        results["fmp_historical"] = f"FAILED: {e}"
    try:
        macro = macro_snapshot()
        results["macro_fred"] = "SUCCESS" if macro else "NO_DATA"
    except Exception as e:
        results["macro_fred"] = f"FAILED: {e}"
    try:
        from openai import OpenAI
        if os.getenv("OPENAI_API_KEY"):
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                temperature=0.0,
            )
            results["openai"] = "SUCCESS" if r and r.choices else "NO_DATA"
        else:
            results["openai"] = "MISSING_KEY"
    except Exception as e:
        results["openai"] = f"FAILED: {e}"
    try:
        model = build_model(ticker)
        results["financial_model"] = "SUCCESS" if model and "error" not in model else "NO_DATA"
    except Exception as e:
        results["financial_model"] = f"FAILED: {e}"
    try:
        comp = comps_table(ticker)
        results["comps"] = "SUCCESS" if comp else "NO_DATA"
    except Exception as e:
        results["comps"] = f"FAILED: {e}"
    return results

# ----------------------------- AI-only JSON ---------------------------------
@app.get("/v1/ai-analysis/{ticker}")
def get_ai_analysis(ticker: str):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="OpenAI API key required")
    if ai_financial_analyst is None:
        raise HTTPException(status_code=500, detail="AI analyst not available")

    try:
        model = build_model(ticker)
        if "error" in model:
            raise HTTPException(status_code=404, detail=model["error"])
        comp = comps_table(ticker)
        financial_data = {
            "fundamentals": model.get("core_financials", {}),
            "dcf_valuation": model.get("dcf_valuation", {}),
            "comps": comp,
        }
        analysis = ai_financial_analyst.analyze_company(ticker, financial_data)
        risks = ai_risk_analyzer.analyze_financial_risks(financial_data) if ai_risk_analyzer else None
        return {
            "symbol": ticker.upper(),
            "analysis": analysis,
            "risks": risks,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------- Report Builder -------------------------------
def _build_report_markdown(ticker: str) -> str:
    model = build_model(ticker)
    if "error" in model:
        raise HTTPException(status_code=404, detail=model["error"])

    macro = macro_snapshot()

    # Prices for quant signals
    try:
        hist = fmp.historical_prices(ticker, limit=300) or []
    except Exception:
        hist = []

    try:
        q_mom = momentum(hist) if hist else {}
    except Exception:
        q_mom = {}
    try:
        q_rsi = rsi(hist) if hist else {}
    except Exception:
        q_rsi = {}
    try:
        q_sma = sma_cross(hist) if hist else {}
    except Exception:
        q_sma = {}

    md = compose(
        ticker.upper(),
        as_of=(macro or {}).get("as_of") or date.today().isoformat(),
        data={
            "call": "Review",
            "conviction": 6.5,
            "target_low": "—",
            "target_high": "—",
            "momentum": q_mom,  # composer expects 'momentum' at top-level
            "fundamentals": model.get("core_financials", {}),
            "dcf": model.get("dcf_valuation", {}),
            "valuation": model.get("valuation", {}),
            "comps": {
                "peers": (model.get("comps") or {}).get("peers")
                         or (comps_table(ticker) or {}).get("peers", [])
            },
            "quarter": model.get("quarter", {}),
            "citations": model.get("citations", []),
            "risks": model.get("risks", []),
        },
    )
    return md

# ----------------------------- Full Report (GET & POST) ---------------------
@app.get("/v1/report/{ticker}", response_model=ReportResponse)
def get_report_v1(ticker: str):
    try:
        md = _build_report_markdown(ticker)
        return ReportResponse(symbol=ticker.upper(), markdown=md)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# GET /report?ticker=AAPL (compat)
@app.get("/report", response_model=ReportResponse)
def get_report_compat(ticker: Optional[str] = Query(default=None), symbol: Optional[str] = Query(default=None)):
    sym = (ticker or symbol or "").strip() or (DEFAULT_TICKER if not STRICT_INPUTS else "")
    if not sym:
        raise HTTPException(status_code=400, detail="ticker (or symbol) is required")
    try:
        md = _build_report_markdown(sym)
        return ReportResponse(symbol=sym.upper(), markdown=md)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# POST /v1/report with JSON body
@app.post("/v1/report", response_model=ReportResponse)
def post_report_v1(req: ReportRequest):
    sym = req.get_symbol() or (DEFAULT_TICKER if not STRICT_INPUTS else "")
    if not sym:
        raise HTTPException(status_code=400, detail="ticker (or symbol) is required")
    try:
        md = _build_report_markdown(sym)
        return ReportResponse(symbol=sym.upper(), markdown=md)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# POST /report – accepts JSON, form-data, headers, querystring, or raw body
@app.post("/report", response_model=ReportResponse)
async def post_report_compat(
    request: Request,
    ticker_body: Optional[str] = Body(default=None),
    symbol_body: Optional[str] = Body(default=None),
    ticker_q: Optional[str] = Query(default=None),
    symbol_q: Optional[str] = Query(default=None),
):
    sym = await _extract_symbol(request, ticker_body, symbol_body, ticker_q, symbol_q)

    if not sym:
        # brief diagnostic (sanitized) to help debug without dumping full payloads
        try:
            body_preview = (await request.body())[:200]
        except Exception:
            body_preview = b""
        log.info("POST /report missing ticker. headers=%s query=%s body_preview=%s",
                 dict(request.headers), dict(request.query_params), body_preview)
        raise HTTPException(status_code=400, detail="ticker (or symbol) is required")

    try:
        md = _build_report_markdown(sym)
        return ReportResponse(symbol=sym.upper(), markdown=md)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------- Dev Entrypoint -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8086")))
