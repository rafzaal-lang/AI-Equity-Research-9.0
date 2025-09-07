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

# Permissive CORS (UI compatibility)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten if you have a fixed UI origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReportResponse(BaseModel):
    symbol: str
    markdown: str

class ReportRequest(BaseModel):
    ticker: Optional[str] = Field(default=None)
    symbol: Optional[str] = Field(default=None)

    def get_symbol(self) -> str:
        return (self.ticker or self.symbol or "").strip()

# --------------------------------------------------------------------------------------
# ROOT & HEALTH
# --------------------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "AI Equity Research - Reports Service",
        "version": "9.2",
        "status": "healthy",
        "endpoints": {
            "health": "/v1/health",
            "report_get": "/v1/report/{ticker}",
            "report_post": "/report (POST), /v1/report (POST), /report (GET), /report/{ticker} (POST)",
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

# --------------------------------------------------------------------------------------
# METRICS
# --------------------------------------------------------------------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# --------------------------------------------------------------------------------------
# DEBUG
# --------------------------------------------------------------------------------------
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
    # FMP: historical prices
    try:
        hist = fmp.historical_prices(ticker, limit=5)
        results["fmp_historical"] = "SUCCESS" if hist else "NO_DATA"
    except Exception as e:
        results["fmp_historical"] = f"FAILED: {e}"
    # Macro (FRED)
    try:
        macro = macro_snapshot()
        results["macro_fred"] = "SUCCESS" if macro else "NO_DATA"
    except Exception as e:
        results["macro_fred"] = f"FAILED: {e}"
    # OpenAI (optional)
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
    # Financial model
    try:
        model = build_model(ticker)
        results["financial_model"] = "SUCCESS" if model and "error" not in model else "NO_DATA"
    except Exception as e:
        results["financial_model"] = f"FAILED: {e}"
    # Comps
    try:
        comp = comps_table(ticker)
        results["comps"] = "SUCCESS" if comp else "NO_DATA"
    except Exception as e:
        results["comps"] = f"FAILED: {e}"
    return results

# --------------------------------------------------------------------------------------
# UTIL: extract & guess ticker from any kind of request
# --------------------------------------------------------------------------------------
_TICKER_GUESS_RE = re.compile(r"[A-Za-z]{1,6}(\.[A-Za-z]{1,3})?")

async def _extract_symbol_from_request(request: Request) -> str:
    """
    Tries very hard to find a ticker symbol in JSON, form-data, query params, or raw text.
    Logs request headers and a short body snippet for debugging (safe).
    """
    # Log content-type + small body sample
    ctype = request.headers.get("content-type", "")
    try:
        raw_body = (await request.body()) or b""
    except Exception:
        raw_body = b""
    body_sample = raw_body[:256].decode(errors="ignore")
    log.info("POST /report content-type=%s body_sample=%r", ctype, body_sample)

    # 1) Attempt JSON
    sym = ""
    try:
        if "application/json" in ctype.lower():
            data = json.loads(body_sample) if body_sample else await request.json()
            if isinstance(data, dict):
                for k in ("ticker", "symbol", "t", "s", "security", "code"):
                    v = data.get(k)
                    if isinstance(v, str) and v.strip():
                        sym = v.strip()
                        break
                # Arrays like {"ticker":["AAPL"]} or nested payloads
                if not sym:
                    for k, v in data.items():
                        if isinstance(v, list) and v and isinstance(v[0], str):
                            m = _TICKER_GUESS_RE.match(v[0].strip())
                            if m:
                                sym = v[0].strip()
                                break
                        if isinstance(v, dict):
                            for kk, vv in v.items():
                                if isinstance(vv, str):
                                    m = _TICKER_GUESS_RE.match(vv.strip())
                                    if m:
                                        sym = vv.strip()
                                        break
                            if sym:
                                break
            elif isinstance(data, str):
                sym = data.strip()
    except Exception:
        pass

    # 2) Form data
    if not sym:
        try:
            form = await request.form()
            # typical names
            for k in ("ticker", "symbol", "t", "s"):
                v = form.get(k)
                if v and isinstance(v, str) and v.strip():
                    sym = v.strip()
                    break
            # catch-all (e.g., ticker[])
            if not sym:
                for k, v in form.items():
                    if isinstance(v, str):
                        m = _TICKER_GUESS_RE.match(v.strip())
                        if m:
                            sym = v.strip()
                            break
        except Exception:
            pass

    # 3) Query string
    if not sym:
        qp = request.query_params
        for k in ("ticker", "symbol", "t", "s", "code"):
            v = qp.get(k)
            if v and v.strip():
                sym = v.strip()
                break

    # 4) Raw text fallback
    if not sym and body_sample:
        m = _TICKER_GUESS_RE.search(body_sample.strip())
        if m:
            sym = m.group(0)

    # Normalize
    sym = (sym or "").upper().strip()
    # FINAL FALLBACK
    if not sym:
        sym = os.getenv("DEFAULT_TICKER", "AAPL").upper()
        log.warning("No ticker found in request; falling back to %s", sym)
    return sym

# --------------------------------------------------------------------------------------
# AI-ONLY ANALYSIS (JSON)
# --------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------
# REPORT BUILDING
# --------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------
# FULL REPORT (GET & POST) – VERY FORGIVING INPUTS
# --------------------------------------------------------------------------------------
@app.get("/v1/report/{ticker}", response_model=ReportResponse)
def get_report_v1(ticker: str):
    try:
        md = _build_report_markdown(ticker)
        return ReportResponse(symbol=ticker.upper(), markdown=md)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# GET /report?ticker=AAPL (compat for some UIs); if missing, fallback to DEFAULT_TICKER/AAPL
@app.get("/report", response_model=ReportResponse)
def get_report_compat(ticker: Optional[str] = Query(default=None), symbol: Optional[str] = Query(default=None)):
    sym = (ticker or symbol or os.getenv("DEFAULT_TICKER", "AAPL")).strip()
    try:
        md = _build_report_markdown(sym)
        return ReportResponse(symbol=sym.upper(), markdown=md)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# POST /v1/report with JSON body (kept for API users)
@app.post("/v1/report", response_model=ReportResponse)
def post_report_v1(req: ReportRequest):
    sym = (req.get_symbol() or os.getenv("DEFAULT_TICKER", "AAPL")).strip()
    try:
        md = _build_report_markdown(sym)
        return ReportResponse(symbol=sym.upper(), markdown=md)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# POST /report – accepts JSON, form-data, querystring, or raw text bodies; falls back to DEFAULT_TICKER/AAPL
@app.post("/report", response_model=ReportResponse)
async def post_report_compat(
    request: Request,
    ticker_body: Optional[str] = Body(default=None),
    symbol_body: Optional[str] = Body(default=None),
    ticker_q: Optional[str] = Query(default=None),
    symbol_q: Optional[str] = Query(default=None),
):
    # Fast path: direct Body fields from Body(...)
    sym = (ticker_body or symbol_body or "").strip()

    # Slow path: examine the request thoroughly
    if not sym:
        sym = await _extract_symbol_from_request(request)

    # Query override if provided explicitly
    if ticker_q or symbol_q:
        sym = (ticker_q or symbol_q).strip()

    try:
        md = _build_report_markdown(sym)
        # Helpful debug header so you can see where the symbol came from (fallback or not)
        return JSONResponse(
            content=ReportResponse(symbol=sym.upper(), markdown=md).model_dump(),
            headers={"X-Report-Symbol": sym.upper()}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Also allow POST /report/{ticker} (some UIs prefer path param on POST)
@app.post("/report/{ticker}", response_model=ReportResponse)
async def post_report_with_path(ticker: str):
    sym = (ticker or os.getenv("DEFAULT_TICKER", "AAPL")).strip()
    try:
        md = _build_report_markdown(sym)
        return ReportResponse(symbol=sym.upper(), markdown=md)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------------------------------------------
# DEV ENTRYPOINT
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8086")))
