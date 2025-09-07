# apis/reports/service.py
import sys, os, re, json, logging
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from datetime import datetime, date
from typing import Any, Dict, Optional
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Response, Request, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
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

VERSION = "9.6"

app = FastAPI(title="Reports Service")

# Permissive CORS for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your UI origin if desired
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReportResponse(BaseModel):
    symbol: str
    markdown: str

# --------------------------------------------------------------------------------------
# Global softeners: keep requests from failing hard with 400/422
# --------------------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def _validation_softener(request: Request, exc: RequestValidationError):
    # If anything tries to 422/400 due to body shape, return a default report instead
    print(f"[reports] validation_softener on {request.url.path}: {exc}")
    sym = os.getenv("DEFAULT_TICKER", "AAPL").upper()
    try:
        md = _build_report_markdown(sym)
        return JSONResponse({"symbol": sym, "markdown": md})
    except Exception as e:
        return JSONResponse({"detail": f"softened validation error: {e}"}, status_code=500)

@app.exception_handler(StarletteHTTPException)
async def _http_softener(request: Request, exc: StarletteHTTPException):
    # If it's a 400 on /report, try to recover by producing a default report
    if request.url.path.rstrip("/") == "/report" and exc.status_code == 400:
        print(f"[reports] http_softener recovered 400 for /report: {exc.detail}")
        sym = os.getenv("DEFAULT_TICKER", "AAPL").upper()
        try:
            md = _build_report_markdown(sym)
            return JSONResponse({"symbol": sym, "markdown": md})
        except Exception as e:
            return JSONResponse({"detail": f"softened http error: {e}"}, status_code=500)
    # otherwise pass through
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

# --------------------------------------------------------------------------------------
# ROOT & HEALTH
# --------------------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "AI Equity Research - Reports Service",
        "version": VERSION,
        "status": "healthy",
        "endpoints": {
            "health": "/v1/health",
            "report_get": "/v1/report/{ticker}",
            "report_post": "/report (POST), /v1/report (POST), /report (GET), /report/{ticker} (POST), /report/ (POST)",
            "ai_analysis": "/v1/ai-analysis/{ticker}",
            "metrics": "/metrics",
            "docs": "/docs",
        },
    }

@app.get("/version")
def version():
    return {"version": VERSION}

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
        err = model.get("error") if isinstance(model, dict) else None
        results["financial_model"] = "SUCCESS" if not err else f"ERROR_VALUE:{err!r}"
    except Exception as e:
        results["financial_model"] = f"FAILED: {e}"
    try:
        comp = comps_table(ticker)
        results["comps"] = "SUCCESS" if comp else "NO_DATA"
    except Exception as e:
        results["comps"] = f"FAILED: {e}"
    return results

# --------------------------------------------------------------------------------------
# Body parsing helpers (zero reliance on request.form())
# --------------------------------------------------------------------------------------
_TICKER_GUESS_RE = re.compile(r"[A-Za-z]{1,6}(\.[A-Za-z]{1,3})?")

def _guess_ticker_from_text(text: str) -> Optional[str]:
    m = _TICKER_GUESS_RE.search(text or "")
    return m.group(0).upper() if m else None

async def _extract_symbol_from_request(request: Request) -> str:
    """
    Avoids Starlette's form parser entirely. Reads raw body bytes, tries:
    1) JSON   2) urlencoded  3) text regex
    Always returns something (defaulting to env or AAPL).
    """
    ctype = (request.headers.get("content-type") or "").lower()
    try:
        raw = (await request.body()) or b""
    except Exception:
        raw = b""
    sample = raw[:256].decode(errors="ignore")
    print(f"[reports] compat hit; ctype={ctype} sample={sample!r}")

    # JSON
    if "json" in ctype:
        try:
            data = json.loads(raw.decode(errors="ignore"))
            if isinstance(data, dict):
                for k in ("ticker", "symbol", "t", "s", "security", "code"):
                    v = data.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip().upper()
            elif isinstance(data, str) and data.strip():
                return data.strip().upper()
        except Exception:
            pass

    # urlencoded (we parse manually; no starlette form parser)
    if "application/x-www-form-urlencoded" in ctype:
        try:
            kv = parse_qs(raw.decode(errors="ignore"), keep_blank_values=True)
            for k in ("ticker", "symbol", "t", "s", "code"):
                arr = kv.get(k)
                if arr and isinstance(arr[0], str) and arr[0].strip():
                    return arr[0].strip().upper()
        except Exception:
            pass

    # text/plain or anything else: regex guess
    tick = _guess_ticker_from_text(sample or "")
    if tick:
        return tick

    return os.getenv("DEFAULT_TICKER", "AAPL").upper()

# --------------------------------------------------------------------------------------
# REPORT BUILDING
# --------------------------------------------------------------------------------------
def _build_report_markdown(ticker: str) -> str:
    model = build_model(ticker)

    # Only treat as error if the error value is meaningful/truthy
    if isinstance(model, dict):
        err = model.get("error", None)
        if err not in (None, "", 0, False):
            raise HTTPException(status_code=404, detail=str(err))
    else:
        print(f"[reports] build_model returned non-dict for {ticker}: {type(model).__name__}")

    macro = macro_snapshot()

    # Prices for quant signals
    try:
        hist = fmp.historical_prices(ticker, limit=300) or []
    except Exception as e:
        print(f"[reports] historical_prices failed for {ticker}: {e}")
        hist = []

    try:
        q_mom = momentum(hist) if hist else {}
    except Exception as e:
        print(f"[reports] momentum failed for {ticker}: {e}")
        q_mom = {}
    try:
        q_rsi = rsi(hist) if hist else {}
    except Exception as e:
        print(f"[reports] rsi failed for {ticker}: {e}")
        q_rsi = {}
    try:
        q_sma = sma_cross(hist) if hist else {}
    except Exception as e:
        print(f"[reports] sma_cross failed for {ticker}: {e}")
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
            "fundamentals": (model or {}).get("core_financials", {}) if isinstance(model, dict) else {},
            "dcf":          (model or {}).get("dcf_valuation", {}) if isinstance(model, dict) else {},
            "valuation":    (model or {}).get("valuation", {})      if isinstance(model, dict) else {},
            "comps": {
                "peers": ((model or {}).get("comps", {}) if isinstance(model, dict) else {}).get("peers")
                         or (comps_table(ticker) or {}).get("peers", [])
            },
            "quarter": (model or {}).get("quarter", {}) if isinstance(model, dict) else {},
            "citations": (model or {}).get("citations", []) if isinstance(model, dict) else [],
            "risks": (model or {}).get("risks", []) if isinstance(model, dict) else [],
        },
    )
    return md

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
        if isinstance(model, dict):
            err = model.get("error", None)
            if err not in (None, "", 0, False):
                raise HTTPException(status_code=404, detail=str(err))
        comp = comps_table(ticker)
        financial_data = {
            "fundamentals": (model or {}).get("core_financials", {}) if isinstance(model, dict) else {},
            "dcf_valuation": (model or {}).get("dcf_valuation", {}) if isinstance(model, dict) else {},
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
# FULL REPORT (GET & POST) – ULTRA-FORGIVING
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

# GET /report?ticker=AAPL
@app.get("/report", response_model=ReportResponse)
def get_report_qs(ticker: Optional[str] = Query(default=None), symbol: Optional[str] = Query(default=None)):
    sym = (ticker or symbol or os.getenv("DEFAULT_TICKER", "AAPL")).strip().upper()
    try:
        md = _build_report_markdown(sym)
        return ReportResponse(symbol=sym, markdown=md)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# POST /v1/report (JSON: {"ticker":"AAPL"} or {"symbol":"AAPL"})
@app.post("/v1/report", response_model=ReportResponse)
async def post_report_v1(request: Request):
    # parse JSON leniently
    sym = ""
    try:
        data = await request.body()
        if data:
            obj = json.loads(data.decode(errors="ignore"))
            if isinstance(obj, dict):
                sym = (obj.get("ticker") or obj.get("symbol") or "").strip().upper()
            elif isinstance(obj, str):
                sym = obj.strip().upper()
    except Exception:
        sym = ""
    if not sym:
        sym = os.getenv("DEFAULT_TICKER", "AAPL").upper()
    try:
        md = _build_report_markdown(sym)
        return ReportResponse(symbol=sym, markdown=md)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# POST /report and /report/ – **no** Body parsers used
@app.post("/report", response_model=ReportResponse)
@app.post("/report/", response_model=ReportResponse)
async def post_report_compat(request: Request, ticker: Optional[str] = Query(default=None), symbol: Optional[str] = Query(default=None)):
    print("[reports] /report compat endpoint invoked")
    # 1) Querystring wins if present
    sym = (ticker or symbol or "").strip().upper()

    # 2) Otherwise, try to extract from raw body (JSON/urlencoded/text)
    if not sym:
        sym = await _extract_symbol_from_request(request)

    # 3) Final fallback
    if not sym:
        sym = os.getenv("DEFAULT_TICKER", "AAPL").upper()

    try:
        md = _build_report_markdown(sym)
        return JSONResponse(
            content=ReportResponse(symbol=sym, markdown=md).model_dump(),
            headers={"X-Report-Symbol": sym}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# POST /report/{ticker}
@app.post("/report/{ticker}", response_model=ReportResponse)
async def post_report_with_path(ticker: str):
    sym = (ticker or os.getenv("DEFAULT_TICKER", "AAPL")).strip().upper()
    try:
        md = _build_report_markdown(sym)
        return ReportResponse(symbol=sym, markdown=md)
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
