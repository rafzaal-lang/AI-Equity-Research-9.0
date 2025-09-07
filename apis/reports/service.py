# apis/reports/service.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from datetime import datetime, date
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
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

app = FastAPI(title="Reports Service")

class ReportResponse(BaseModel):
    symbol: str
    markdown: str

# Accept both { "ticker": "AAPL" } and { "symbol": "AAPL" }
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
        "version": "9.0",
        "status": "healthy",
        "endpoints": {
            "health": "/v1/health",
            "report": "/v1/report/{ticker}",
            "report_post": "/report  (POST)",
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
    return Response(status_code=204)  # No content

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
# FULL REPORT (GET & POST)
# --------------------------------------------------------------------------------------
def _build_report_markdown(ticker: str) -> str:
    # Core data
    model = build_model(ticker)
    if "error" in model:
        raise HTTPException(status_code=404, detail=model["error"])

    macro = macro_snapshot()

    # Prices for quant signals (guard if provider lacks the function)
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

    # Compose markdown (new signature compose(symbol, as_of, data=...))
    md = compose(
        ticker.upper(),
        as_of=(macro or {}).get("as_of") or date.today().isoformat(),
        data={
            "call": "Review",
            "conviction": 6.5,
            "target_low": "—",
            "target_high": "—",
            "momentum": q_mom,
            "fundamentals": model.get("core_financials", {}),
            "dcf": model.get("dcf_valuation", {}),
            "valuation": model.get("valuation", {}),
            "comps": {"peers": (model.get("comps") or {}).get("peers") or (comps_table(ticker) or {}).get("peers", [])},
            "quarter": model.get("quarter", {}),
            "citations": model.get("citations", []),
            "risks": model.get("risks", []),
        },
    )
    return md

@app.get("/v1/report/{ticker}", response_model=ReportResponse)
def get_report(ticker: str):
    try:
        md = _build_report_markdown(ticker)
        return ReportResponse(symbol=ticker.upper(), markdown=md)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# UI compatibility: allow POST /report and POST /v1/report with body
@app.post("/report", response_model=ReportResponse)
@app.post("/v1/report", response_model=ReportResponse)
def post_report(req: ReportRequest):
    symbol = req.get_symbol()
    if not symbol:
        raise HTTPException(status_code=400, detail="ticker (or symbol) is required")
    try:
        md = _build_report_markdown(symbol)
        return ReportResponse(symbol=symbol.upper(), markdown=md)
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
