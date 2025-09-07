import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.services.ai.financial_analyst import ai_financial_analyst
from src.services.ai.risk_analyzer import ai_risk_analyzer
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import json
try:
    from src.services.report.composer import compose
except Exception as _composer_err:
    def compose(symbol: str, as_of: str, data=None, **kw):
        parts = [
            f"# {symbol} — Equity Research Note (as of {as_of})",
            "",
            "_(composer fallback used — original import failed)_",
        ]
        if data:
            parts += ["", "```json", json.dumps(data, indent=2), "```"]
        return "\n".join(parts)

from src.services.financial_modeler import build_model
from src.services.macro.snapshot import macro_snapshot
from src.services.quant.signals import momentum, rsi, sma_cross
from src.services.comps.engine import comps_table
from src.services.providers import fmp_provider as fmp
from src.services.ai.financial_analyst import ai_financial_analyst
from src.services.ai.risk_analyzer import ai_risk_analyzer

app = FastAPI(title="Reports Service")

class ReportResponse(BaseModel):
    symbol: str
    markdown: str

# ROOT ENDPOINT - Fix for 404 on /
@app.get("/")
def root():
    return {
        "service": "AI Equity Research - Reports Service",
        "version": "8.0",
        "status": "healthy",
        "endpoints": {
            "health": "/v1/health",
            "report": "/v1/report/{ticker}",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

# HEALTH CHECK ENDPOINTS - Fix for health check 404s
@app.get("/health")
def health_simple():
    return {"status": "healthy"}

@app.get("/v1/health")
def health():
    return {"status": "healthy"}

# FAVICON ENDPOINT - Fix for favicon.ico 404s
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)  # No content

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# DEBUG ENDPOINT - Check API credentials
@app.get("/debug/apis")
def debug_apis():
    import os
    return {
        "redis_url": "SET" if os.getenv('REDIS_URL') else "MISSING",
        "fmp_key": "SET" if os.getenv('FMP_API_KEY') else "MISSING", 
        "fred_key": "SET" if os.getenv('FRED_API_KEY') else "MISSING",
        "openai_key": "SET" if os.getenv('OPENAI_API_KEY') else "MISSING",
        "sec_agent": os.getenv('SEC_USER_AGENT', 'MISSING')
    }

# DEBUG ENDPOINT - Test individual APIs
@app.get("/debug/test-apis/{ticker}")
def test_individual_apis(ticker: str):
    results = {}
    
    # Test FMP API
    try:
        hist = fmp.historical_prices(ticker, limit=5)
        results["fmp_historical"] = "SUCCESS" if hist else "NO_DATA"
    except Exception as e:
        results["fmp_historical"] = f"FAILED: {str(e)}"
    
    # Test Macro (FRED)
    try:
        macro = macro_snapshot()
        results["macro_fred"] = "SUCCESS" if macro else "NO_DATA"
    except Exception as e:
        results["macro_fred"] = f"FAILED: {str(e)}"
    
   # NEW: AI Analysis endpoint
@app.get("/v1/ai-analysis/{ticker}")
def get_ai_analysis(ticker: str):
    """Get pure AI analysis without full report."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="OpenAI API key required")
    
    try:
        # Get financial data
        model = build_model(ticker)
        if "error" in model:
            raise HTTPException(status_code=404, detail=model["error"])
        
        comp = comps_table(ticker)
        
        # Prepare data for AI
        financial_data = {
            "fundamentals": model.get("core_financials", {}),
            "dcf_valuation": model.get("dcf_valuation", {}),
            "comps": comp
        }
        
        # Generate AI analysis
        ai_analysis = ai_financial_analyst.analyze_company(ticker, financial_data)
        ai_risks = ai_risk_analyzer.analyze_financial_risks(financial_data)
        
        return {
            "symbol": ticker.upper(),
            "analysis": ai_analysis,
            "risks": ai_risks,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Test OpenAI API
    try:
        import openai
        import os
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Simple test call
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        results["openai"] = "SUCCESS" if response else "NO_DATA"
    except Exception as e:
        results["openai"] = f"FAILED: {str(e)}"
    
    # Test Financial Model
    try:
        model = build_model(ticker)
        results["financial_model"] = "SUCCESS" if model and "error" not in model else "NO_DATA"
    except Exception as e:
        results["financial_model"] = f"FAILED: {str(e)}"
    
    # Test Comps
    try:
        comp = comps_table(ticker)
        results["comps"] = "SUCCESS" if comp else "NO_DATA"
    except Exception as e:
        results["comps"] = f"FAILED: {str(e)}"
    
    return results

@app.get("/v1/report/{ticker}", response_model=ReportResponse)
def get_report(ticker: str):
    try:
        # Gather all data
        model = build_model(ticker)
        if "error" in model:
            raise HTTPException(status_code=404, detail=model["error"])
        
        macro = macro_snapshot()
        hist = fmp.historical_prices(ticker, limit=300)
        q = {
            "momentum": momentum(hist),
            "rsi": rsi(hist),
            "sma": sma_cross(hist),
        }
        comp = comps_table(ticker)
        
        # Compose report
        md = compose({
            "symbol": ticker.upper(),
            "as_of": macro.get("as_of"),
            "call": "Review", "conviction": 6.5, "target_low": "—", "target_high": "—",
            "macro": macro,
            "quant": {"momentum": q["momentum"], "rsi": q["rsi"], "sma": q["sma"]},
            "fundamentals": model["core_financials"],
            "dcf": model["dcf_valuation"],
            "comps": {"peers": comp["peers"][:10]},
        })
        return ReportResponse(symbol=ticker.upper(), markdown=md)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8086)




