# ui_minimal.py
from __future__ import annotations
import os
import logging
from typing import Optional, List
from fastapi import FastAPI, Form, Request, Query, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from jinja2 import Environment, BaseLoader, select_autoescape
from datetime import datetime

app = FastAPI(title="Equity Research — Minimal UI")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/health")
def health():
    return {"ok": True}

# Quiet Render "HEAD / 405" health checks
@app.head("/", response_class=PlainTextResponse)
def _head_root():
    return PlainTextResponse("", status_code=200)

BASE_CURRENCY = os.getenv("BASE_CURRENCY", "USD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
DEFAULT_TICKER = os.getenv("DEFAULT_TICKER", "AAPL").strip() or "AAPL"

# Import your services
try:
    from src.services.financial_modeler import build_model
    from src.services.comps.engine import comps_table, latest_metrics
    from src.services.wacc.peer_beta import peer_beta_wacc
    from reports.composer import compose as compose_report
    from src.services.macro.snapshot import macro_snapshot
    from src.services.quant.signals import momentum, rsi, sma_cross
    from src.services.providers import fmp_provider as fmp
except ImportError as e:
    logger.error(f"Import error: {e}")

def _compat_call(func, *args, **kwargs):
    """Drop unknown kwargs so older service signatures won't crash."""
    import inspect
    sig = inspect.signature(func)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(*args, **allowed)

def _html_error(note: str, status: int = 502) -> HTMLResponse:
    from html import escape
    block = f'<pre class="muted" style="white-space:pre-wrap">{escape(note)}</pre>'
    return HTMLResponse(f"<html><body><h1>Error</h1>{block}</body></html>", status_code=status)

def _build_report_markdown(ticker: str) -> str:
    """Build report markdown."""
    try:
        model = build_model(ticker, force_refresh=False)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model error: {str(e)}")

        if isinstance(model, dict) and "error" in model:
        logging.getLogger(__name__).error("Model error for %s: %r", ticker, model["error"])
        # Force a readable string for HTML
        err = model["error"]
        if not isinstance(err, str):
            err = repr(err)
        return _html_error(f"Model error: {err}", status=400)

    if isinstance(model, dict) and "error" in model:
        raise HTTPException(status_code=400, detail=f"Model error: {model['error']}")

    try:
        macro = macro_snapshot()
        
        # Get price history for quant signals
        try:
            hist = fmp.historical_prices(ticker, limit=300) or []
        except Exception:
            hist = []

        try:
            q_mom = momentum(hist) if hist else {}
        except Exception:
            q_mom = {}

        # Compose markdown
        md = compose_report({
            "symbol": ticker.upper(),
            "as_of": "latest",
            "call": "Review",
            "conviction": 7.0,
            "target_low": "—",
            "target_high": "—",
            "base_currency": BASE_CURRENCY,
            "fundamentals": model.get("core_financials", {}),
            "dcf": model.get("dcf_valuation", {}),
            "valuation": {"wacc": None},
            "comps": {"peers": []},
            "citations": [],
            "quarter": {},
            "artifact_id": "ui-session",
        })
        return md
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")

# Simple HTML template
BASE_HTML = """
<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Equity Research</title>
<style>
:root { --bg:#f7f7f5; --panel:#fff; --ink:#0a0a0a; --muted:#6b7280; --line:#e5e7eb; --radius:16px; }
*{box-sizing:border-box} html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);
  font-family: ui-sans-serif,-apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue",Helvetica,Arial,"Segoe UI",Roboto,"Noto Sans",sans-serif}
a{color:#0f172a;text-decoration:none} a:hover{text-decoration:underline}
.container{max-width:1100px;margin:0 auto;padding:24px}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}
.brand{font-weight:700;letter-spacing:.3px}
.nav a{margin-left:12px;padding:8px 12px;border:1px solid var(--line);border-radius:999px;background:#fff}
.nav a.active{background:var(--ink);color:#fff;border-color:var(--ink)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:18px}
.btn{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:12px;border:1px solid var(--ink);color:#fff;background:var(--ink);cursor:pointer}
.input{width:100%;padding:10px 12px;border-radius:12px;border:1px solid var(--line);background:#fff}
pre{white-space:pre-wrap;background:#f8f9fa;padding:12px;border-radius:8px;overflow-x:auto}
</style></head><body>
  <div class="container">
    <header>
      <div class="brand">Equity Research</div>
    </header>
    {{ content | safe }}
  </div>
</body></html>
"""

REPORT_FORM = """
<div class="panel">
  <form method="post" action="/report">
    <div style="margin-bottom:10px;">
      <label>Ticker:</label>
      <input class="input" type="text" name="ticker" placeholder="AAPL" required />
    </div>
    <button class="btn" type="submit">Generate Report</button>
  </form>
</div>
"""

env = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html"]))
def render(page: str, **kw) -> str:
    tpl = env.from_string(BASE_HTML)
    content_tpl = env.from_string(page)
    return tpl.render(content=content_tpl.render(**kw))

# Routes
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(render(REPORT_FORM))

@app.get("/report")
def get_report(ticker: Optional[str] = Query(default=None), symbol: Optional[str] = Query(default=None)):
    """GET /report?ticker=AAPL - returns JSON"""
    sym = (ticker or symbol or "").strip() or DEFAULT_TICKER
    if not sym:
        raise HTTPException(status_code=400, detail="ticker or symbol is required")
    
    try:
        md = _build_report_markdown(sym)
        return {"symbol": sym.upper(), "markdown": md}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/report", response_class=HTMLResponse)
def post_report(ticker: str = Form(...)):
    """POST /report with form data - returns HTML"""
    try:
        md = _build_report_markdown(ticker)
        import markdown as md_parser
        html_content = md_parser.markdown(md, extensions=["tables"])
        
        result_html = f"""
        <div class="panel">
          <h2>{ticker.upper()} Report</h2>
          <div style="margin-bottom:12px;">
            <a href="/report.md?ticker={ticker}" class="btn" style="text-decoration:none;">Download Markdown</a>
          </div>
          <div style="border:1px solid var(--line);padding:18px;border-radius:12px;">
            {html_content}
          </div>
        </div>
        """
        return HTMLResponse(render(result_html))
    except Exception as e:
        return _html_error(f"Error generating report: {str(e)}")

@app.get("/report.md", response_class=PlainTextResponse)
def download_report_md(ticker: str = Query(...)):
    """Download report as markdown"""
    try:
        md = _build_report_markdown(ticker)
        return PlainTextResponse(md, headers={"Content-Disposition": f"attachment; filename={ticker}_report.md"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/env", response_class=PlainTextResponse)
def debug_env():
    keys = [
        ("FMP_API_KEY", bool(os.getenv("FMP_API_KEY"))),
        ("OPENAI_API_KEY", bool(os.getenv("OPENAI_API_KEY"))),
        ("REDIS_URL", bool(os.getenv("REDIS_URL"))),
        ("BASE_CURRENCY", os.getenv("BASE_CURRENCY", "")),
        ("DEFAULT_TICKER", DEFAULT_TICKER),
    ]
    lines = [f"{k}={'SET' if v else 'MISSING'}" if isinstance(v, bool) else f"{k}={v}" for k, v in keys]
    return PlainTextResponse("\n".join(lines))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8090")))

