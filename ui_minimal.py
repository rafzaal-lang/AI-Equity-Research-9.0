# ui_minimal.py
from __future__ import annotations
import os
import logging
from typing import Optional
from fastapi import FastAPI, Form, Query, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from jinja2 import Environment, BaseLoader, select_autoescape

app = FastAPI(title="Equity Research — Minimal UI")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/health")
def health():
    return {"ok": True}

# Quiet Render "HEAD /" health probes
@app.head("/", response_class=PlainTextResponse)
def _head_root():
    return PlainTextResponse("", status_code=200)

BASE_CURRENCY = os.getenv("BASE_CURRENCY", "USD")
DEFAULT_TICKER = (os.getenv("DEFAULT_TICKER", "AAPL") or "AAPL").strip() or "AAPL"

# Imports
try:
    from src.services.financial_modeler import build_model
    from reports.composer import compose as compose_report
    from src.services.macro.snapshot import macro_snapshot
    from src.services.quant.signals import momentum
    from src.services.providers import fmp_provider as fmp
except ImportError as e:
    logger.error("Import error: %s", e)

def _html_error(note: str, status: int = 502) -> HTMLResponse:
    from html import escape
    block = f'<pre class="muted" style="white-space:pre-wrap">{escape(note)}</pre>'
    return HTMLResponse(f"<html><body><h1>Error</h1>{block}</body></html>", status_code=status)

def _build_report_markdown(ticker: str) -> str:
    """Build report markdown or raise HTTPException with useful details."""
    # 1) Build core model
    try:
        model = build_model(ticker, force_refresh=False)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model error: {type(e).__name__}: {e}")

    # 2) If service returned an error envelope, surface it
    if isinstance(model, dict) and "error" in model:
        logger.error("Model error for %s: %r", ticker, model["error"])
        err = model["error"]
        if not isinstance(err, str):
            err = repr(err)
        raise HTTPException(status_code=400, detail=f"Model error: {err}")

    # 3) Optional aux data (safe-fail)
    try:
        try:
            macro = macro_snapshot()
        except Exception:
            macro = {}

        try:
            hist = fmp.historical_prices(ticker, limit=300) or []
        except Exception:
            hist = []
        try:
            q_mom = momentum(hist) if hist else {}
        except Exception:
            q_mom = {}

        # 4) Compose final markdown
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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {type(e).__name__}: {e}")

# ---- Minimal HTML shell ----
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
.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:18px}
.btn{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:12px;border:1px solid var(--ink);color:#fff;background:var(--ink);cursor:pointer}
.input{width:100%;padding:10px 12px;border-radius:12px;border:1px solid var(--line);background:#fff}
pre{white-space:pre-wrap;background:#f8f9fa;padding:12px;border-radius:8px;overflow-x:auto}
</style></head><body>
  <div class="container">
    <header><div class="brand">Equity Research</div></header>
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

# ---- Routes ----
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(render(REPORT_FORM))

@app.get("/report")
def get_report(ticker: Optional[str] = Query(default=None), symbol: Optional[str] = Query(default=None)):
    sym = (ticker or symbol or "").strip() or DEFAULT_TICKER
    if not sym:
        raise HTTPException(status_code=400, detail="ticker or symbol is required")
    md = _build_report_markdown(sym)
    return {"symbol": sym.upper(), "markdown": md}

@app.post("/report", response_class=HTMLResponse)
def post_report(ticker: str = Form(...)):
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
          <div style="border:1px solid var(--line);padding:18px;border-radius:12px;">{html_content}</div>
        </div>
        """
        return HTMLResponse(render(result_html))
    except HTTPException as he:
        return _html_error(he.detail, status=he.status_code)
    except Exception as e:
        return _html_error(f"Error generating report: {type(e).__name__}: {e}")

@app.get("/report.md", response_class=PlainTextResponse)
def download_report_md(ticker: str = Query(...)):
    md = _build_report_markdown(ticker)
    return PlainTextResponse(md, headers={"Content-Disposition": f"attachment; filename={ticker}_report.md"})

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
