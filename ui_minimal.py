# ui_minimal.py
# Minimal, production-friendly FastAPI UI.
# Start (Render): uvicorn ui_minimal:app --host 0.0.0.0 --port $PORT

from __future__ import annotations
import os
import logging
from typing import Optional, List
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
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

# ---------- compat + error helpers ----------
def _compat_call(func, *args, **kwargs):
    """Drop unknown kwargs so older service signatures won't crash."""
    import inspect
    sig = inspect.signature(func)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(*args, **allowed)

try:
    from tenacity import RetryError as _TenacityRetryError
except Exception:  # if tenacity not installed
    class _TenacityRetryError(Exception): ...

def _html_error(note: str, status: int = 502) -> HTMLResponse:
    from html import escape
    block = f'<pre class="muted" style="white-space:pre-wrap">{escape(note)}</pre>'
    return HTMLResponse(render(REPORT_FORM + block, active="report"), status_code=status)

# ---------- Fundamentals normalizer + FMP fallbacks ----------
def _pick(d, *names):
    for n in names:
        if isinstance(d, dict) and (n in d) and (d[n] is not None):
            return d[n]
    return None

def _normalize_fundamentals_from_model(model):
    """Map whatever the model returns to the names the composer expects."""
    src = model if isinstance(model, dict) else getattr(model, "__dict__", {})
    cf = (src.get("core_financials") or src.get("fundamentals") or {}) if isinstance(src, dict) else {}
    out = {
        "revenue":      _pick(cf, "revenue", "ttm_revenue", "revenue_ttm", "total_revenue"),
        "ebit":         _pick(cf, "ebit", "operating_income", "operatingIncome"),
        "net_income":   _pick(cf, "net_income", "netIncome", "netIncomeTTM"),
        "fcf":          _pick(cf, "fcf", "free_cash_flow", "freeCashFlow", "freeCashFlowTTM"),
        "gross_margin": _pick(cf, "gross_margin", "grossMarginTTM", "gross_margin_ttm"),
        "op_margin":    _pick(cf, "op_margin", "operatingMarginTTM", "operating_margin_ttm"),
        "fcf_margin":   _pick(cf, "fcf_margin"),
        "roic":         _pick(cf, "roic", "roicTTM"),
        "de_ratio":     _pick(cf, "de_ratio", "debtToEquityTTM", "debt_equity"),
    }
    return out if any(v is not None for v in out.values()) else None

def _ttm_from_fmp_quarters(arr, key, n=4):
    try:
        s, seen = 0.0, 0
        for row in (arr or [])[:n]:
            v = row.get(key)
            if v is None:
                continue
            s += float(v)
            seen += 1
        return s if seen else None
    except Exception:
        return None

def fundamentals_fallback_fmp(ticker: str):
    """Lightweight TTM fallback using FMP (sum of last 4 quarters)."""
    import requests
    key = os.getenv("FMP_API_KEY")
    if not key:
        return {}

    base = "https://financialmodelingprep.com/api/v3"
    s = requests.Session()
    s.headers.update({"User-Agent": "equity-ui/1.0"})

    try:
        inc = s.get(f"{base}/income-statement/{ticker}",
                    params={"period": "quarter", "limit": 4, "apikey": key},
                    timeout=30).json()
        cfs = s.get(f"{base}/cash-flow-statement/{ticker}",
                    params={"period": "quarter", "limit": 4, "apikey": key},
                    timeout=30).json()
    except Exception:
        return {}

    rev = _ttm_from_fmp_quarters(inc, "revenue")
    opi = _ttm_from_fmp_quarters(inc, "operatingIncome")   # EBIT proxy
    ni  = _ttm_from_fmp_quarters(inc, "netIncome")
    gp  = _ttm_from_fmp_quarters(inc, "grossProfit")
    fcf = _ttm_from_fmp_quarters(cfs, "freeCashFlow")

    gm = (gp / rev) if (gp is not None and rev not in (None, 0)) else None
    om = (opi / rev) if (opi is not None and rev not in (None, 0)) else None
    fm = (fcf / rev) if (fcf is not None and rev not in (None, 0)) else None

    return {
        "revenue": rev,
        "ebit": opi,
        "net_income": ni,
        "fcf": fcf,
        "gross_margin": gm,
        "op_margin": om,
        "fcf_margin": fm,
    }

# ---------- FX + USD market cap normalization ----------
def _fx_rate_fmp(frm: str, to: str, session=None):
    """Return frm->to rate using FMP; tries /convert then /fx/{pair}."""
    import requests
    key = os.getenv("FMP_API_KEY")
    base = "https://financialmodelingprep.com/api/v3"
    s = session or requests.Session()

    # 1) /convert
    try:
        r = s.get(f"{base}/convert", params={"from": frm, "to": to, "amount": 1, "apikey": key}, timeout=20)
        j = r.json()
        if isinstance(j, dict):
            for k in ("result", "price", "conversion_result"):
                v = j.get(k)
                if v is not None:
                    return float(v)
    except Exception:
        pass

    # 2) /fx/{pair}
    for pair, inv in ((f"{frm}{to}", 1.0), (f"{to}{frm}", -1.0)):
        try:
            j = s.get(f"{base}/fx/{pair}", params={"apikey": key}, timeout=20).json()
            if isinstance(j, list) and j:
                v = j[0].get("price") or j[0].get("bid") or j[0].get("ask")
                if v is not None:
                    rate = float(v)
                    return rate if inv > 0 else (1.0 / rate)
        except Exception:
            pass
    return None

def _market_cap_usd_fmp(ticker: str):
    """Compute MC in USD = price * sharesOutstanding * FX(cur->USD)."""
    import requests
    key = os.getenv("FMP_API_KEY")
    if not key:
        return None
    base = "https://financialmodelingprep.com/api/v3"
    s = requests.Session()
    s.headers.update({"User-Agent": "equity-ui/1.0"})

    try:
        prof = s.get(f"{base}/profile/{ticker}", params={"apikey": key}, timeout=20).json() or []
        q    = s.get(f"{base}/quote/{ticker}",   params={"apikey": key}, timeout=20).json() or []
        p0, q0 = (prof[0] if prof else {}), (q[0] if q else {})
        price  = q0.get("price")
        shares = p0.get("sharesOutstanding")
        native_mc = (float(price) * float(shares)) if (price is not None and shares is not None) else (q0.get("marketCap") or p0.get("mktCap"))
        if native_mc is None:
            return None
        cur = (p0.get("currency") or "USD").upper()
        if cur == "USD":
            return float(native_mc)
        rate = _fx_rate_fmp(cur, "USD", session=s)
        return float(native_mc) * float(rate) if rate else float(native_mc)
    except Exception:
        return None

# ---------- FMP filings fallback (citations if vector DB empty) ----------
def _fmp_recent_filings(ticker: str, limit: int = 5, forms=None):
    """Return recent filings (form, date, link) using FMP; forms=['10-K','10-Q'] to filter."""
    import requests
    key = os.getenv("FMP_API_KEY")
    if not key:
        return []
    base = "https://financialmodelingprep.com/api/v3"
    s = requests.Session()
    s.headers.update({"User-Agent": "equity-ui/1.0"})
    try:
        j = s.get(f"{base}/sec_filings/{ticker}", params={"apikey": key, "limit": limit}, timeout=25).json() or []
        rows = []
        for r in j:
            form = (r.get("type") or r.get("form") or "").upper()
            if forms and form not in forms:
                continue
            rows.append({
                "form": form,
                "date": r.get("acceptedDate") or r.get("fillingDate") or r.get("date"),
                "title": r.get("title") or r.get("finalLink") or r.get("link"),
                "url": r.get("finalLink") or r.get("link")
            })
        return rows
    except Exception:
        return []

# ---------- Optional: latest quarter snapshot (YoY deltas + short AI blurb) ----------
def latest_quarter_snapshot(ticker: str):
    """Return period + simple YoY deltas; optionally add 2-sentence narrative."""
    import requests
    key = os.getenv("FMP_API_KEY")
    if not key:
        return {}
    base = "https://financialmodelingprep.com/api/v3"
    s = requests.Session(); s.headers.update({"User-Agent": "equity-ui/1.0"})

    inc = s.get(f"{base}/income-statement/{ticker}", params={"period":"quarter","limit":6,"apikey":key}, timeout=30).json() or []
    if len(inc) < 5:
        return {}

    cur = inc[0]; yr_ago = inc[4]  # same quarter prior year
    def pct(a,b):
        try:
            if b in (0, None) or a is None: return None
            return (float(a)-float(b))/float(b)
        except Exception: return None

    snap = {
        "period": cur.get("date") or cur.get("calendarYear"),
        "revenue_yoy": pct(cur.get("revenue"), yr_ago.get("revenue")),
        "eps_yoy":     pct(cur.get("eps"), yr_ago.get("eps")),
        "op_income_yoy": pct(cur.get("operatingIncome"), yr_ago.get("operatingIncome")),
        "notes": None,
    }

    # Optional short blurb if OpenAI key present
    try:
        if OPENAI_API_KEY:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            prompt = (
                f"Write 2 concise sentences on {ticker}'s latest quarter "
                f"using YoY deltas (Revenue {snap['revenue_yoy']}, EPS {snap['eps_yoy']}, "
                f"Operating income {snap['op_income_yoy']}). No hype, just facts."
            )
            blurb = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                max_tokens=80,
                temperature=0.2,
            ).choices[0].message.content.strip()
            snap["notes"] = blurb
    except Exception:
        pass

    return snap

# ---------- inline templates ----------
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
.grid{display:grid;gap:16px} @media (min-width:960px){.grid-3{grid-template-columns:320px 1fr}}
.label{font-size:12px;color:var(--muted);margin-bottom:6px}
.input,.number{width:100%;padding:10px 12px;border-radius:12px;border:1px solid var(--line);background:#fff}
.row{display:grid;gap:10px;grid-template-columns:repeat(3,1fr)}
.btn{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:12px;border:1px solid var(--ink);color:#fff;background:var(--ink);cursor:pointer}
.btn.secondary{background:#fff;color:#0a0a0a;border-color:var(--line)}
.muted{color:var(--muted)}
.cards{display:grid;gap:10px;grid-template-columns:repeat(2,1fr)} @media (min-width:960px){.cards{grid-template-columns:repeat(4,1fr)}}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:12px}
.k{font-size:12px;color:var(--muted);margin-bottom:2px} .v{font-size:18px;font-weight:600}
table{width:100%;border-collapse:collapse} th,td{padding:10px;border-bottom:1px solid var(--line);text-align:left;font-size:14px}
th{color:var(--muted);font-weight:600} .pill{display:inline-block;padding:2px 8px;border:1px solid var(--line);border-radius:999px;font-size:12px;color:var(--muted)}
.md{line-height:1.6} .md h1,.md h2,.md h3{margin-top:1.2em}
footer{margin-top:26px;font-size:12px;color:var(--muted)}
</style></head><body>
  <div class="container">
    <header>
      <div class="brand">Equity Research</div>
      <nav class="nav">
        <a href="/" class="{{ 'active' if active=='report' else '' }}">Report</a>
        <a href="/screens" class="{{ 'active' if active=='screens' else '' }}">Screens</a>
        <a href="/retriever" class="{{ 'active' if active=='retriever' else '' }}">Retriever</a>
      </nav>
    </header>
    {{ content | safe }}
    <footer>Base Currency: {{ base_currency }} · Minimal UI</footer>
  </div>
</body></html>
"""

REPORT_FORM = """
<div class="grid grid-3">
  <div class="panel">
    <form method="post" action="/report">
      <div class="label">Ticker</div>
      <input class="input" type="text" name="ticker" placeholder="AAPL" required />
      <div class="label" style="margin-top:10px;">As of (YYYY-MM-DD, optional)</div>
      <input class="input" type="text" name="as_of" placeholder="" />
      <div class="row" style="margin-top:10px;">
        <div><div class="label">Risk-free</div><input class="number" type="number" step="0.0001" name="rf" value="0.045" /></div>
        <div><div class="label">MRP</div><input class="number" type="number" step="0.0001" name="mrp" value="0.055" /></div>
        <div><div class="label">Debt Cost</div><input class="number" type="number" step="0.0001" name="kd" value="0.0500" /></div>
      </div>
      <label style="display:flex;align-items:center;gap:8px;margin-top:10px;">
        <input type="checkbox" name="include_citations" />
        <span class="muted">Include citations (needs embeddings + filings)</span>
      </label>
      <div style="margin-top:12px;"><button class="btn" type="submit">Build report</button></div>
    </form>
  </div>
  <div class="panel" id="hero">
    <div class="muted">Welcome</div>
    <h2 style="margin:.2rem 0 1rem 0;">Quiet, deliberate analysis</h2>
    <p class="muted">Generate a one-page research note with fundamentals, a peer-beta WACC and a DCF. Add citations if you have EDGAR chunks in your vector store.</p>
    <div style="margin-top:12px;"><span class="pill">FMP</span> <span class="pill">SEC/EDGAR</span> <span class="pill">Qdrant</span> <span class="pill">OpenAI</span></div>
  </div>
</div>
"""

REPORT_RESULT = """
<div class="grid" style="margin-top:16px;">
  <div class="cards">
    <div class="card"><div class="k">Market Cap</div><div class="v">{{ kpis.market_cap }}</div></div>
    <div class="card"><div class="k">Enterprise Value</div><div class="v">{{ kpis.ev }}</div></div>
    <div class="card"><div class="k">P/E</div><div class="v">{{ kpis.pe }}</div></div>
    <div class="card"><div class="k">FCF Yield</div><div class="v">{{ kpis.fcf_yield }}</div></div>
  </div>
  <div class="panel md" style="margin-top:12px;">
    {{ report_html | safe }}
    <div style="margin-top:12px; display:flex; gap:8px;">
      <a class="btn secondary" href="/report.md?ticker={{ ticker }}&as_of={{ as_of or '' }}&rf={{ rf }}&mrp={{ mrp }}&kd={{ kd }}&cit={{ '1' if include_citations else '0' }}">Download Markdown</a>
      <a class="btn secondary" href="/report_plain?ticker={{ ticker }}&as_of={{ as_of or '' }}&rf={{ rf }}&mrp={{ mrp }}&kd={{ kd }}&cit={{ '1' if include_citations else '0' }}">Plain text</a>
    </div>
  </div>
</div>
"""

SCREENS_PAGE = """
<div class="panel">
  <form method="post" action="/screens">
    <div class="row">
      <div><div class="label">Base Ticker</div><input class="input" type="text" name="base_ticker" placeholder="AAPL" required /></div>
      <div><div class="label">Min Market Cap (USD)</div><input class="number" type="number" name="size_min" placeholder="10000000000" /></div>
      <div><div class="label">As of (optional)</div><input class="input" type="text" name="as_of" placeholder="" /></div>
    </div>
    <div style="margin-top:12px;"><button class="btn" type="submit">Run screen</button></div>
  </form>
</div>
{% if rows %}
<div class="panel" style="margin-top:12px;">
  <div class="muted" style="margin-bottom:8px;">Top 20 (by composite value score)</div>
  <table>
    <thead><tr><th>Rank</th><th>Ticker</th><th>P/E</th><th>P/S</th><th>EV/EBITDA</th><th>FCF Yield</th><th>Market Cap</th></tr></thead>
    <tbody>
    {% for r in rows[:20] %}
      <tr>
        <td>{{ r.value_rank }}</td>
        <td>{{ r.ticker }}</td>
        <td>{{ '%.2f'|format(r.pe) if r.pe is not none else '—' }}</td>
        <td>{{ '%.2f'|format(r.ps) if r.ps is not none else '—' }}</td>
        <td>{{ '%.2f'|format(r.ev_ebitda) if r.ev_ebitda is not none else '—' }}</td>
        <td>{{ '%.2f%%'|format((r.fcf_yield or 0)*100) if r.fcf_yield is not none else '—' }}</td>
        <td>{{ '{:,.0f}'.format(r.market_cap) if r.market_cap else '—' }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>
{% endif %}
"""

RETRIEVER_PAGE = """
<div class="panel">
  <form method="post" action="/retriever">
    <div class="label">Query</div>
    <input class="input" type="text" name="query" placeholder="risk factors supply chain disruptions" required />
    <div class="label" style="margin-top:10px;">Tickers (optional, comma-separated)</div>
    <input class="input" type="text" name="tickers" placeholder="AAPL,MSFT" />
    <div style="margin-top:12px;">
      <button class="btn" type="submit">Search filings</button>
      <span class="muted" style="margin-left:8px;">Embeddings required: {{ 'Yes' if has_openai else 'No' }}</span>
    </div>
  </form>
</div>
{% if hits %}
<div class="panel" style="margin-top:12px;">
  <div class="muted" style="margin-bottom:8px;">Top results</div>
  <table>
    <thead><tr><th>Score</th><th>Ticker</th><th>Type</th><th>Date</th><th>Section</th><th>Link</th></tr></thead>
    <tbody>
    {% for h in hits %}
      <tr>
        <td>{{ '%.3f'|format(h.score) }}</td>
        <td>{{ h.payload.ticker or '—' }}</td>
        <td>{{ h.payload.source_type or '—' }}</td>
        <td>{{ h.payload.filing_date or '—' }}</td>
        <td>{{ h.payload.section or '—' }}</td>
        <td>{% if h.payload.source_url %}<a href="{{ h.payload.source_url }}" target="_blank">Open</a>{% else %}—{% endif %}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>
{% endif %}
"""

env = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html"]))
def render(page: str, **kw) -> str:
    tpl = env.from_string(BASE_HTML)
    content_tpl = env.from_string(page)
    return tpl.render(content=content_tpl.render(**kw), active=kw.get("active", "report"), base_currency=BASE_CURRENCY)

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(render(REPORT_FORM, active="report"))

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

    # Build model with readable errors
    try:
        model = build_model(ticker, force_refresh=False)
    except Exception as e:
        if isinstance(e, _TenacityRetryError) and hasattr(e, "last_attempt"):
            root = e.last_attempt.exception()
            return _html_error(f"RetryError -> {type(root).__name__}: {root}")
        return _html_error(f"{type(e).__name__}: {e}")

    if isinstance(model, dict) and "error" in model:
        return _html_error(f"Model error: {model['error']}", status=400)

    # comps / wacc / metrics (compat-safe)
    try:
        comps = _compat_call(comps_table, ticker, as_of=as_of, max_peers=25)
        peers = [r["ticker"] for r in (comps or {}).get("peers", [])[1:]]
        w = _compat_call(peer_beta_wacc, ticker, peers, rf=rf, mrp=mrp, kd=kd, target_d_e=None, as_of=as_of)
        lm = _compat_call(latest_metrics, ticker, as_of=as_of)
    except Exception as e:
        return _html_error(f"Comps/WACC/metrics error: {type(e).__name__}: {e}")

    # --- normalize peer market caps to USD for all rows ---
    try:
        rows = (comps or {}).get("peers", [])
        for r in rows:
            usd = _market_cap_usd_fmp(r.get("ticker"))
            if usd:
                cur_mc = r.get("market_cap")
                # update if missing or differs materially (>25%)
                if (cur_mc is None) or (abs(usd - cur_mc) / max(usd, cur_mc) > 0.25):
                    r["market_cap"] = usd
        comps["peers"] = rows
    except Exception:
        pass

    # Optional citations via vector DB; fallback to FMP filings if empty
    citations: List[dict] = []
    if include_citations and OPENAI_API_KEY:
        try:
            from openai import OpenAI
            from src.services.vector_service import search as vec_search
            from src.services.ranking_service import final_score
            client = OpenAI(api_key=OPENAI_API_KEY)
            emb = client.embeddings.create(model=EMBED_MODEL, input=f"{ticker} latest 10-K 10-Q risk factors management discussion").data[0].embedding
            raw_hits = vec_search(emb, k=8, tickers=[ticker.upper()]) or []
            rescored = []
            for h in raw_hits:
                p = h.get("payload", {}) or {}
                s = final_score(h.get("score", 0.0), p.get("filing_date"), p.get("source_type"),
                                [p.get("ticker")] if p.get("ticker") else None, [ticker.upper()])
                rescored.append((s, p))
            rescored.sort(key=lambda x: x[0], reverse=True)
            for _, p in rescored[:5]:
                citations.append({
                    "title": p.get("section", "document"),
                    "source_type": p.get("source_type"),
                    "date": p.get("filing_date"),
                    "url": p.get("source_url"),
                })
        except Exception:
            pass

    # Fallback to FMP filing links if no embedding hits
    if include_citations and not citations:
        for f in _fmp_recent_filings(ticker, limit=5, forms=["10-K", "10-Q"]):
            citations.append({
                "title": f"{f['form']} {f['date']}",
                "source_type": "SEC via FMP",
                "date": f["date"],
                "url": f["url"],
            })

    # Fundamentals (normalize or fallback to FMP)
    fundamentals = _normalize_fundamentals_from_model(model) or fundamentals_fallback_fmp(ticker)

    # Optional latest quarter snapshot
    quarter = latest_quarter_snapshot(ticker)

    # AI Enhanced Analysis (properly integrated here)
    ai_enhanced_analysis = None
    if OPENAI_API_KEY and include_citations:  # Reuse the citations checkbox for AI
        try:
            from src.services.ai.financial_analyst import ai_financial_analyst
            financial_data = {
                "fundamentals": fundamentals or {},
                "dcf_valuation": model.get("dcf_valuation") if isinstance(model, dict) else getattr(model, "dcf_valuation", None),
                "comps": comps
            }
            ai_enhanced_analysis = ai_financial_analyst.analyze_company(ticker, financial_data)
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")

    # Compose + KPIs
    try:
        md_text = compose_report({
            "symbol": ticker.upper(),
            "as_of": as_of or "latest",
            "call": ai_enhanced_analysis.analyst_rating if ai_enhanced_analysis else "Review",
            "conviction": 7.0,
            "target_low": f"${ai_enhanced_analysis.price_target_range['low']:,.0f}" if ai_enhanced_analysis else "—",
            "target_high": f"${ai_enhanced_analysis.price_target_range['high']:,.0f}" if ai_enhanced_analysis else "—",
            "base_currency": BASE_CURRENCY,
            "fundamentals": fundamentals or {},
            "dcf": (model.get("dcf_valuation") if isinstance(model, dict) else getattr(model, "dcf_valuation", None)),
            "valuation": {"wacc": (w or {}).get("wacc")},
            "comps": {"peers": (comps or {}).get("peers", [])},
            "citations": citations,
            "quarter": quarter or {},
            "ai_analysis": ai_enhanced_analysis,  # Include AI analysis
            "artifact_id": "ui-session",
        })
        report_html = md.markdown(md_text, extensions=["tables"])
    except Exception as e:
        return _html_error(f"Compose error: {type(e).__name__}: {e}")

    def money(x): return f"${x:,.0f}" if isinstance(x, (int, float)) and x is not None else "—"
    def pct(x): return f"{x*100:.2f}%" if isinstance(x, (int, float)) and x is not None else "—"
    kpis = {
        "market_cap": money((lm or {}).get("market_cap")),
        "ev": money((lm or {}).get("enterprise_value")),
        "pe": f"{(lm or {}).get('pe'):.2f}" if (lm or {}).get("pe") is not None else "—",
        "fcf_yield": pct((lm or {}).get("fcf_yield")),
    }

    return HTMLResponse(
        render(
            REPORT_RESULT,
            active="report",
            report_html=report_html,
            ticker=ticker.upper(),
            as_of=as_of,
            rf=rf,
            mrp=mrp,
            kd=kd,
            include_citations=bool(include_citations),
            kpis=kpis,
        )
    )

# Plain-text and md aliases (useful for debugging)
@app.get("/report_plain", response_class=PlainTextResponse)
@app.get("/report.md",    response_class=PlainTextResponse)
def download_report_md(
    ticker: str,
    as_of: Optional[str] = None,
    rf: float = 0.045,
    mrp: float = 0.055,
    kd: float = 0.05,
    cit: str = "0",
):
    from src.services.financial_modeler import build_model
    from src.services.comps.engine import comps_table
    from src.services.wacc.peer_beta import peer_beta_wacc
    from reports.composer import compose as compose_report

    try:
        model = build_model(ticker, force_refresh=False)
    except Exception as e:
        if isinstance(e, _TenacityRetryError) and hasattr(e, "last_attempt"):
            root = e.last_attempt.exception()
            return PlainTextResponse(f"Error: RetryError -> {type(root).__name__}: {root}", status_code=502)
        return PlainTextResponse(f"Error: {type(e).__name__}: {e}", status_code=502)

    if isinstance(model, dict) and "error" in model:
        return PlainTextResponse(f"Error: {model['error']}", status_code=400)

    comps = _compat_call(comps_table, ticker, as_of=as_of, max_peers=25)

    # Apply same market cap USD normalization
    try:
        rows = (comps or {}).get("peers", [])
        for r in rows:
            usd = _market_cap_usd_fmp(r.get("ticker"))
            if usd:
                cur_mc = r.get("market_cap")
                if (cur_mc is None) or (abs(usd - cur_mc) / max(usd, cur_mc) > 0.25):
                    r["market_cap"] = usd
        comps["peers"] = rows
    except Exception:
        pass

    peers = [r["ticker"] for r in (comps or {}).get("peers", [])[1:]]
    w = _compat_call(peer_beta_wacc, ticker, peers, rf=rf, mrp=mrp, kd=kd, target_d_e=None, as_of=as_of)

    # Citations
    citations: List[dict] = []
    if cit == "1" and OPENAI_API_KEY:
        try:
            from openai import OpenAI
            from src.services.vector_service import search as vec_search
            from src.services.ranking_service import final_score
            client = OpenAI(api_key=OPENAI_API_KEY)
            emb = client.embeddings.create(model=EMBED_MODEL, input=f"{ticker} latest 10-K 10-Q risk factors management discussion").data[0].embedding
            raw_hits = vec_search(emb, k=8, tickers=[ticker.upper()]) or []
            rescored = []
            for h in raw_hits:
                p = h.get("payload", {}) or {}
                s = final_score(h.get("score", 0.0), p.get("filing_date"), p.get("source_type"),
                                [p.get("ticker")] if p.get("ticker") else None, [ticker.upper()])
                rescored.append((s, p))
            rescored.sort(key=lambda x: x[0], reverse=True)
            for _, p in rescored[:5]:
                citations.append({
                    "title": p.get("section", "document"),
                    "source_type": p.get("source_type"),
                    "date": p.get("filing_date"),
                    "url": p.get("source_url"),
                })
        except Exception:
            pass

    if (cit == "1") and not citations:
        for f in _fmp_recent_filings(ticker, limit=5, forms=["10-K", "10-Q"]):
            citations.append({
                "title": f"{f['form']} {f['date']}",
                "source_type": "SEC via FMP",
                "date": f["date"],
                "url": f["url"],
            })

    fundamentals = _normalize_fundamentals_from_model(model) or fundamentals_fallback_fmp(ticker)
    quarter = latest_quarter_snapshot(ticker)

    # AI analysis for plain text version too
    ai_enhanced_analysis = None
    if cit == "1" and OPENAI_API_KEY:
        try:
            from src.services.ai.financial_analyst import ai_financial_analyst
            financial_data = {
                "fundamentals": fundamentals or {},
                "dcf_valuation": model.get("dcf_valuation") if isinstance(model, dict) else getattr(model, "dcf_valuation", None),
                "comps": comps
            }
            ai_enhanced_analysis = ai_financial_analyst.analyze_company(ticker, financial_data)
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")

    md_text = compose_report({
        "symbol": ticker.upper(),
        "as_of": as_of or "latest",
        "call": ai_enhanced_analysis.analyst_rating if ai_enhanced_analysis else "Review",
        "conviction": 7.0,
        "target_low": f"${ai_enhanced_analysis.price_target_range['low']:,.0f}" if ai_enhanced_analysis else "—",
        "target_high": f"${ai_enhanced_analysis.price_target_range['high']:,.0f}" if ai_enhanced_analysis else "—",
        "base_currency": BASE_CURRENCY,
        "fundamentals": fundamentals or {},
        "dcf": (model.get("dcf_valuation") if isinstance(model, dict) else getattr(model, "dcf_valuation", None)),
        "valuation": {"wacc": (w or {}).get("wacc")},
        "comps": {"peers": (comps or {}).get("peers", [])},
        "citations": citations,
        "quarter": quarter or {},
        "ai_analysis": ai_enhanced_analysis,
        "artifact_id": "ui-session",
    })
    return PlainTextResponse(md_text)

@app.get("/screens", response_class=HTMLResponse)
def screens_get():
    return HTMLResponse(render(SCREENS_PAGE, active="screens", rows=[]))

@app.post("/screens", response_class=HTMLResponse)
def screens_post(
    base_ticker: str = Form(...),
    size_min: Optional[float] = Form(None),
    as_of: Optional[str] = Form(None),
):
    from src.services.comps.engine import comps_table
    data = _compat_call(comps_table, base_ticker, as_of=as_of, max_peers=50)
    rows = (data or {}).get("peers", [])
    if size_min is not None:
        rows = [r for r in rows if (r.get("market_cap") or 0) >= size_min]
    return HTMLResponse(render(SCREENS_PAGE, active="screens", rows=rows))

@app.get("/retriever", response_class=HTMLResponse)
def retriever_get():
    return HTMLResponse(render(RETRIEVER_PAGE, active="retriever", hits=[], has_openai=bool(OPENAI_API_KEY)))

@app.post("/retriever", response_class=HTMLResponse)
def retriever_post(query: str = Form(...), tickers: Optional[str] = Form(None)):
    if not OPENAI_API_KEY:
        note = '<p class="muted" style="margin-top:10px;">OpenAI API key not set; cannot embed query.</p>'
        return HTMLResponse(render(RETRIEVER_PAGE + note, active="retriever", hits=[], has_openai=False), status_code=400)

    try:
        from openai import OpenAI
        from src.services.vector_service import search as vec_search
        from src.services.ranking_service import final_score

        client = OpenAI(api_key=OPENAI_API_KEY)
        emb = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
        tickers_list = [t.strip().upper() for t in tickers.split(",")] if tickers else None
        raw_hits = vec_search(emb, k=12, tickers=tickers_list) or []
        rescored = []
        for h in raw_hits:
            p = h.get("payload", {}) or {}
            s = final_score(
                h.get("score", 0.0),
                p.get("filing_date"),
                p.get("source_type"),
                [p.get("ticker")] if p.get("ticker") else None,
                tickers_list,
            )
            rescored.append({"score": s, "payload": p})
        rescored.sort(key=lambda x: x["score"], reverse=True)
        return HTMLResponse(render(RETRIEVER_PAGE, active="retriever", hits=rescored[:8], has_openai=True))
    except Exception as e:
        note = f'<p class="muted" style="margin-top:10px;">Error: {str(e)}</p>'
        return HTMLResponse(render(RETRIEVER_PAGE + note, active="retriever", hits=[], has_openai=True), status_code=500)

# ---------- debug ----------
@app.get("/debug/env", response_class=PlainTextResponse)
def debug_env():
    keys = [
        ("FMP_API_KEY", bool(os.getenv("FMP_API_KEY"))),
        ("OPENAI_API_KEY", bool(os.getenv("OPENAI_API_KEY"))),
        ("REDIS_URL", bool(os.getenv("REDIS_URL"))),
        ("BASE_CURRENCY", os.getenv("BASE_CURRENCY", "")),
    ]
    lines = [f"{k}={'SET' if v else 'MISSING'}" if isinstance(v, bool) else f"{k}={v}" for k, v in keys]
    return PlainTextResponse("\n".join(lines))

@app.get("/debug/fmp2", response_class=PlainTextResponse)
def debug_fmp2(ticker: str = "AAPL"):
    import socket, requests
    key = os.getenv("FMP_API_KEY")
    if not key:
        return PlainTextResponse("FMP_API_KEY missing", status_code=500)
    try:
        ip = socket.gethostbyname("financialmodelingprep.com")
        dns_line = f"DNS OK -> financialmodelingprep.com -> {ip}"
    except Exception as e:
        return PlainTextResponse(f"DNS ERROR: {repr(e)}", status_code=500)

    url = "https://financialmodelingprep.com/api/v3/quote-short/" + ticker
    try:
        r = requests.get(url, params={"apikey": key}, timeout=30, headers={"User-Agent":"equity-ui/1.0"})
        return PlainTextResponse(f"{dns_line}\nHTTP status={r.status_code}\nbody={r.text[:500]}", status_code=(200 if r.ok else 502))
    except Exception as e:
        return PlainTextResponse(f"{dns_line}\nREQUEST ERROR: {repr(e)}", status_code=500)

@app.get("/debug/mc", response_class=PlainTextResponse)
def debug_mc(ticker: str):
    usd = _market_cap_usd_fmp(ticker)
    return PlainTextResponse(f"{ticker} market_cap_usd={usd:,.0f}" if usd else f"{ticker} failed")
