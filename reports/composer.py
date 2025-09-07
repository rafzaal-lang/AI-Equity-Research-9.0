# reports/composer.py
# Compose a one-page equity research note (Markdown).

from __future__ import annotations
from typing import Dict, Any, List, Optional

def _money(x: Optional[float]) -> str:
    try:
        return f"${float(x):,.0f}" if x is not None else "—"
    except Exception:
        return "—"

def _num(x: Optional[float], nd=2) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"

def _pct(x: Optional[float], nd=2) -> str:
    try:
        return f"{float(x)*100:.{nd}f}%"
    except Exception:
        return "—"

def _get(d: Dict[str, Any], *keys):
    for k in keys:
        if isinstance(d, dict) and (k in d):
            return d[k]
    return None

def compose(payload: Dict[str, Any]) -> str:
    """
    Expected keys (all optional-safe):
      - symbol: str
      - as_of: str
      - base_currency: str
      - fundamentals: dict with revenue, ebit, net_income, fcf, gross_margin, op_margin, fcf_margin, roic, de_ratio
      - dcf: dict or None
      - valuation: {"wacc": float}
      - comps: {"peers": [ {ticker, pe, ps, ev_ebitda, fcf_yield, market_cap}, ... ]}
      - citations: [ {title, date, source_type, url}, ... ]
      - quarter: {period, revenue_yoy, eps_yoy, op_income_yoy, notes}
    """
    out: List[str] = []

    sym = payload.get("symbol", "—")
    as_of = payload.get("as_of", "latest")
    base_ccy = payload.get("base_currency", "USD")

    out.append(f"# {sym} — Equity Research Note\n")
    out.append(f"*As of:* `{as_of}`  ·  *Base currency:* *{base_ccy.lower()}*")
    out.append("")

    # Summary
    val = payload.get("valuation") or {}
    wacc = _get(val, "wacc")
    dcf = payload.get("dcf") or {}
    base = _get(dcf, "base")
    low  = _get(dcf, "low")
    high = _get(dcf, "high")

    out.append("## Summary")
    out.append(f"- Estimated WACC: **{_pct(wacc, 2)}**")
    out.append(f"- DCF Value (Low / Base / High): **{_money(low)} / {_money(base)} / {_money(high)}**")
    out.append("")

    # Fundamentals
    f = payload.get("fundamentals") or {}
    rows = [
        ("Revenue",        _money(_get(f, "revenue"))),
        ("EBIT",           _money(_get(f, "ebit"))),
        ("Net Income",     _money(_get(f, "net_income"))),
        ("Free Cash Flow", _money(_get(f, "fcf"))),
        ("Gross Margin",   _pct(_get(f, "gross_margin"))),
        ("Operating Margin", _pct(_get(f, "op_margin"))),
        ("FCF Margin",     _pct(_get(f, "fcf_margin"))),
        ("ROIC",           _pct(_get(f, "roic"))),
        ("Debt/Equity",    _num(_get(f, "de_ratio"))),
    ]

    out.append("## Fundamentals (TTM / latest)")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    for k, v in rows:
        out.append(f"| {k} | {v} |")
    out.append("")

    # Valuation
    out.append("## Valuation")
    out.append(f"- **WACC:** {_pct(wacc, 2)}")
    out.append(f"- **DCF Value (Low / Base / High):** {_money(low)} / {_money(base)} / {_money(high)}")
    out.append("")

    # Comps table
    comps = (payload.get("comps") or {}).get("peers") or []
    out.append("## Peers (selected)")
    out.append("")
    out.append("| Ticker | P/E | P/S | EV/EBITDA | FCF Yield | Market Cap |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for r in comps:
        out.append(
            "| {t} | {pe} | {ps} | {ev} | {fcf} | {mc} |".format(
                t=r.get("ticker", "—"),
                pe=("—" if r.get("pe") is None else _num(r.get("pe"), 2)),
                ps=("—" if r.get("ps") is None else _num(r.get("ps"), 2)),
                ev=("—" if r.get("ev_ebitda") is None else _num(r.get("ev_ebitda"), 2)),
                fcf=("—" if r.get("fcf_yield") is None else _pct(r.get("fcf_yield"), 2)),
                mc=("—" if r.get("market_cap") is None else _money(r.get("market_cap"))),
            )
        )
    out.append("")

    # Latest quarter (optional)
    q = payload.get("quarter") or {}
    if q:
        out.append("## Latest quarter")
        period = q.get("period") or "—"
        rev_yoy = q.get("revenue_yoy")
        eps_yoy = q.get("eps_yoy")
        opi_yoy = q.get("op_income_yoy")
        out.append(
            f"*Period:* `{period}`  •  *Revenue YoY:* {_pct(rev_yoy,1)}  •  "
            f"*EPS YoY:* {_pct(eps_yoy,1)}  •  *OpInc YoY:* {_pct(opi_yoy,1)}"
        )
        if q.get("notes"):
            out.append("")
            out.append(q["notes"])
        out.append("")

    # Citations (optional)
    cites = payload.get("citations") or []
    if cites:
        out.append("## Sources / Citations")
        for c in cites:
            title = c.get("title", "document")
            typ = c.get("source_type", "")
            date = c.get("date", "")
            url = c.get("url", "")
            if url:
                out.append(f"- [{title}]({url}) — {typ} {date}".strip())
            else:
                out.append(f"- {title} — {typ} {date}".strip())
        out.append("")

    return "\n".join(out)
