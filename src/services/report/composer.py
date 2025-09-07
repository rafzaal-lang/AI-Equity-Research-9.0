from __future__ import annotations
from typing import Dict, Any
from jinja2 import Template

TEMPLATE = Template("""# {{ symbol }} — Equity Research Note (as of {{ as_of }})
**Call:** {{ call }}
**Conviction:** {{ conviction }} / 10
**Target Range:** {{ target_low }} – {{ target_high }}

## Macro Snapshot
- Regime: **{{ macro.regime }}**
- 10Y: {{ macro.ten_year_last }} | Fed Funds: {{ macro.fed_funds_last }} | HY OAS: {{ macro.hy_oas_last }}
- CPI (last): {{ macro.cpi_last }}

## Quant Signals
{% for k,v in quant.momentum.items() %}- {{ k }}: {{ '%.2f%%' | format(v*100) if v is not none else 'n/a' }}
{% endfor -%}
- RSI(14): {{ '%.1f' | format(quant.rsi) if quant.rsi is not none else 'n/a' }}
- SMA Cross (20/50): {{ quant.sma.cross if quant.sma.cross is not none else 'n/a' }}

def compose(payload: Dict[str, Any]) -> str:
    """Enhanced report composition with AI insights."""
    out: List[str] = []

    sym = payload.get("symbol", "—")
    as_of = payload.get("as_of", "latest")
    base_ccy = payload.get("base_currency", "USD")

    out.append(f"# {sym} — AI-Enhanced Equity Research Note\n")
    out.append(f"*As of:* `{as_of}`  ·  *Base currency:* *{base_ccy.lower()}*")
    out.append("")

    # NEW: AI Executive Summary
    ai_analysis = payload.get("ai_analysis")
    if ai_analysis:
        out.append("## Executive Summary")
        out.append(f"**{ai_analysis.analyst_rating}** - {ai_analysis.executive_summary}")
        out.append("")
        
        # Investment Thesis
        out.append("### Investment Thesis")
        out.append(f"**Bull Case:** {ai_analysis.investment_thesis['bull_case']}")
        out.append("")
        out.append(f"**Bear Case:** {ai_analysis.investment_thesis['bear_case']}")
        out.append("")

    # Valuation section (existing)
    val = payload.get("valuation") or {}
    wacc = _get(val, "wacc")
    dcf = payload.get("dcf") or {}
    base = _get(dcf, "base")
    low  = _get(dcf, "low")
    high = _get(dcf, "high")

    out.append("## Valuation Summary")
    out.append(f"- Estimated WACC: **{_pct(wacc, 2)}**")
    out.append(f"- DCF Value (Low / Base / High): **{_money(low)} / {_money(base)} / {_money(high)}**")
    
    if ai_analysis:
        out.append(f"- AI Price Target: **{_money(ai_analysis.price_target_range['low'])} - {_money(ai_analysis.price_target_range['high'])}**")
    out.append("")

    # NEW: AI Key Insights
    if ai_analysis and ai_analysis.key_insights:
        out.append("## Key Investment Insights")
        out.append("")
        
        for insight in ai_analysis.key_insights:
            emoji = {"strength": "💪", "weakness": "⚠️", "opportunity": "🚀", "threat": "🔴"}.get(insight.category, "📊")
            out.append(f"**{emoji} {insight.category.title()}:** {insight.insight}")
            out.append("")

    # Fundamentals (existing code)
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

    # NEW: AI Risk Analysis
    ai_risks = payload.get("ai_risks", [])
    if ai_risks:
        out.append("## AI Risk Analysis")
        out.append("")
        
        # Group risks by severity
        high_risks = [r for r in ai_risks if r.severity == "high"]
        medium_risks = [r for r in ai_risks if r.severity == "medium"]
        
        if high_risks:
            out.append("### High Priority Risks")
            for risk in high_risks:
                out.append(f"**{risk.category.title()} Risk:** {risk.description}")
                out.append(f"*Impact:* {risk.potential_impact}")
                out.append("")
        
        if medium_risks:
            out.append("### Medium Priority Risks")
            for risk in medium_risks[:3]:  # Limit to top 3
                out.append(f"**{risk.category.title()}:** {risk.description}")
                out.append("")

    # Existing comps table and other sections...
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

    # Rest of existing code for citations, etc.
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

    out.append("---")
    out.append("*Generated by AI-Enhanced Equity Research Platform*")
    
    return "\n".join(out)
    
## Fundamentals (TTM / Latest)
- Gross Margin: {{ fundamentals.margins.gross_margin | default('n/a') }}
- Operating Margin: {{ fundamentals.margins.operating_margin | default('n/a') }}
- Net Margin: {{ fundamentals.margins.net_margin | default('n/a') }}
- FCF: {{ fundamentals.reported.free_cash_flow | default('n/a') }}

## Valuation (Quick DCF)
- EV (model): {{ dcf.enterprise_value }}
- Terminal value share: {{ '%.0f%%' | format((dcf.terminal_value_pct or 0)*100) }}

## Comps (selected peers: {{ comps.peers | length }})
| Ticker | P/E | P/S | EV/EBITDA | MktCap |
|---|---:|---:|---:|---:|
{% for r in comps.peers %}| {{ r.ticker }} | {{ r.pe | default('n/a') }} | {{ r.ps | default('n/a') }} | {{ r.ev_ebitda | default('n/a') }} | {{ r.market_cap | default('n/a') }} |
{% endfor %}

## What changed
- (auto-fill from diffs in statements/prices)

---
*Generated by Equity Research Platform*
""")

def compose(data: Dict[str, Any]) -> str:
    return TEMPLATE.render(**data)


