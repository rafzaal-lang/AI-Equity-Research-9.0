# src/services/report/composer.py
from typing import Any, Dict, Optional, List

def _fmt_pct(x: Any, already_pct: bool = False) -> str:
    try:
        if x is None:
            return "–"
        v = float(x) if already_pct else (float(x) * 100.0)
        sign = "+" if v > 0 else ""
        return f"{sign}{v:.1f}%"
    except Exception:
        return "–"

def _fmt_money(x: Any, currency: str = "USD") -> str:
    try:
        if x is None:
            return "–"
        v = float(x)
        if abs(v) >= 1e9:
            return f"{currency} {v/1e9:.1f}B"
        elif abs(v) >= 1e6:
            return f"{currency} {v/1e6:.1f}M"
        elif abs(v) >= 1e3:
            return f"{currency} {v/1e3:.1f}K"
        else:
            return f"{currency} {v:,.0f}"
    except Exception:
        return "–"

def compose(symbol: str, as_of: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> str:
    """
    Build a comprehensive markdown research note from inputs.

    Expected keys in `data` or kwargs:
      - call: str (Buy/Hold/Sell)
      - conviction: float (1-10 scale)
      - target_low: str
      - target_high: str
      - base_currency: str
      - highlights: List[str]
      - momentum: {m5d, m3m, m6m, breadth50, breadth200}
      - fundamentals: {revenue, net_income, fcf, margins, ratios}
      - dcf: DCF valuation results
      - valuation: {wacc, etc}
      - comps: {peers: [...]}
      - quarter: {period, revenue_yoy, eps_yoy, notes}
      - citations: List[{title, source_type, date, url}]
      - ai_analysis: AI-generated analysis object
      - risks: List[str]
    """
    ctx: Dict[str, Any] = {}
    if data:   ctx.update(data)
    if kwargs: ctx.update(kwargs)

    call = ctx.get("call", "Review")
    conviction = ctx.get("conviction", 7.0)
    target_low = ctx.get("target_low", "—")
    target_high = ctx.get("target_high", "—")
    base_currency = ctx.get("base_currency", "USD")
    
    md: List[str] = [
        f"# {symbol} — Equity Research Report",
        f"**Date:** {as_of} | **Rating:** {call} | **Conviction:** {conviction}/10",
        f"**Price Targets:** {target_low} - {target_high}",
        ""
    ]

    # Executive Summary from AI if available
    ai_analysis = ctx.get("ai_analysis")
    if ai_analysis and hasattr(ai_analysis, 'executive_summary'):
        md.append("## Executive Summary")
        md.append(ai_analysis.executive_summary)
        md.append("")
        
        # Investment Thesis
        if hasattr(ai_analysis, 'investment_thesis'):
            thesis = ai_analysis.investment_thesis
            md.append("### Investment Thesis")
            if isinstance(thesis, dict):
                if thesis.get("bull_case"):
                    md.append(f"**Bull Case:** {thesis['bull_case']}")
                if thesis.get("bear_case"):
                    md.append(f"**Bear Case:** {thesis['bear_case']}")
            md.append("")

    # Latest Quarter Performance
    quarter = ctx.get("quarter") or {}
    if quarter and quarter.get("period"):
        md.append("## Latest Quarter Highlights")
        md.append(f"**Period:** {quarter['period']}")
        
        if quarter.get("revenue_yoy") is not None:
            md.append(f"- Revenue YoY: {_fmt_pct(quarter['revenue_yoy'])}")
        if quarter.get("eps_yoy") is not None:
            md.append(f"- EPS YoY: {_fmt_pct(quarter['eps_yoy'])}")
        if quarter.get("op_income_yoy") is not None:
            md.append(f"- Operating Income YoY: {_fmt_pct(quarter['op_income_yoy'])}")
            
        if quarter.get("notes"):
            md.append(f"\n*{quarter['notes']}*")
        md.append("")

    # Financial Overview
    fundamentals = ctx.get("fundamentals") or {}
    if fundamentals:
        md.append("## Financial Overview")
        
        # Core metrics
        if fundamentals.get("revenue"):
            md.append(f"- **Revenue:** {_fmt_money(fundamentals['revenue'], base_currency)}")
        if fundamentals.get("net_income"):
            md.append(f"- **Net Income:** {_fmt_money(fundamentals['net_income'], base_currency)}")
        if fundamentals.get("fcf"):
            md.append(f"- **Free Cash Flow:** {_fmt_money(fundamentals['fcf'], base_currency)}")
            
        # Margins
        margins = []
        if fundamentals.get("gross_margin"):
            margins.append(f"Gross: {_fmt_pct(fundamentals['gross_margin'])}")
        if fundamentals.get("op_margin"):
            margins.append(f"Operating: {_fmt_pct(fundamentals['op_margin'])}")
        if margins:
            md.append(f"- **Margins:** {' | '.join(margins)}")
            
        # Returns
        returns = []
        if fundamentals.get("roic"):
            returns.append(f"ROIC: {_fmt_pct(fundamentals['roic'])}")
        if fundamentals.get("de_ratio"):
            returns.append(f"D/E: {fundamentals['de_ratio']:.1f}x")
        if returns:
            md.append(f"- **Key Ratios:** {' | '.join(returns)}")
        md.append("")

    # Valuation Analysis
    dcf = ctx.get("dcf") or {}
    valuation = ctx.get("valuation") or {}
    if dcf or valuation:
        md.append("## Valuation Analysis")
        
        if dcf.get("enterprise_value"):
            md.append(f"- **Enterprise Value:** {_fmt_money(dcf['enterprise_value'], base_currency)}")
        if dcf.get("fair_value_per_share"):
            md.append(f"- **DCF Fair Value:** {base_currency} {dcf['fair_value_per_share']:.2f} per share")
            
        # DCF assumptions
        assumptions = dcf.get("assumptions") or {}
        if assumptions:
            wacc = assumptions.get("wacc") or valuation.get("wacc")
            if wacc:
                md.append(f"- **WACC:** {_fmt_pct(wacc)}")
            if assumptions.get("g1"):
                md.append(f"- **Growth Assumptions:** Stage 1: {_fmt_pct(assumptions['g1'])}, Terminal: {_fmt_pct(assumptions.get('g_terminal', 0.025))}")
        md.append("")

    # Peer Comparison
    comps = ctx.get("comps") or {}
    peers = comps.get("peers", [])
    if peers and len(peers) > 1:
        md.append("## Peer Comparison")
        md.append("| Company | P/E | P/S | EV/EBITDA | Market Cap |")
        md.append("|---------|-----|-----|-----------|------------|")
        
        for peer in peers[:6]:  # Show top 6 peers
            ticker_name = peer.get("ticker", "—")
            pe = f"{peer['pe']:.1f}" if peer.get("pe") else "—"
            ps = f"{peer['ps']:.1f}" if peer.get("ps") else "—"
            ev_ebitda = f"{peer['ev_ebitda']:.1f}" if peer.get("ev_ebitda") else "—"
            mc = _fmt_money(peer.get("market_cap"), base_currency)
            md.append(f"| {ticker_name} | {pe} | {ps} | {ev_ebitda} | {mc} |")
        md.append("")

    # AI Key Insights
    if ai_analysis and hasattr(ai_analysis, 'key_insights'):
        insights = ai_analysis.key_insights
        if insights:
            md.append("## Key Investment Insights")
            
            # Group insights by category
            insights_by_category = {}
            for insight in insights:
                category = insight.category.title()
                if category not in insights_by_category:
                    insights_by_category[category] = []
                insights_by_category[category].append(insight)
            
            for category, category_insights in insights_by_category.items():
                md.append(f"### {category}s")
                for insight in category_insights:
                    confidence_emoji = "🟢" if insight.confidence > 0.8 else "🟡" if insight.confidence > 0.6 else "🟠"
                    md.append(f"- {confidence_emoji} {insight.insight}")
                md.append("")

    # Momentum & Technical
    momentum = ctx.get("momentum") or {}
    if momentum:
        md.append("## Technical & Momentum")
        
        timeframes = []
        if momentum.get("m5d") is not None:
            timeframes.append(f"5D: {_fmt_pct(momentum['m5d'])}")
        if momentum.get("m3m") is not None:
            timeframes.append(f"3M: {_fmt_pct(momentum['m3m'])}")
        if momentum.get("m6m") is not None:
            timeframes.append(f"6M: {_fmt_pct(momentum['m6m'])}")
        
        if timeframes:
            md.append(f"- **Returns:** {' | '.join(timeframes)}")
            
        breadth = []
        if momentum.get("breadth50") is not None:
            breadth.append(f"{_fmt_pct(momentum['breadth50'], True)} above 50DMA")
        if momentum.get("breadth200") is not None:
            breadth.append(f"{_fmt_pct(momentum['breadth200'], True)} above 200DMA")
        
        if breadth:
            md.append(f"- **Breadth:** {' | '.join(breadth)}")
        md.append("")

    # Risk Factors
    risks = ctx.get("risks", [])
    # Also check AI analysis for risks
    if ai_analysis and hasattr(ai_analysis, 'key_insights'):
        ai_risks = [insight.insight for insight in ai_analysis.key_insights 
                   if insight.category in ['weakness', 'threat']]
        risks.extend(ai_risks)
    
    if risks:
        md.append("## Risk Factors")
        for risk in risks[:5]:  # Limit to top 5 risks
            md.append(f"- {risk}")
        md.append("")

    # Citations/Sources
    citations = ctx.get("citations", [])
    if citations:
        md.append("## Sources & Citations")
        for i, citation in enumerate(citations[:5], 1):
            title = citation.get("title", "Document")
            source_type = citation.get("source_type", "")
            date = citation.get("date", "")
            url = citation.get("url", "")
            
            if url:
                md.append(f"{i}. [{title}]({url}) - {source_type} ({date})")
            else:
                md.append(f"{i}. {title} - {source_type} ({date})")
        md.append("")

    # Disclaimer
    md.append("---")
    md.append("**Disclaimer:** This report is for informational purposes only and does not constitute investment advice. Past performance does not guarantee future results. Please consult with a qualified financial advisor before making investment decisions.")
    
    # Metadata
    artifact_id = ctx.get("artifact_id", "report")
    md.append(f"\n*Generated by AI Equity Research Platform | ID: {artifact_id}*")

    return "\n".join(md).strip()
