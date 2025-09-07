# src/services/report/composer.py
from __future__ import annotations

from typing import Any, Dict, Optional, List
from datetime import date
from dataclasses import asdict, is_dataclass

# ---------- formatting helpers ----------

def _fmt_money(x: Any, currency: str = "USD") -> str:
    try:
        if x is None:
            return "—"
        v = float(x)
        abs_v = abs(v)
        if abs_v >= 1_000_000_000_000:
            return f"${v/1_000_000_000_000:.2f}T"
        if abs_v >= 1_000_000_000:
            return f"${v/1_000_000_000:.2f}B"
        if abs_v >= 1_000_000:
            return f"${v/1_000_000:.2f}M"
        return f"${v:,.0f}"
    except Exception:
        # also allow strings like "$1.2B" to pass through
        return str(x) if isinstance(x, str) and x else "—"

def _fmt_pct(x: Any, already_pct: bool = False) -> str:
    try:
        if x is None:
            return "—"
        v = float(x) if already_pct else (float(x) * 100.0)
        sign = "+" if v > 0 else ""
        return f"{sign}{v:.1f}%"
    except Exception:
        return "—"

def _fmt_ratio(x: Any, decimals: int = 2) -> str:
    try:
        if x is None:
            return "—"
        v = float(x)
        return f"{v:.{decimals}f}"
    except Exception:
        return "—"

def _fmt_plain(x: Any) -> str:
    return str(x).strip() if x is not None else ""

def _take_n(items: List[Dict[str, Any]], n: int = 8) -> List[Dict[str, Any]]:
    return items[:n] if isinstance(items, list) else []

def _to_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    # unknown type: do not render repr; return empty
    return {}

# ---------- main composer ----------

def compose(
    symbol: str,
    as_of: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """
    Build a clean markdown research note from your payload without dumping raw dicts.

    Accepts:
      - data: dict payload, may include:
          base_currency, fundamentals, dcf, comps, citations, ai_analysis (dataclass), valuation, momentum, highlights, transcripts, risks
      - kwargs: merged into context
    """
    if not as_of:
        as_of = date.today().isoformat()

    ctx: Dict[str, Any] = {}
    if data:
        ctx.update(data)
    if kwargs:
        ctx.update(kwargs)

    currency = ctx.get("base_currency", "USD")

    # ---- Topline KPIs (if present) ----
    fundamentals = ctx.get("fundamentals") or {}
    kpi_market_cap = ctx.get("market_cap") or fundamentals.get("market_cap") or _nested(ctx, ["comps", "peers", 0, "market_cap"])
    kpi_ev = ctx.get("enterprise_value") or fundamentals.get("enterprise_value") or _nested(ctx, ["comps", "peers", 0, "enterprise_value"])
    kpi_pe = fundamentals.get("pe") or _nested(ctx, ["comps", "peers", 0, "pe"])
    kpi_fcf_yield = fundamentals.get("fcf_yield")  # expect a decimal e.g. 0.04

    # ---- DCF (if present) ----
    dcf = ctx.get("dcf") or ctx.get("dcf_valuation") or {}
    dcf_ev = dcf.get("enterprise_value")
    dcf_eq = dcf.get("equity_value")
    dcf_tv = dcf.get("terminal_value")
    dcf_tv_pct = dcf.get("terminal_value_pct")
    dcf_assump = dcf.get("assumptions") or {}
    dcf_rg = dcf_assump.get("revenue_growth")
    dcf_dr = dcf_assump.get("discount_rate")
    dcf_tg = dcf_assump.get("terminal_growth")
    dcf_years = dcf_assump.get("projection_years")

    # ---- AI analysis (dataclass CompanyAnalysis) ----
    ai_analysis = _to_dict(ctx.get("ai_analysis"))
    ai_exec = ai_analysis.get("executive_summary")
    ai_rating = ai_analysis.get("analyst_rating")
    ai_targets = ai_analysis.get("price_target_range") or {}
    ai_thesis = ai_analysis.get("investment_thesis") or {}
    ai_insights = ai_analysis.get("key_insights") or []

    # ---- Comps (compact table) ----
    peers = (ctx.get("comps") or {}).get("peers") or []
    peers_rows = []
    for row in _take_n(peers, 8):
        if not isinstance(row, dict):
            continue
        peers_rows.append({
            "Ticker": row.get("ticker"),
            "P/E": _fmt_ratio(row.get("pe")),
            "EV/EBITDA": _fmt_ratio(row.get("ev_ebitda")),
            "P/S": _fmt_ratio(row.get("ps")),
            "Mkt Cap": _fmt_money(row.get("market_cap"), currency),
        })

    # ---- Citations ----
    citations = ctx.get("citations") or []

    md: List[str] = [f"# {symbol} — Equity Research Note *(as of {as_of})*", ""]

    # Snapshot
    if any([kpi_market_cap, kpi_ev, kpi_pe, kpi_fcf_yield]):
        md.append("## Snapshot")
        md.append(f"- **Market Cap:** {_fmt_money(kpi_market_cap, currency)}")
        md.append(f"- **Enterprise Value:** {_fmt_money(kpi_ev, currency)}")
        md.append(f"- **P/E:** {_fmt_ratio(kpi_pe)}")
        md.append(f"- **FCF Yield:** {_fmt_pct(kpi_fcf_yield)}")
        md.append("")

    # Analyst Rating & Targets
    if ai_rating or ai_targets:
        md.append("## Analyst View")
        if ai_rating:
            md.append(f"- **Rating:** {ai_rating}")
        if isinstance(ai_targets, dict) and ai_targets:
            low = _fmt_money(ai_targets.get("low"), currency)
            base = _fmt_money(ai_targets.get("base"), currency)
            high = _fmt_money(ai_targets.get("high"), currency)
            md.append(f"- **Target Range (EV):** Low {low} · Base {base} · High {high}")
        md.append("")

    # Executive Summary
    if ai_exec:
        md.append("## Executive Summary")
        md.append(ai_exec.strip())
        md.append("")

    # Key Insights
    if isinstance(ai_insights, list) and ai_insights:
        md.append("## Key Insights")
        by_cat: Dict[str, List[str]] = {"strength": [], "opportunity": [], "weakness": [], "threat": []}
        for it in ai_insights:
            it_d = _to_dict(it)
            cat = str(it_d.get("category", "other")).lower()
            insight = _fmt_plain(it_d.get("insight"))
            if not insight:
                continue
            by_cat.setdefault(cat, []).append(insight)
        for cat in ["strength", "opportunity", "weakness", "threat"]:
            items = by_cat.get(cat) or []
            if items:
                md.append(f"**{cat.title()}s**")
                md.extend(f"- {s}" for s in items)
        md.append("")

    # DCF Summary
    if any([dcf_ev, dcf_eq, dcf_tv, dcf_tv_pct, dcf_assump]):
        md.append("## DCF Summary")
        if any([dcf_ev, dcf_eq]):
            md.append(f"- **Enterprise Value:** {_fmt_money(dcf_ev, currency)}  ·  **Equity Value:** {_fmt_money(dcf_eq, currency)}")
        if dcf_tv is not None or dcf_tv_pct is not None:
            md.append(f"- **Terminal Value:** {_fmt_money(dcf_tv, currency)}  ({_fmt_pct(dcf_tv_pct, already_pct=True)} of EV)")
        if any([dcf_rg, dcf_dr, dcf_tg, dcf_years]):
            md.append(f"- **Assumptions:** Rev growth {_fmt_pct(dcf_rg)} · Discount {_fmt_pct(dcf_dr)} · Terminal {_fmt_pct(dcf_tg)} · Years {_fmt_ratio(dcf_years, 0)}")
        md.append("")

    # Highlights (your existing field)
    highlights = ctx.get("highlights") or []
    if isinstance(highlights, list) and highlights:
        md.append("## Highlights")
        md.extend(f"- {b}" for b in highlights)
        md.append("")

    # Momentum (your existing field)
    momentum = ctx.get("momentum") or {}
    if isinstance(momentum, dict) and momentum:
        m5d  = _fmt_pct(momentum.get("m5d"))
        m3m  = _fmt_pct(momentum.get("m3m"))
        m6m  = _fmt_pct(momentum.get("m6m"))
        b50  = _fmt_pct(momentum.get("breadth50"), already_pct=True)
        b200 = _fmt_pct(momentum.get("breadth200"), already_pct=True)
        md.append("## Momentum Snapshot")
        md.append(f"- **5D:** {m5d} · **3M:** {m3m} · **6M:** {m6m}")
        md.append(f"- **Breadth:** {b50} above 50DMA · {b200} above 200DMA")
        md.append("")

    # Valuation note (freeform)
    valuation_note = ctx.get("valuation")
    if valuation_note:
        md.append("## Valuation Notes")
        md.append(str(valuation_note).strip())
        md.append("")

    # Peers table (compact)
    if peers_rows:
        md.append("## Peer Snapshot")
        md.append("| Ticker | P/E | EV/EBITDA | P/S | Mkt Cap |")
        md.append("|:------:|----:|----------:|----:|-------:|")
        for r in peers_rows:
            md.append(f"| {r['Ticker']} | {r['P/E']} | {r['EV/EBITDA']} | {r['P/S']} | {r['Mkt Cap']} |")
        md.append("")

    # Transcript notes (freeform)
    transcripts = ctx.get("transcripts")
    if transcripts:
        md.append("## Transcript Q&A Notes")
        md.append(str(transcripts).strip())
        md.append("")

    # Risks (list)
    risks = ctx.get("risks") or []
    if isinstance(risks, list) and risks:
        md.append("## Risks")
        md.extend(f"- {r}" for r in risks)
        md.append("")

    # Citations
    if isinstance(citations, list) and citations:
        md.append("## Citations")
        for c in citations:
            if not isinstance(c, dict):
                continue
            title = c.get("title") or "Source"
            url = c.get("url")
            date_str = c.get("date")
            src_type = c.get("source_type")
            line = f"- **{title}**"
            if src_type:
                line += f" — {src_type}"
            if date_str:
                line += f" ({date_str})"
            if url:
                line += f" — [{url}]({url})"
            md.append(line)
        md.append("")

    return "\n".join(md).strip()

# ---------- tiny helper to safely walk nested dicts ----------

def _nested(d: Dict[str, Any], path: List[Any], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        try:
            if isinstance(key, int):
                if isinstance(cur, list) and 0 <= key < len(cur):
                    cur = cur[key]
                else:
                    return default
            else:
                if isinstance(cur, dict):
                    cur = cur.get(key, default)
                else:
                    return default
        except Exception:
            return default
    return cur
