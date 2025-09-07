# src/services/report/composer.py
from __future__ import annotations

from typing import Any, Dict, Optional, List
from datetime import date
from dataclasses import asdict, is_dataclass
import math

# ---------- formatting helpers ----------

def _fmt_money(x: Any) -> str:
    try:
        if x is None:
            return "—"
        v = float(x)
        a = abs(v)
        if a >= 1_000_000_000_000:
            return f"${v/1_000_000_000_000:.2f}T"
        if a >= 1_000_000_000:
            return f"${v/1_000_000_000:.2f}B"
        if a >= 1_000_000:
            return f"${v/1_000_000:.2f}M"
        return f"${v:,.0f}"
    except Exception:
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

def _first(lst: Any) -> Dict[str, Any]:
    if isinstance(lst, list) and lst:
        x = lst[0]
        return x if isinstance(x, dict) else {}
    return {}

def _to_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {}

def _is_pos_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0

def _robust_peer_pe(peers: List[Dict[str, Any]], min_n: int = 3) -> Optional[float]:
    vals = [
        float(p.get("pe")) for p in peers or []
        if isinstance(p, dict) and _is_pos_num(p.get("pe")) and 0 < float(p.get("pe")) < 100
    ]
    if len(vals) < min_n:
        return None
    vals.sort()
    k1, k2 = int(0.25 * len(vals)), int(0.75 * len(vals))
    mid = vals[k1:k2] or vals
    return sum(mid) / len(mid)

# ---------- main composer ----------

def compose(
    symbol: str,
    as_of: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """
    Build a clean markdown research note from your payload.

    Accepts context keys like:
      base_currency, fundamentals, dcf, comps, citations, ai_analysis, valuation,
      momentum, highlights, transcripts, risks, quarter
    """
    if not as_of:
        as_of = date.today().isoformat()

    ctx: Dict[str, Any] = {}
    if data:
        ctx.update(data)
    if kwargs:
        ctx.update(kwargs)

    fundamentals = (ctx.get("fundamentals") or {}).copy()
    comps = (ctx.get("comps") or {}).copy()
    peers = comps.get("peers") or []

    # ---- derive KPIs / fallbacks ----
    mc = fundamentals.get("market_cap")
    if mc is None:
        mc = _first(peers).get("market_cap")

    ev = fundamentals.get("enterprise_value")
    if ev is None:
        ev = _first(peers).get("enterprise_value")

    pe = fundamentals.get("pe")
    peer_pe_robust = _robust_peer_pe(peers) if peers else None

    fcf_yield = fundamentals.get("fcf_yield")
    if fcf_yield is None:
        fcf = fundamentals.get("fcf")
        if _is_pos_num(fcf) and _is_pos_num(mc):
            fcf_yield = float(fcf) / float(mc)

    # ---- DCF bits ----
    dcf = ctx.get("dcf") or ctx.get("dcf_valuation") or {}
    dcf_ev = dcf.get("enterprise_value")
    dcf_eq = dcf.get("equity_value")
    dcf_tv = dcf.get("terminal_value")
    dcf_tv_pct = dcf.get("terminal_value_pct")
    dcf_a = dcf.get("assumptions") or {}
    dcf_rg, dcf_dr, dcf_tg, dcf_years = (
        dcf_a.get("revenue_growth"),
        dcf_a.get("discount_rate"),
        dcf_a.get("terminal_growth"),
        dcf_a.get("projection_years"),
    )

    # ---- AI analysis ----
    ai = _to_dict(ctx.get("ai_analysis"))
    ai_exec = ai.get("executive_summary")
    ai_rating = ai.get("analyst_rating")
    ai_targets = ai.get("price_target_range") or {}
    ai_insights = ai.get("key_insights") or []

    # ---- markdown ----
    md: List[str] = [f"# {symbol} — Equity Research Note *(as of {as_of})*", ""]

    # Snapshot
    if any([mc, ev, pe, fcf_yield, peer_pe_robust]):
        md.append("## Snapshot")
        if mc is not None:
            md.append(f"- **Market Cap:** {_fmt_money(mc)}")
        if ev is not None:
            md.append(f"- **Enterprise Value:** {_fmt_money(ev)}")
        if pe is not None:
            md.append(f"- **P/E (TTM):** {_fmt_ratio(pe)}")
        if peer_pe_robust is not None:
            md.append(f"- **Peer Avg P/E (robust):** {_fmt_ratio(peer_pe_robust)}")
        if fcf_yield is not None:
            md.append(f"- **FCF Yield:** {_fmt_pct(fcf_yield)}")
        md.append("")

    # Analyst View
    if ai_rating or ai_targets:
        md.append("## Analyst View")
        if ai_rating:
            md.append(f"- **Rating:** {ai_rating}")
        if isinstance(ai_targets, dict) and ai_targets:
            low = _fmt_money(ai_targets.get("low"))
            base = _fmt_money(ai_targets.get("base"))
            high = _fmt_money(ai_targets.get("high"))
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
    if any([dcf_ev, dcf_eq, dcf_tv, dcf_tv_pct, dcf_a]):
        md.append("## DCF Summary")
        if any([dcf_ev, dcf_eq]):
            md.append(f"- **Enterprise Value:** {_fmt_money(dcf_ev)}  ·  **Equity Value:** {_fmt_money(dcf_eq)}")
        if dcf_tv is not None or dcf_tv_pct is not None:
            md.append(f"- **Terminal Value:** {_fmt_money(dcf_tv)}  ({_fmt_pct(dcf_tv_pct, already_pct=True)} of EV)")
        if any([dcf_rg, dcf_dr, dcf_tg, dcf_years]):
            yrs = _fmt_ratio(dcf_years, 0) if dcf_years is not None else "—"
            md.append(f"- **Assumptions:** Rev growth {_fmt_pct(dcf_rg)} · Discount {_fmt_pct(dcf_dr)} · Terminal {_fmt_pct(dcf_tg)} · Years {yrs}")
        md.append("")

    # Quarterly Update (optional)
    q = ctx.get("quarter") or {}
    if isinstance(q, dict) and any(q.get(k) is not None for k in ("period","revenue_yoy","eps_yoy","op_income_yoy","notes")):
        md.append("## Quarterly Update")
        if q.get("period"):
            md.append(f"- **Period:** {_fmt_plain(q.get('period'))}")
        if q.get("revenue_yoy") is not None:
            md.append(f"- **Revenue YoY:** {_fmt_pct(q.get('revenue_yoy'))}")
        if q.get("eps_yoy") is not None:
            md.append(f"- **EPS YoY:** {_fmt_pct(q.get('eps_yoy'))}")
        if q.get("op_income_yoy") is not None:
            md.append(f"- **Operating Income YoY:** {_fmt_pct(q.get('op_income_yoy'))}")
        if q.get("notes"):
            md.append(f"- {_fmt_plain(q.get('notes'))}")
        md.append("")

    # Highlights
    highlights = ctx.get("highlights") or []
    if isinstance(highlights, list) and highlights:
        md.append("## Highlights")
        md.extend(f"- {b}" for b in highlights)
        md.append("")

    # Momentum
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

    # Valuation note
    valuation_note = ctx.get("valuation")
    if valuation_note:
        md.append("## Valuation Notes")
        md.append(str(valuation_note).strip())
        md.append("")

    # Peers (compact)
    rows = []
    if isinstance(peers, list) and peers:
        md.append("## Peer Snapshot")
        md.append("| Ticker | P/E | EV/EBITDA | P/S | Mkt Cap |")
        md.append("|:------:|----:|----------:|----:|-------:|")
        for p in peers[:8]:
            if not isinstance(p, dict):
                continue
            md.append(
                f"| {p.get('ticker')} | "
                f"{_fmt_ratio(p.get('pe'))} | "
                f"{_fmt_ratio(p.get('ev_ebitda'))} | "
                f"{_fmt_ratio(p.get('ps'))} | "
                f"{_fmt_money(p.get('market_cap'))} |"
            )
        md.append("")

    # Transcript notes
    transcripts = ctx.get("transcripts")
    if transcripts:
        md.append("## Transcript Q&A Notes")
        md.append(str(transcripts).strip())
        md.append("")

    # Risks
    risks = ctx.get("risks") or []
    if isinstance(risks, list) and risks:
        md.append("## Risks")
        md.extend(f"- {r}" for r in risks)
        md.append("")

    # Citations
    citations = ctx.get("citations") or []
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
