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

def compose(symbol: str, as_of: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> str:
    """
    Build a simple markdown research note from inputs.

    Optional keys in `data` or kwargs:
      - highlights: List[str]
      - momentum:  {m5d, m3m, m6m, breadth50, breadth200}
      - valuation: str (markdown)
      - transcripts: str (markdown)
      - risks: List[str]
    """
    ctx: Dict[str, Any] = {}
    if data:   ctx.update(data)
    if kwargs: ctx.update(kwargs)

    md: List[str] = [f"# {symbol} — Equity Research Note (as of {as_of})", ""]

    # Highlights
    if isinstance(ctx.get("highlights"), list) and ctx["highlights"]:
        md.append("## Highlights")
        md += [f"- {b}" for b in ctx["highlights"]]
        md.append("")

    # Momentum snapshot
    m = ctx.get("momentum") or {}
    if isinstance(m, dict) and m:
        md.append("## Momentum Snapshot")
        md.append(f"- 5D: {_fmt_pct(m.get('m5d'))} · 3M: {_fmt_pct(m.get('m3m'))} · 6M: {_fmt_pct(m.get('m6m'))}")
        md.append(f"- Breadth: {_fmt_pct(m.get('breadth50'), True)} above 50DMA · {_fmt_pct(m.get('breadth200'), True)} above 200DMA")
        md.append("")

    # Valuation vs Peers
    if ctx.get("valuation"):
        md.append("## Valuation vs Peers")
        md.append(str(ctx["valuation"]).strip())
        md.append("")

    # Transcript Q&A notes
    if ctx.get("transcripts"):
        md.append("## Transcript Q&A Notes")
        md.append(str(ctx["transcripts"]).strip())
        md.append("")

    # Risks
    if isinstance(ctx.get("risks"), list) and ctx["risks"]:
        md.append("## Risks")
        md += [f"- {r}" for r in ctx["risks"]]
        md.append("")

    return "\n".join(md).strip()
