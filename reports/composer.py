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
    Build a simple markdown research note. You can pass fields either via the
    `data` dict or as keyword args.

    Expected (optional) keys:
      - highlights: List[str]
      - momentum:  Dict with keys m5d, m3m, m6m, breadth50, breadth200
      - valuation: str or markdown
      - transcripts: str or markdown (e.g., analyst Q&A bullets)
      - risks: List[str]
    """
    ctx: Dict[str, Any] = {}
    if data:
        ctx.update(data)
    if kwargs:
        ctx.update(kwargs)

    md: List[str] = [f"# {symbol} — Equity Research Note (as of {as_of})", ""]

    # Highlights
    highlights = ctx.get("highlights") or []
    if isinstance(highlights, list) and highlights:
        md.append("## Highlights")
        md.extend(f"- {b}" for b in highlights)
        md.append("")

    # Momentum snapshot
    momentum = ctx.get("momentum") or {}
    if isinstance(momentum, dict) and momentum:
        m5d  = _fmt_pct(momentum.get("m5d"))
        m3m  = _fmt_pct(momentum.get("m3m"))
        m6m  = _fmt_pct(momentum.get("m6m"))
        b50  = _fmt_pct(momentum.get("breadth50"), already_pct=True)
        b200 = _fmt_pct(momentum.get("breadth200"), already_pct=True)
        md.append("## Momentum Snapshot")
        md.append(f"- 5D: {m5d} · 3M: {m3m} · 6M: {m6m}")
        md.append(f"- Breadth: {b50} above 50DMA · {b200} above 200DMA")
        md.append("")

    # Valuation vs peers
    valuation = ctx.get("valuation")
    if valuation:
        md.append("## Valuation vs Peers")
        md.append(str(valuation).strip())
        md.append("")

    # Transcript Q&A notes
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

    return "\n".join(md).strip()
