# src/services/llm/json_client.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Create a single global client (reads OPENAI_API_KEY from env)
_client: Optional[OpenAI] = None

def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

def _extract_json_str(s: str) -> Optional[str]:
    if not s:
        return None
    # 1) Try fenced code blocks first
    m = _FENCE_RE.search(s)
    if m:
        return m.group(1).strip()

    # 2) Try to locate first JSON object in the string (naive but works well)
    first = s.find("{")
    last  = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = s[first:last+1].strip()
        # quick sanity check
        if candidate.count("{") == candidate.count("}"):
            return candidate

    return None

def _force_json(content: str) -> Dict[str, Any]:
    """
    Try strict JSON; if that fails, try to extract from fences; 
    finally, return a safe fallback envelope.
    """
    if not content:
        return {"_raw": "", "_note": "empty model response"}

    # strict
    try:
        return json.loads(content)
    except Exception:
        pass

    # fenced / substring extraction
    extracted = _extract_json_str(content)
    if extracted:
        try:
            return json.loads(extracted)
        except Exception:
            pass

    # last-gasp: return as text inside an envelope so caller can still use it
    return {"_raw": content, "_note": "non-JSON model response; parsed as raw text"}

def chat_json(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    max_tokens: int = 2000,
    temperature: float = 0.2,
    enforce_json_mode: bool = True,
) -> Dict[str, Any]:
    """
    Get a JSON object from the model, robustly.

    If enforce_json_mode=True, uses response_format={'type':'json_object'} (OpenAI v1).
    Otherwise, falls back to text and post-parses.
    """
    client = _client_singleton()

    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if enforce_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    # Call the API
    resp = client.chat.completions.create(**kwargs)

    # Pull out content
    content = ""
    try:
        content = resp.choices[0].message.content or ""
    except Exception:
        content = ""

    # If JSON mode was enforced, content is supposed to be valid JSON
    # But still guard-rail parse
    return _force_json(content)
