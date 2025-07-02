# utils.py — 2025-06-20
from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple

# ─────────────────────────── constants ────────────────────────────
DEFAULT_ACTION_LIMIT = 3          # fallback if caller omits a limit
# ──────────────────────────────────────────────────────────────────

# ─────────────────────────── regex helpers ────────────────────────
_DONE_RE   = re.compile(r'^Action:\s*Done!?$', re.I)
_ACTION_RE = re.compile(r'''
    ^Action:\s*                     # literal “Action:”
    (\w+)\s*                        # verb  – Search / Open / Find / Done
    (?:\(\s*)?                      # optional opening parenthesis
    (?:"([^"]+?)"|([^\n)"]+?))?     # quoted OR bare argument (optional)
    \s*\)?$                         # optional “)” + EOL
''', re.I | re.X)
# ──────────────────────────────────────────────────────────────────

# ─────────────────────── pretty-print search hits ─────────────────
def format_search_results(results: List[Dict[str, Any]]) -> str:
    """Human-friendly console view of search hits."""
    if not results:
        return "No results found."
    return "\n".join(
        f"{i+1}. {r.get('title','N/A')}\n"
        f"   {r.get('href','N/A')}\n"
        f"   {r.get('body','')}\n"
        for i, r in enumerate(results)
    )

# ───────────────────────── parse a single Action ──────────────────
def parse_action(line: str) -> Tuple[str, str] | None:
    """Return (verb, arg) if `line` is a valid Action … else None."""
    if _DONE_RE.match(line):
        return ("done", "")
    m = _ACTION_RE.match(line)
    if m:
        verb, q1, q2 = m.groups()
        return (verb.lower(), (q1 or q2 or "").strip())
    return None

# ───────────────────────── parse multiple Actions ─────────────────
def parse_actions(text: str,
                  limit: int | None = None) -> List[Tuple[str, str]]:
    """
    Extract ≤ `limit` Actions (preserving order). Stops early on Done!.
    """
    if limit is None:
        limit = DEFAULT_ACTION_LIMIT
    out: List[Tuple[str, str]] = []
    for raw in text.splitlines():
        maybe = parse_action(raw.strip())
        if maybe:
            out.append(maybe)
            if len(out) >= limit or maybe[0] == "done":
                break
    return out

# ─────────────────────── sanity-check a model reply ───────────────
def validate_action_format(text: str,
                           find_allowed: bool = True,
                           limit: int | None = None) -> bool:
    """
    True ↦ block starts with “Thought:” *and* contains ≥1 executable Action.
    """
    text = text.lstrip()
    if not text.startswith("Thought:"):
        return False
    actions = parse_actions(text, limit or DEFAULT_ACTION_LIMIT)
    if not actions:
        return False
    verb, _ = actions[0]
    return find_allowed or verb != "find"