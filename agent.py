# agent.py — 2025-07-01
"""
Web-research agent with **Crawl** and **Recall** tools + lightweight RAG.
Keeps your original streaming logic & action throttling.
"""

from __future__ import annotations

import re, textwrap, time
from typing import List, Tuple

import utils, web_search, llm_interface
from memory import Memory
import crawler            # ← new helper module

# ─────────────────────────── tunables ────────────────────────────
MAX_LOOPS            = 4
MIN_SUPPORT_SOURCES  = 1
SELF_CONSISTENCY_N   = 2
THREAD_POOL_WORKERS  = 6
MAX_HISTORY_CHARS    = 12_000
# Max Action lines the LLM may emit per turn
ACTION_LIMIT         = 4
# Max Actions actually executed before the next user turn
ACTIONS_PER_TURN     = 4
CRAWL_MAX_PAGES      = 40
# ──────────────────────────────────────────────────────────────────

_llm      = llm_interface._llm
_budget   = llm_interface._budget
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

# ─────────────────────── utility helpers ────────────────────────
def _words(t: str) -> set[str]:
    return {w.lower() for w in _TOKEN_RE.findall(t) if len(w) > 3}

def _trim_history(text: str, reserve: int = 400) -> str:
    """Trim history so that token count fits within the model context."""
    try:
        max_in = _llm.n_ctx() - reserve
        tokens = _llm.tokenize(text.encode())
    except Exception:
        return text[-MAX_HISTORY_CHARS:]
    if len(tokens) <= max_in:
        return text
    keep = tokens[-max_in:]
    try:
        trimmed = _llm.detokenize(keep).decode("utf-8", "ignore")
    except Exception:
        ratio = max_in / len(tokens)
        keep_chars = int(len(text) * ratio)
        trimmed = text[-keep_chars:]
    return trimmed

def _find_chunks(text: str, needle: str,
                 context: int = 120, max_hits: int = 3) -> list[str]:
    lowered, pos, out = text.lower(), 0, []
    needle_l = needle.lower()
    while len(out) < max_hits:
        idx = lowered.find(needle_l, pos)
        if idx == -1:
            break
        start = max(0, idx - context)
        end   = min(len(text), idx + len(needle) + context)
        out.append(text[start:end].strip())
        pos = idx + len(needle_l)
    return out

# ────────────── Action-parsing helpers (multi-action aware) ───────────
_ACTION_RE = re.compile(
    r'^Action:\s*(\w+)\s*\(\s*["\']?(.*?)["\']?\s*\)\s*$',
    re.I,
)
_DONE_RE   = re.compile(r"^Action:\s*Done!\s*$", re.I)
_SUPPORTED = {"search", "open", "find", "crawl", "recall", "done"}

def _parse_single_action(line: str) -> Tuple[str, str] | None:
    if _DONE_RE.match(line):
        return ("done", "")
    m = _ACTION_RE.match(line)
    if m:
        verb, arg = m.groups()
        verb_l = verb.lower()
        if verb_l not in _SUPPORTED:
            print(f"[Warning] Ignoring unsupported action '{verb}'.")
            return None
        return (verb_l, (arg or "").strip())
    return None

def _extract_actions(block: str,
                     limit: int = ACTION_LIMIT) -> List[Tuple[str, str]]:
    actions: List[Tuple[str, str]] = []
    for raw in block.splitlines():
        raw = raw.strip()
        if not raw.lower().startswith("action:"):
            continue
        maybe = _parse_single_action(raw)
        if maybe:
            actions.append(maybe)
            if len(actions) >= limit or maybe[0] == "done":
                break
    return actions

def _has_valid_format(block: str) -> bool:
    return block.lstrip().startswith("Thought:") and bool(_extract_actions(block))

def _select_actions(actions: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Return the first suggested action only."""

    return actions[:1] if actions else []

# ─────────────────── prompt-construction helpers ─────────────────────
def _build_system_prompt(q: str) -> str:
    return textwrap.dedent(f"""\
        You are an AI web-research agent.

        Tools you can call (multiple per turn):
          • Action: Search("query")   – search the web
          • Action: Open("url")       – open a URL
          • Action: Find("keyword")   – keyword search in the open page
          • Action: Crawl("site")     – crawl that site; add pages to memory
          • Action: Recall("query")   – retrieve pages similar to query
          • Action: Done!             – when you have enough evidence

        Output every turn (exactly one Thought then ≤{ACTION_LIMIT} Action lines):
          Thought: <multi-sentence reasoning>
          Action:  <tool call>
          …

        Rules:
          0-a. First turn MUST be Action: Search("query") (paraphrase allowed).
         0-b. After each Search, review the results and then use
               Action: Open("url") on the following turn.
         1. Gather evidence from ≥{MIN_SUPPORT_SOURCES} distinct URLs.
         2. The agent may execute up to **{ACTIONS_PER_TURN}** Actions per turn.
            After each Action you will see an Observation before deciding what
            to do next.
         3. NEVER fabricate information; rely only on opened pages.
         4. If evidence is insufficient, admit it instead of guessing.

        User question: "{q}"
        Begin.
    """)

def _summarise(user_q: str, sources: List[Tuple[str, str]]) -> str:
    src_block = "\n".join(f"[{i+1}] {u} — {s}" for i, (u, s) in enumerate(sources))
    prompt = f"""Write a concise answer (3–6 bullet points or a short paragraph).
Question:
{user_q}

Sources:
{src_block}

Answer:"""
    return llm_interface.call_llm_long(prompt, hard_cap=2048)

# ───────────── streaming generation – stop after final Action ─────────
def _stream_until_actions(prompt: str, verbose: bool = False) -> str:
    out, curr_line = "", ""
    actions: list[tuple[str, str]] = []
    max_tokens = _budget(prompt, reserve_ctx=400, hard_cap=None)

    try:
        for chunk in _llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.32,
            top_p=0.8,
            repeat_penalty=1.15,
            stop=["Observation:"],
            stream=True,
        ):
            tok = chunk["choices"][0]["text"]
            out += tok
            curr_line += tok
            if verbose:
                print(tok, end="", flush=True)
            if tok.endswith("\n"):
                maybe = _parse_single_action(curr_line.strip())
                if maybe:
                    actions.append(maybe)
                    break  # run action immediately
                curr_line = ""
    except Exception as e:
        print(f"⚠️ LLM decoding error: {e}")
        return ""
    if verbose:
        print()
    return out.strip()

# ───────────────────────── main agent loop ───────────────────────────
def run_research_agent(user_query: str, verbose: bool = False) -> None:
    mem  = Memory()
    hist = _build_system_prompt(user_query)
    loop = 0

    while loop < MAX_LOOPS:
        loop += 1
        if verbose:
            print(f"\n— LOOP {loop} —")

        actions_run = 0

        while actions_run < ACTIONS_PER_TURN:
            hist = _trim_history(hist)
            llm_out = _stream_until_actions(hist, verbose=verbose)

            if not _has_valid_format(llm_out):
                hist += "Observation: Reply must start with Thought: and contain an Action.\n"
                continue

            actions = _extract_actions(llm_out)
            if not actions:
                hist += "Observation: No valid Action detected; read instructions.\n"
                continue

            actions_to_run = _select_actions(actions)
            observations: list[str] = []

            # ---------- execute one tool ----------
            for action, arg in actions_to_run:
                last_action = action
                if action == "search":
                    formatted, _ = web_search.search_web(arg, max_results=8)
                    observations.append(formatted)

                elif action == "open":
                    snippet = mem.open_or_fetch(arg, query=user_query)
                    log_msg = f"[Open] {arg}\n{snippet[:160]}…"
                    mem.log_step(log_msg)
                    observations.append(log_msg)

                elif action == "find":
                    if mem.last_page_text is None:
                        observations.append("Nothing open—use Open(url) first.")
                    else:
                        hits = _find_chunks(mem.last_page_text, arg)
                        observations.append("…\n".join(hits) if hits
                                            else f"No occurrences of “{arg}”.")

                # -------- new tools --------
                elif action == "crawl":
                    pages = crawler.crawl_site(arg, max_pages=CRAWL_MAX_PAGES)
                    for url, html in pages.items():
                        mem.add_web_content(url, html)
                    observations.append(f"Crawled {len(pages)} pages from {arg}")

                elif action == "recall":
                    hits = mem.vstore.similarity_search(arg, k=5)
                    if not hits:
                        observations.append("No relevant docs yet – crawl first?")
                    else:
                        out = []
                        for url, score in hits:
                            snip = mem.open_or_fetch(url, query=arg, auto_snippet=True)
                            mem.log_step(f"[Recall] {url} ({score:.2f})\n{snip[:160]}…")
                            out.append(f"[{score:.2f}] {url}\n{snip[:160]}…")
                        observations.append("\n\n".join(out))

                elif action == "done":
                    if len(mem.sources) < MIN_SUPPORT_SOURCES:
                        observations.append(
                            f"Only {len(mem.sources)} source(s); need {MIN_SUPPORT_SOURCES}."
                        )
                    else:
                        answer = _summarise(user_query, mem.sources)
                        print("\n— FINAL ANSWER —\n" + answer)
                        return

                else:  # should never hit
                    warn = f"Unknown action “{action}”."
                    mem.log_step(warn)
                    observations.append(warn)

            # ---------- update conversation history ----------
            hist += f"{llm_out}\n\nObservation: {'\n----\n'.join(observations)}\n"
            hist = _trim_history(hist)
            if len(hist) > MAX_HISTORY_CHARS:
                hist = hist[-MAX_HISTORY_CHARS:]

            time.sleep(0.2)
            actions_run += 1

            if action == "done":
                break

        if action == "done":
            break

    print("\n— MAX LOOPS REACHED —")
    print(_summarise(user_query, mem.sources))