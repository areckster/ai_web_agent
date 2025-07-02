# agent.py — 2025-07-01
"""
Web-research agent with **Crawl** and **Recall** tools + lightweight RAG.
Keeps your original streaming logic & action throttling.
"""

from __future__ import annotations

import concurrent.futures
import re, textwrap, time
from typing import List, Tuple

import utils, web_search, llm_interface
from memory import Memory
import crawler            # ← new helper module

# ─────────────────────────── tunables ────────────────────────────
MAX_LOOPS            = 4
MIN_SUPPORT_SOURCES  = 1
SELF_CONSISTENCY_N   = 2
AUTO_OPEN_TOP_K      = 1
THREAD_POOL_WORKERS  = 6
MAX_HISTORY_CHARS    = 12_000
ACTION_LIMIT         = 3
CRAWL_MAX_PAGES      = 40
# ──────────────────────────────────────────────────────────────────

_llm      = llm_interface._llm
_budget   = llm_interface._budget
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

# ─────────────────────── utility helpers ────────────────────────
def _words(t: str) -> set[str]:
    return {w.lower() for w in _TOKEN_RE.findall(t) if len(w) > 3}

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
    r'^Action:\s*(Search|Open|Find|Crawl|Recall)\s*\(\s*["\']?(.*?)["\']?\s*\)\s*$',
    re.I,
)
_DONE_RE   = re.compile(r"^Action:\s*Done!\s*$", re.I)

def _parse_single_action(line: str) -> Tuple[str, str] | None:
    if _DONE_RE.match(line):
        return ("done", "")
    m = _ACTION_RE.match(line)
    if m:
        verb, arg = m.groups()
        return (verb.lower(), arg.strip())
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
    # Prefer the first Search if any; else run the very first action
    for a, arg in actions:
        if a == "search":
            return [(a, arg)]
    return [actions[0]]

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
          0-b. After each Search, immediately Open one promising URL.
          1. Gather evidence from ≥{MIN_SUPPORT_SOURCES} distinct URLs.
          2. The agent will execute at most **one** Action per cycle.

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
                if len(actions) >= ACTION_LIMIT or maybe[0] == "done":
                    break
            curr_line = ""
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

        # ---------- execute ≤1 tool ----------
        for action, arg in actions_to_run:
            if action == "search":
                formatted, results = web_search.search_web(arg, max_results=8)
                obs = formatted

                def _fetch(res):
                    url = res.get("href")
                    if not url:
                        return None
                    snippet = mem.open_or_fetch(url, query=arg)
                    return url, snippet

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=THREAD_POOL_WORKERS
                ) as pool:
                    futures = [pool.submit(_fetch, r)
                               for r in results[:AUTO_OPEN_TOP_K]]
                    for f in concurrent.futures.as_completed(futures):
                        res = f.result()
                        if res:
                            u, sn = res
                            obs += f"\n\n[Auto-opened] {u}\nSnippet: {sn}…"
                observations.append(obs)

            elif action == "open":
                observations.append(mem.open_or_fetch(arg, query=user_query))

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
                observations.append(f"Unknown action “{action}”.")

        # ---------- update conversation history ----------
        hist += f"{llm_out}\n\nObservation: {'\n----\n'.join(observations)}\n"
        if len(hist) > MAX_HISTORY_CHARS:
            hist = hist[-MAX_HISTORY_CHARS:]

        time.sleep(0.2)

    print("\n— MAX LOOPS REACHED —")
    print(_summarise(user_query, mem.sources))