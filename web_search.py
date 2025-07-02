# web_search.py  — 2025-06-18  (clean version)
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import html, random, time
import requests
from bs4 import BeautifulSoup

import utils
from fetcher import _ua

# ────────────────────────── constants ────────────────────────── #
_DDG_HTML = "https://duckduckgo.com/html/"
_STARTPAGE = "https://startpage.com/do/search"
_HEADERS   = {
    "User-Agent": _ua(),
    "Referer": "https://duckduckgo.com/",
    "Content-Type": "application/x-www-form-urlencoded",
    "Accept-Language": "en-US,en;q=0.9",
}

# ─────────────────────── helpers ─────────────────────────────── #
def _clean_hit(href: str) -> bool:
    """True if the URL should be kept (drops ads / trackers)."""
    if not href.startswith("http"):
        return False
    bad_sub = ("duckduckgo.com/y.js", "amazon.com", "adserver", "bing.com/aclick")
    return not any(s in href for s in bad_sub)



def _ddg_html(query: str, k: int = 8) -> List[Dict[str, Any]]:
    """
    DuckDuckGo lightweight HTML search (GET).
    Works 2025-06-18; update selectors here if DDG tweaks markup again.
    """
    params = {"q": query, "kl": "us-en"}
    time.sleep(random.uniform(0, 1))  # jitter → fewer 403s
    r = requests.get(_DDG_HTML, params=params, headers=_HEADERS, timeout=10)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    hits: List[Dict[str, Any]] = []

    for body in soup.select("div.result__body"):
        a = body.select_one("a.result__a")
        if not a:
            continue
        href = a.get("href", "")
        if not _clean_hit(href): 
            continue
        title = a.get_text(" ", strip=True)
        snip  = body.select_one("div.result__snippet")
        snippet = snip.get_text(" ", strip=True) if snip else ""
        hits.append({"title": title,
                     "href": href,
                     "body": html.unescape(snippet)})
        if len(hits) >= k:
            break
    return hits

def _startpage_html(query: str, k: int = 8) -> List[Dict[str, Any]]:
    """Fallback search using Startpage Lite."""
    params = {"query": query, "language": "english"}
    hdrs = {**_HEADERS, "Referer": "https://www.startpage.com/"}
    r = requests.get(_STARTPAGE, params=params, headers=hdrs, timeout=10)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    hits: List[Dict[str, Any]] = []

    for res in soup.select("a.w-gl__result-title"):
        href = res.get("href", "")
        if not _clean_hit(href):
            continue
        title = res.get_text(" ", strip=True)
        snip = res.find_next("p", class_="w-gl__description")
        snippet = snip.get_text(" ", strip=True) if snip else ""
        hits.append({"title": title, "href": href, "body": snippet})
        if len(hits) >= k:
            break
    return hits

# ─────────────────────── public API ──────────────────────────── #
def search_web(query: str,
               max_results: int = 8) -> Tuple[str, List[Dict[str, Any]]]:
    """Entry point called by the agent."""
    print(f"\n🔍  SEARCH «{query}»")
    t0 = time.time()

    try:
        results = _ddg_html(query, k=max_results)
    except Exception as e:
        print(f"DuckDuckGo failed ({e}); trying Startpage.")
        try:
            results = _startpage_html(query, k=max_results)
        except Exception as e2:
            print(f"Startpage also failed: {e2}")
            results = []

    dt = (time.time() - t0) * 1000
    print(f"   ↳ {len(results)} hits in {dt:0.0f} ms")
    return utils.format_search_results(results), results