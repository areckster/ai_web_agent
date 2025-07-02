# crawler.py  — minimal, sync, zero external deps beyond requests/bs4

from collections import deque
from urllib.parse import urljoin, urlparse

import re, time, requests
from bs4 import BeautifulSoup

_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Safari/605.1.15"
)

def _is_html(resp):
    ct = resp.headers.get("content-type", "")
    return "text/html" in ct or "application/xhtml+xml" in ct

def _extract_links(base_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        if urlparse(href).netloc == urlparse(base_url).netloc:  # stay on-site
            links.append(href.split("#")[0])
    return links

def crawl_site(seed_url: str, max_pages=40, delay=0.5) -> dict[str, str]:
    """
    Breadth-first crawl up to `max_pages` pages under the same domain.
    Returns {url: plaintext}.
    """
    seen, out = set(), {}
    q = deque([seed_url])

    while q and len(out) < max_pages:
        url = q.popleft()
        if url in seen:
            continue
        seen.add(url)

        try:
            resp = requests.get(url, headers={"User-Agent": _DEFAULT_UA}, timeout=10)
            if not _is_html(resp):
                continue
            html = resp.text
        except requests.RequestException:
            continue

        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        out[url] = text[:8000]          # cap to keep RAM sane
        for link in _extract_links(url, html):
            if link not in seen and len(seen) + len(q) < max_pages * 3:
                q.append(link)
        time.sleep(delay)

    return out