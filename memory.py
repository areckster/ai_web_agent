# memory.py — 2025-07-01
"""
Cache HTML, track evidence, **and host the mini vector-store** used for RAG.
No external dependencies beyond bs4 (already in parser.py).
"""

from __future__ import annotations
from typing import List, Tuple, Optional

import fetcher, parser
from rag import MiniVectorStore


class Memory:
    """Caches pages, research traces, and index for similarity search."""

    def __init__(self) -> None:
        # ---------- raw storage ----------
        self.cache: dict[str, str] = {}             # url → html
        self.vstore = MiniVectorStore()             # url → tf-idf vector

        # ---------- runtime trace ----------
        self.research_log: List[str] = []
        self.sources: List[Tuple[str, str]] = []    # (url, snippet)
        self.last_opened_url: Optional[str] = None
        self.last_page_text: Optional[str] = None

    # ---------- evidence ----------
    def add_source(self, url: str, snippet: str) -> None:
        """Save a snippet that backs up the eventual answer."""
        self.sources.append((url, snippet[:200]))

    # ---------- trace helpers ----------
    def log_step(self, msg: str) -> None:
        self.research_log.append(msg)
        print(msg)

    # ---------- cache helpers ----------
    def add_web_content(self, url: str, html: str) -> None:
        """Store raw HTML and push plain-text into the vector store."""
        self.cache[url] = html
        text = parser.parse_html(html)
        if text:
            self.vstore.add(url, text)

    def get_web_content(self, url: str) -> str | None:
        return self.cache.get(url)

    # ---------- public high-level ----------
    def open_or_fetch(
        self,
        url: str,
        query: str | None = None,
        auto_snippet: bool = True,
    ) -> str:
        """
        Return a human-readable snippet from cache or network.
        If `query` is provided, highlight sentences that mention query terms.
        """
        html = self.get_web_content(url)
        if html is None:
            html = fetcher.fetch_page(url)
            if html:
                self.add_web_content(url, html)

        if not html:
            return f"⚠️ Failed to fetch {url}"

        text = parser.parse_html(html)
        self.last_opened_url = url
        self.last_page_text = text

        if query and auto_snippet:
            snippet = parser.extract_relevant_snippets(text, query) or text[:400]
        else:
            snippet = text[:400]

        self.add_source(url, snippet)
        return snippet