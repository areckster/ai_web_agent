#fetcher.py
from __future__ import annotations
import requests, random, time

_UA_POOL = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; arm64; Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko)",
] 

def _ua() -> str:
    return random.choice(_UA_POOL)

def fetch_page(url: str, timeout: int = 10, retries: int = 2) -> str | None:
    for attempt in range(1, retries + 2):
        try:
            print(f"FETCH {url}  (try {attempt})")
            r = requests.get(url, headers={"User-Agent": _ua()}, timeout=timeout)
            r.raise_for_status()
            return r.text
        except requests.RequestException as exc:
            print(f"  ↳ error: {exc}")
            if attempt <= retries:
                time.sleep(1.5 * attempt)
    return None