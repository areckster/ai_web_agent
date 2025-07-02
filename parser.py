#parser.py
import re
from bs4 import BeautifulSoup

def _sentence_tokens(text: str):
    """Very simple sentence tokenizer that splits on '.', '!' or '?'."""
    return re.split(r"(?<=[.!?])\s+", text)

def extract_relevant_snippets(text: str, query: str, n: int = 4) -> str:
    words = [w.lower() for w in query.split()]
    sents = _sentence_tokens(text)
    hits = [s for s in sents if any(w in s.lower() for w in words)]
    return " • ".join(hits[:n])
 
def parse_html(html: str, max_length: int = 3500) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.find("div", role="main")
    text = (main or soup.body or soup).get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text[:max_length]


def find_chunks(text: str, needle: str,
                context: int = 120,
                max_hits: int = 3) -> list[str]:
    lowered, pos, out = text.lower(), 0, []
    while len(out) < max_hits:
        idx = lowered.find(needle.lower(), pos)
        if idx == -1:
            break
        start = max(0, idx - context)
        end   = min(len(text), idx + len(needle) + context)
        out.append(text[start:end].strip())
        pos = idx + len(needle)
    return out