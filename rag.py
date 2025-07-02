# rag.py — 2025-07-01 FIXED

import math, re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

def _tokens(text):           # very cheap word tokenizer
    return [w.lower() for w in _TOKEN_RE.findall(text)]

def _tfidf_vector(doc_tokens, idf):
    counts = Counter(doc_tokens)
    return {t: counts[t] * idf[t] for t in counts}

def _cosine(a, b):
    if not a or not b:
        return 0.0
    num = sum(a[t] * b.get(t, 0.0) for t in a)
    den = math.sqrt(sum(v*v for v in a.values())) * math.sqrt(sum(v*v for v in b.values()))
    return num / den if den else 0.0

class MiniVectorStore:
    def __init__(self):
        self.docs = []          # list[tuple[url, tokens, tfidf]]
        self.idf  = Counter()
        self._finalised = False

    def add(self, url, text):
        toks = _tokens(text)
        if not toks:
            return
        self.docs.append((url, toks, None))
        self.idf.update(set(toks))
        self._finalised = False

    def _finalize(self):
        N = len(self.docs) or 1
        for t in self.idf:
            self.idf[t] = math.log(N / (1 + self.idf[t]))

        for i, (url, toks, _) in enumerate(self.docs):
            self.docs[i] = (url, toks, _tfidf_vector(toks, self.idf))
        self._finalised = True

    def similarity_search(self, query, k=5):
        if self.docs and not self._finalised:
            self._finalize()

        q_vec = _tfidf_vector(_tokens(query), self.idf)
        scored = []
        for url, _, doc_vec in self.docs:
            if doc_vec:
                sim = _cosine(doc_vec, q_vec)
                scored.append((url, sim))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]