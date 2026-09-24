"""Real BM25 retrieval over chunked documents. Standard library only."""
import math
import re
from collections import Counter


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """Okapi BM25 index. add_chunk / search are the whole API."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks = []          # list of dicts: {id, doc_id, text, tokens}
        self.doc_freq = Counter()
        self.avg_len = 0.0

    def add_chunk(self, chunk_id: str, doc_id: str, text: str):
        tokens = tokenize(text)
        self.chunks.append({"id": chunk_id, "doc_id": doc_id, "text": text, "tokens": tokens})
        for t in set(tokens):
            self.doc_freq[t] += 1
        self.avg_len = sum(len(c["tokens"]) for c in self.chunks) / max(len(self.chunks), 1)

    def idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self.doc_freq.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_terms, chunk) -> float:
        tf = Counter(chunk["tokens"])
        dl = len(chunk["tokens"])
        s = 0.0
        for t in query_terms:
            if t not in tf:
                continue
            idf = self.idf(t)
            denom = tf[t] + self.k1 * (1 - self.b + self.b * dl / max(self.avg_len, 1e-9))
            s += idf * tf[t] * (self.k1 + 1) / denom
        return s

    def search(self, query: str, top_k: int = 5):
        qterms = tokenize(query)
        scored = [(self.score(qterms, c), c) for c in self.chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"chunk_id": c["id"], "doc_id": c["doc_id"], "text": c["text"], "score": round(s, 4)}
            for s, c in scored[:top_k] if s > 0
        ]


def chunk_text(text: str, max_chars: int = 800):
    """Split on blank lines; merge small paragraphs up to max_chars."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            # hard-split very long paragraphs
            while len(p) > max_chars:
                chunks.append(p[:max_chars])
                p = p[max_chars:]
            cur = p
    if cur:
        chunks.append(cur)
    return chunks or [text[:max_chars]]
