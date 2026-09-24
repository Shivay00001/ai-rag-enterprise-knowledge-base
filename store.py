"""Tiny on-disk document store: metadata + raw text under ./data."""
import json
import os
import uuid
from datetime import datetime, timezone

DATA_DIR = os.environ.get("RAG_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
DOCS_FILE = os.path.join(DATA_DIR, "docs.json")


def _load():
    if os.path.exists(DOCS_FILE):
        with open(DOCS_FILE) as f:
            return json.load(f)
    return []


def _save(docs):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DOCS_FILE, "w") as f:
        json.dump(docs, f, indent=2)


def add_document(filename: str, text: str, chunks: list[str]) -> dict:
    docs = _load()
    doc_id = uuid.uuid4().hex[:12]
    doc = {
        "doc_id": doc_id,
        "filename": filename,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "chars": len(text),
        "num_chunks": len(chunks),
        "text": text,
        "chunk_ids": [f"{doc_id}:{i}" for i in range(len(chunks))],
    }
    docs.append(doc)
    _save(docs)
    return doc


def list_documents():
    return [
        {k: d[k] for k in ("doc_id", "filename", "uploaded_at", "chars", "num_chunks")}
        for d in _load()
    ]


def rebuild_index(index):
    """Repopulate a BM25Index from everything on disk (called at startup)."""
    for d in _load():
        for i, cid in enumerate(d["chunk_ids"]):
            index.add_chunk(cid, d["doc_id"], _chunk_text(d, i))


def _chunk_text(doc, i):
    # recompute chunks deterministically the same way as at upload time
    from retrieval import chunk_text
    return chunk_text(doc["text"])[i]
