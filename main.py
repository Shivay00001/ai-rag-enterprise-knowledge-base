"""ai-rag-enterprise-knowledge-base: real retrieval-augmented generation API.

Upload .txt/.md docs -> chunked + BM25-indexed -> ask questions answered by a
real LLM grounded in the retrieved chunks, with citations.
"""
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from retrieval import BM25Index, chunk_text
from llm import LLMError, chat
import store

app = FastAPI(title="RAG Knowledge Base", version="1.0.0")
index = BM25Index()
store.rebuild_index(index)

ALLOWED = {".txt", ".md"}


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(store.list_documents()),
            "chunks": len(index.chunks)}


@app.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Only {sorted(ALLOWED)} files are supported, got '{ext}'")
    raw = file.file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "File is not valid UTF-8 text")
    if not text.strip():
        raise HTTPException(400, "Document is empty")
    chunks = chunk_text(text)
    doc = store.add_document(file.filename, text, chunks)
    for i, chunk in enumerate(chunks):
        index.add_chunk(doc["chunk_ids"][i], doc["doc_id"], chunk)
    return {"doc_id": doc["doc_id"], "filename": doc["filename"],
            "chunks": len(chunks), "chars": len(text)}


@app.get("/documents")
def list_docs():
    return {"documents": store.list_documents()}


@app.post("/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(400, "query must not be empty")
    return {"query": req.query, "results": index.search(req.query, req.top_k)}


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")
    hits = index.search(req.question, req.top_k)
    if not hits:
        raise HTTPException(404, "No indexed documents match the question. Upload documents first.")
    context = "\n\n".join(f"[{i+1}] (doc {h['doc_id']}) {h['text']}" for i, h in enumerate(hits))
    system = (
        "You are a precise assistant. Answer ONLY from the context below. "
        "Cite every factual claim with [1], [2], ... matching the numbered sources. "
        "If the context does not contain the answer, say so explicitly."
    )
    try:
        answer = chat(system, f"Context:\n{context}\n\nQuestion: {req.question}")
    except LLMError as e:
        # Honest error path: never fabricate an answer.
        code = 503 if e.status is None else (502 if e.status and e.status >= 500 else e.status)
        raise HTTPException(code, str(e))
    return {
        "question": req.question,
        "answer": answer,
        "citations": [
            {"n": i + 1, "chunk_id": h["chunk_id"], "doc_id": h["doc_id"],
             "score": h["score"], "excerpt": h["text"][:300]}
            for i, h in enumerate(hits)
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
