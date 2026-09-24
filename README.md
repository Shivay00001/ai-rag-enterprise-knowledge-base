# ai-rag-enterprise-knowledge-base

Real retrieval-augmented generation API. Upload `.txt`/`.md` documents, they are
chunked and indexed with **BM25** (stdlib implementation, no ML framework), and
questions are answered by a **real LLM call** grounded in the retrieved chunks,
with numbered citations.

## What it does

- `POST /documents/upload` — upload a `.txt`/`.md` file; it is chunked (~800 chars),
  indexed, and persisted under `./data`.
- `GET /documents` — list indexed documents.
- `POST /search` — raw BM25 retrieval (returns scored chunks).
- `POST /ask` — retrieves top-k chunks, calls the LLM with a strict
  "answer only from context, cite sources" prompt, returns answer + citations.

## API key

Live answers need an OpenAI-compatible chat-completions endpoint.

| Env var          | Purpose                                  | Default                      |
|------------------|------------------------------------------|------------------------------|
| `OPENAI_API_KEY` | **API key** for the LLM call             | (unset)                      |
| `OPENAI_BASE_URL`| Compatible endpoint base URL             | `https://api.openai.com/v1`  |
| `OPENAI_MODEL`   | Model name                               | `gpt-4o-mini`                |
| `RAG_DATA_DIR`   | Document store location                  | `./data`                     |

Without `OPENAI_API_KEY`, `/ask` returns **HTTP 503** with a clear message —
it never fabricates an answer. Retrieval (`/search`, upload, listing) works
fully offline.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --port 8001
# with a key:
OPENAI_API_KEY=sk-... uvicorn main:app --port 8001
```

## Verify

```bash
curl -F "file=@notes.md" http://localhost:8001/documents/upload
curl -X POST http://localhost:8001/search -H 'Content-Type: application/json' \
     -d '{"query":"refund window","top_k":3}'
curl -X POST http://localhost:8001/ask -H 'Content-Type: application/json' \
     -d '{"question":"What is the refund window?"}'
```

## Tests

```bash
python -m pytest tests/ -q
```

Tests cover: upload → BM25 retrieval finds the right chunk, `/ask` with no key
→ honest 503, `/ask` with a dummy key → real upstream 401 surfaced (proves the
HTTP client is real).
