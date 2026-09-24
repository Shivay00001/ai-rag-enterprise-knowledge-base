from fastapi.testclient import TestClient
import os, shutil

os.environ["RAG_DATA_DIR"] = "/tmp/rag-test-data"
shutil.rmtree("/tmp/rag-test-data", ignore_errors=True)
# sandbox quirk: NO_PROXY contains patterns (e.g. *[::1]) that httpx's URL
# parser rejects; outbound goes through the egress proxy instead.
os.environ.pop("no_proxy", None)
os.environ.pop("NO_PROXY", None)

from main import app  # noqa: E402

client = TestClient(app)

DOC1 = """# Company Refund Policy

All purchases are eligible for a full refund within 30 days of purchase.
After 30 days, refunds are issued as store credit only.
Refunds are processed within 5 business days of approval."""

DOC2 = """# Shipping Information

Standard shipping takes 5 to 7 business days and is free for orders over $50.
Express shipping takes 2 business days and costs $12.
International shipping is available to 40 countries."""


def _seed():
    for name, doc, ctype in (("refund.md", DOC1, "text/markdown"),
                             ("shipping.txt", DOC2, "text/plain")):
        r = client.post("/documents/upload", files={"file": (name, doc, ctype)})
        assert r.status_code == 200, r.text


def test_upload_and_search():
    _seed()
    s = client.post("/search", json={"query": "refund policy 30 days full refund", "top_k": 3})
    assert s.status_code == 200
    results = s.json()["results"]
    assert results, "expected retrieval hits"
    assert "refund" in results[0]["text"].lower()
    assert results[0]["doc_id"] != ""

    s2 = client.post("/search", json={"query": "express shipping costs $12", "top_k": 3})
    r2 = s2.json()["results"]
    assert r2 and "$12" in r2[0]["text"]


def test_ask_without_key_is_honest_error():
    _seed()
    os.environ.pop("OPENAI_API_KEY", None)
    r = client.post("/ask", json={"question": "What is the refund window?"})
    assert r.status_code == 503
    assert "OPENAI_API_KEY" in r.json()["detail"]


def test_ask_with_bad_key_returns_upstream_error():
    _seed()
    os.environ["OPENAI_API_KEY"] = "sk-dummy-invalid-key-for-testing"
    try:
        r = client.post("/ask", json={"question": "What is the refund window?"})
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
    # real client -> real provider -> honest 401 surfaced (proves the HTTP client is real)
    assert r.status_code == 401, (r.status_code, r.text[:300])
    assert "Incorrect API key" in r.text or "401" in r.text
