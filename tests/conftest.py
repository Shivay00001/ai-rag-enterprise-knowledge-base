"""Pytest configuration and fixtures."""
import pytest
import asyncio
from typing import Generator, AsyncGenerator

from fastapi.testclient import TestClient
from httpx import AsyncClient

# Ensure event loop is available for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    from src.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
async def async_client():
    """Async test client for async tests."""
    from src.main import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_text():
    """Sample text for testing chunkers."""
    return """
    This is a sample document for testing purposes.
    
    Chapter 1: Introduction
    
    This chapter introduces the main concepts. We will cover multiple topics
    including data processing, machine learning, and best practices.
    
    The quick brown fox jumps over the lazy dog. This sentence contains all
    letters of the alphabet and is commonly used for testing.
    
    Chapter 2: Methods
    
    In this chapter, we describe the methods used in our approach.
    We use a combination of statistical analysis and neural networks.
    """


@pytest.fixture
def sample_pdf_bytes():
    """Generate minimal PDF bytes for testing."""
    # Minimal valid PDF
    pdf_content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj
4 0 obj << /Length 44 >> stream
BT /F1 12 Tf 100 700 Td (Test Document) Tj ET
endstream endobj
xref
0 5
trailer << /Size 5 /Root 1 0 R >>
startxref
EOF"""
    return pdf_content


@pytest.fixture
def auth_headers(client):
    """Get authentication headers for API tests."""
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "demo", "password": "demo123"}
    )
    if response.status_code == 200:
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    return {}
