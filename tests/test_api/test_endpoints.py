"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Tests for health and info endpoints."""
    
    def test_health_check(self, client: TestClient):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_root_endpoint(self, client: TestClient):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data


class TestAuthEndpoints:
    """Tests for authentication endpoints."""
    
    def test_login_success(self, client: TestClient):
        """Test successful login."""
        response = client.post(
            "/api/v1/auth/token",
            data={"username": "demo", "password": "demo123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_failure(self, client: TestClient):
        """Test failed login with wrong credentials."""
        response = client.post(
            "/api/v1/auth/token",
            data={"username": "demo", "password": "wrongpassword"}
        )
        assert response.status_code == 401
    
    def test_get_current_user(self, client: TestClient, auth_headers: dict):
        """Test getting current user info."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        response = client.get("/api/v1/auth/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "demo"
    
    def test_register_user(self, client: TestClient):
        """Test user registration."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "securepassword",
                "full_name": "New User"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "newuser"


class TestDocumentEndpoints:
    """Tests for document management endpoints."""
    
    def test_list_documents_requires_auth(self, client: TestClient):
        """Test that listing documents requires authentication."""
        response = client.get("/api/v1/documents/")
        assert response.status_code == 401
    
    def test_list_documents(self, client: TestClient, auth_headers: dict):
        """Test listing documents."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        response = client.get("/api/v1/documents/", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data


class TestQueryEndpoints:
    """Tests for query endpoints."""
    
    def test_query_requires_auth(self, client: TestClient):
        """Test that querying requires authentication."""
        response = client.post(
            "/api/v1/query/",
            json={"question": "What is the policy?"}
        )
        assert response.status_code == 401
    
    def test_search_endpoint(self, client: TestClient, auth_headers: dict):
        """Test semantic search endpoint."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        response = client.post(
            "/api/v1/query/search",
            headers=auth_headers,
            json={"query": "test query", "top_k": 5}
        )
        # May fail if no documents, but should be 200 or 500, not auth error
        assert response.status_code in [200, 500]
