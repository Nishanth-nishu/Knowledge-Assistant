"""
Tests for the API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_readiness_check(self, client):
        """Test readiness check."""
        response = client.get("/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "checks" in data


class TestQueryEndpoints:
    """Tests for query endpoints."""
    
    def test_query_basic(self, client):
        """Test basic query."""
        response = client.post(
            "/api/v1/query",
            json={"query": "What is the policy on remote work?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "query_id" in data
        assert "answer" in data
        assert data["query"] == "What is the policy on remote work?"
    
    def test_query_with_options(self, client):
        """Test query with options."""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "What is the policy?",
                "top_k": 5,
                "use_reranker": True,
                "use_knowledge_graph": True,
            }
        )
        
        assert response.status_code == 200
    
    def test_query_empty_rejected(self, client):
        """Test empty query is rejected."""
        response = client.post(
            "/api/v1/query",
            json={"query": ""}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_query_not_found(self, client):
        """Test getting non-existent query."""
        response = client.get("/api/v1/query/nonexistent-id")
        
        assert response.status_code == 404


class TestDocumentEndpoints:
    """Tests for document endpoints."""
    
    def test_list_documents(self, client):
        """Test listing documents."""
        response = client.get("/api/v1/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
    
    def test_get_document_not_found(self, client):
        """Test getting non-existent document."""
        response = client.get("/api/v1/documents/nonexistent-id")
        
        assert response.status_code == 404


class TestAdminEndpoints:
    """Tests for admin endpoints."""
    
    def test_system_status(self, client):
        """Test system status endpoint."""
        response = client.get("/api/v1/admin/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
    
    def test_index_stats(self, client):
        """Test index stats endpoint."""
        response = client.get("/api/v1/admin/index/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "collection_name" in data
    
    def test_list_models(self, client):
        """Test list models endpoint."""
        response = client.get("/api/v1/admin/models")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_config(self, client):
        """Test get config endpoint."""
        response = client.get("/api/v1/admin/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "retrieval" in data
        assert "agent" in data
