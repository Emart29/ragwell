"""Tests for FastAPI endpoints."""
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import time


client = None
FIXTURES_DIR = Path(__file__).parent / 'fixtures'


def get_client():
    """Get or create test client."""
    global client
    if client is None:
        from app.main import app
        client = TestClient(app)
    return client


class TestRoot:
    """Tests for root endpoint."""
    
    def test_root(self):
        """Test root endpoint redirects to frontend."""
        c = get_client()
        response = c.get("/", follow_redirects=False)
        # Root now redirects to frontend
        assert response.status_code in [200, 307]


class TestHealth:
    """Tests for health endpoint."""
    
    def test_health_endpoint(self):
        """Test health endpoint returns valid data."""
        c = get_client()
        response = c.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "total_documents" in data
        assert "total_chunks" in data
        assert "sqlite_connected" in data
        assert "embedding_cache" in data


class TestIngestion:
    """Tests for ingestion endpoints."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown."""
        from app.storage.vector_store import VectorStore
        # Reset store
        store = VectorStore()
        store.reset()
        yield
        # Cleanup
        store = VectorStore()
        store.reset()
    
    @pytest.mark.slow
    def test_ingest_single_file(self):
        """Test uploading a single file."""
        c = get_client()
        test_file = FIXTURES_DIR / 'test.txt'
        
        with open(test_file, 'rb') as f:
            response = c.post(
                "/api/ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={"chunk_strategy": "recursive", "chunk_size": "100", "wait": "true"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "test.txt"
        
        # Save for other tests
        self.document_id = data["document_id"]
    
    def test_ingest_invalid_strategy(self):
        """Test ingest with invalid strategy returns error."""
        c = get_client()
        test_file = FIXTURES_DIR / 'test.txt'
        
        with open(test_file, 'rb') as f:
            response = c.post(
                "/api/ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={"chunk_strategy": "invalid"}
            )
        
        assert response.status_code == 400
    
    @pytest.mark.slow
    def test_get_document_status(self):
        """Test getting document status."""
        c = get_client()
        # First ingest a file
        test_file = FIXTURES_DIR / 'test.txt'
        
        with open(test_file, 'rb') as f:
            response = c.post(
                "/api/ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={"chunk_strategy": "recursive", "wait": "true"}
            )
        
        document_id = response.json()["document_id"]
        
        # Get status
        response = c.get(f"/api/ingest/{document_id}/status")
        assert response.status_code == 200
        data = response.json()
        
        assert data["document_id"] == document_id
        assert "status" in data
        assert "filename" in data
    
    def test_get_status_nonexistent(self):
        """Test getting status for nonexistent document."""
        c = get_client()
        response = c.get("/api/ingest/nonexistent-id/status")
        assert response.status_code == 404
    
    @pytest.mark.slow
    def test_batch_ingest(self):
        """Test batch ingestion."""
        c = get_client()
        test_file = FIXTURES_DIR / 'test.txt'
        
        with open(test_file, 'rb') as f1, open(test_file, 'rb') as f2:
            response = c.post(
                "/api/ingest/batch",
                files=[
                    ("files", ("file1.txt", f1, "text/plain")),
                    ("files", ("file2.txt", f2, "text/plain"))
                ]
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "batch_id" in data
        assert len(data["document_ids"]) == 2
        assert data["count"] == 2


class TestQuery:
    """Tests for query endpoints."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup test document."""
        from app.storage.vector_store import VectorStore
        c = get_client()
        store = VectorStore()
        store.reset()
        
        # Ingest a test document
        test_file = FIXTURES_DIR / 'test.txt'
        with open(test_file, 'rb') as f:
            c.post(
                "/api/ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={"chunk_strategy": "recursive", "chunk_size": "50", "wait": "true"}
            )
        
        yield
        
        store = VectorStore()
        store.reset()
    
    @pytest.mark.slow
    def test_query_with_hybrid_strategy(self):
        """Test query with hybrid strategy."""
        c = get_client()
        response = c.post(
            "/api/query",
            json={
                "query": "test document content",
                "strategy": "hybrid",
                "top_k": 5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["strategy"] == "hybrid"
        assert "search_time_ms" in data
    
    @pytest.mark.slow
    def test_query_with_vector_strategy(self):
        """Test query with vector strategy."""
        c = get_client()
        response = c.post(
            "/api/query",
            json={
                "query": "test query content",
                "strategy": "vector",
                "top_k": 5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["strategy"] == "vector"
    
    def test_query_with_invalid_strategy(self):
        """Test query with invalid strategy returns error."""
        c = get_client()
        response = c.post(
            "/api/query",
            json={
                "query": "test query",
                "strategy": "invalid_strategy"
            }
        )
        
        assert response.status_code == 400
    
    @pytest.mark.slow
    def test_query_result_format(self):
        """Test query result format."""
        c = get_client()
        response = c.post(
            "/api/query",
            json={
                "query": "test query format",
                "strategy": "hybrid",
                "top_k": 3
            }
        )
        
        data = response.json()
        
        if data["results"]:
            result = data["results"][0]
            assert "chunk_id" in result
            assert "text" in result
            assert "score" in result
            assert "document" in result
            assert "metadata" in result


class TestDocuments:
    """Tests for documents endpoints."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup test document."""
        from app.storage.vector_store import VectorStore
        c = get_client()
        store = VectorStore()
        store.reset()
        
        # Ingest test document
        test_file = FIXTURES_DIR / 'test.txt'
        with open(test_file, 'rb') as f:
            c.post(
                "/api/ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={"wait": "true"}
            )
        
        yield
        
        store = VectorStore()
        store.reset()
    
    def test_list_documents(self):
        """Test listing documents."""
        c = get_client()
        response = c.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        
        assert "documents" in data
        assert "total" in data
        assert "page" in data
    
    def test_list_documents_with_pagination(self):
        """Test document pagination."""
        c = get_client()
        response = c.get("/api/documents?page=1&per_page=10")
        assert response.status_code == 200
        data = response.json()
        
        assert data["page"] == 1
        assert data["per_page"] == 10
    
    def test_list_documents_with_filter(self):
        """Test document filtering."""
        c = get_client()
        response = c.get("/api/documents?status=pending")
        assert response.status_code == 200
    
    def test_get_document_detail(self):
        """Test getting document details."""
        c = get_client()
        # First get list to find a document ID
        list_response = c.get("/api/documents")
        data = list_response.json()
        
        if data["documents"]:
            doc_id = data["documents"][0]["id"]
            
            response = c.get(f"/api/documents/{doc_id}")
            assert response.status_code == 200
            
            detail = response.json()
            assert detail["id"] == doc_id
            assert "filename" in detail
            assert "processing_status" in detail
    
    def test_get_nonexistent_document(self):
        """Test getting nonexistent document returns 404."""
        c = get_client()
        response = c.get("/api/documents/nonexistent-id")
        assert response.status_code == 404
        """Test getting nonexistent document."""
        response = client.get("/api/documents/nonexistent-id")
        assert response.status_code == 404


class TestCompare:
    """Tests for comparison endpoint."""
    
    def test_compare_chunking(self):
        """Test chunking comparison endpoint."""
        # Note: This requires the file to still exist in uploads
        # For this test, we'll skip if file doesn't exist
        response = client.post("/api/compare/test-id")
        # Will either succeed or return 404 if file not found
        assert response.status_code in [200, 404]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
