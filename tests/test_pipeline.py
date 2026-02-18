"""Tests for full processing pipeline."""
import pytest
from pathlib import Path
from app.pipeline import process_document, Pipeline, ProcessingResult
from app.storage.vector_store import VectorStore
from app.storage.database import DatabaseManager
from app.storage.models import Document, Chunk


FIXTURES_DIR = Path(__file__).parent / 'fixtures'


class TestPipeline:
    """Integration tests for full processing pipeline."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Reset vector store before each test
        store = VectorStore()
        store.reset()
        yield
        # Cleanup after test
        store = VectorStore()
        store.reset()
    
    def test_process_pdf_document(self):
        """Test processing a PDF document end-to-end."""
        result = process_document(
            FIXTURES_DIR / 'test.pdf',
            chunk_strategy='recursive',
            chunk_size=100,
            chunk_overlap=20
        )
        
        assert result.success is True
        assert result.document_id
        assert result.filename == 'test.pdf'
        assert result.chunk_count > 0
        assert result.processing_time_seconds > 0
        assert result.cache_hit_rate >= 0.0
    
    def test_process_txt_document(self):
        """Test processing a TXT document."""
        result = process_document(
            FIXTURES_DIR / 'test.txt',
            chunk_strategy='recursive',
            chunk_size=100
        )
        
        assert result.success is True
        assert result.chunk_count > 0
    
    def test_process_with_semantic_strategy(self):
        """Test processing with semantic chunking."""
        result = process_document(
            FIXTURES_DIR / 'test.txt',
            chunk_strategy='semantic',
            chunk_size=150
        )
        
        assert result.success is True
        assert result.chunk_count > 0
    
    def test_document_stored_in_sqlite(self):
        """Test that document is stored in SQLite."""
        result = process_document(FIXTURES_DIR / 'test.txt')
        
        with DatabaseManager() as db:
            doc = db.query(Document).filter(Document.id == result.document_id).first()
            
            assert doc is not None
            assert doc.filename == 'test.txt'
            assert doc.processing_status == 'completed'
    
    def test_chunks_stored_in_sqlite(self):
        """Test that chunks are stored in SQLite."""
        result = process_document(FIXTURES_DIR / 'test.txt', chunk_size=50)
        
        with DatabaseManager() as db:
            chunks = db.query(Chunk).filter(Chunk.document_id == result.document_id).all()
            
            assert len(chunks) == result.chunk_count
            
            for chunk in chunks:
                assert chunk.text
                assert chunk.token_count > 0
                assert chunk.quality_score is not None
    
    def test_chunks_stored_in_chromadb(self):
        """Test that chunks are stored in ChromaDB."""
        result = process_document(FIXTURES_DIR / 'test.txt', chunk_size=50)
        
        store = VectorStore()
        
        # Search for chunks from this document
        results = store.collection.get(
            where={'document_id': result.document_id}
        )
        
        assert len(results['ids']) == result.chunk_count
    
    def test_quality_filtering(self):
        """Test that low-quality chunks are filtered out."""
        result = process_document(FIXTURES_DIR / 'test.txt')
        
        # All processed chunks should be quality-checked
        assert result.chunks_rejected >= 0
        
        with DatabaseManager() as db:
            chunks = db.query(Chunk).filter(
                Chunk.document_id == result.document_id
            ).all()
            
            # All stored chunks should have quality scores >= 0.5
            for chunk in chunks:
                assert chunk.quality_score >= 0.5
    
    def test_caching_improves_performance(self):
        """Test that caching improves performance on second run."""
        # First run - cache misses
        result1 = process_document(FIXTURES_DIR / 'test.txt')
        
        # Reset but keep cache
        store = VectorStore()
        store.reset()
        
        # Second run - should have cache hits
        result2 = process_document(FIXTURES_DIR / 'test.txt')
        
        # Second run should be faster or have cache hits
        assert result2.cache_hit_rate > result1.cache_hit_rate or result2.cache_hit_rate > 0
    
    def test_vector_store_stats(self):
        """Test vector store statistics after processing."""
        # Process a document
        process_document(FIXTURES_DIR / 'test.txt')
        
        store = VectorStore()
        stats = store.get_stats()
        
        assert stats['total_documents'] >= 1
        assert stats['total_chunks'] > 0
        assert stats['average_quality_score'] > 0
    
    def test_search_after_processing(self):
        """Test that search works after processing."""
        result = process_document(FIXTURES_DIR / 'test.txt')
        
        store = VectorStore()
        
        # Create a simple query embedding (normally from embedder)
        from app.embeddings.embedder import Embedder
        embedder = Embedder.get_instance()
        query_embedding = embedder.embed("test query")
        
        # Search
        search_results = store.search(query_embedding, n_results=5)
        
        assert len(search_results['ids'][0]) > 0
    
    def test_failed_processing(self):
        """Test handling of failed processing."""
        result = process_document(
            Path('nonexistent_file.txt'),
            chunk_strategy='recursive'
        )
        
        assert result.success is False
        assert result.error is not None
        assert result.chunk_count == 0
    
    def test_pipeline_class(self):
        """Test Pipeline class directly."""
        pipeline = Pipeline()
        
        result = pipeline.process_document(
            FIXTURES_DIR / 'test.txt',
            chunk_strategy='recursive',
            chunk_size=100,
            chunk_overlap=20
        )
        
        assert result.success is True
        assert result.chunk_count > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
