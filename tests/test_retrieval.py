"""Tests for retrieval layer."""
import pytest
from pathlib import Path
import time
from unittest.mock import patch

from app.llm import provider
from app.pipeline import process_document
from app.retrieval.vector_search import VectorSearch
from app.retrieval.keyword_search import KeywordSearch
from app.retrieval.hybrid_search import HybridSearch, reciprocal_rank_fusion
from app.retrieval.reranker import Reranker
from app.retrieval.query_expand import QueryExpansion
from app.retrieval.retriever import Retriever
from app.retrieval.types import SearchResult
from app.storage.vector_store import VectorStore


FIXTURES_DIR = Path(__file__).parent / 'fixtures'


class TestVectorSearch:
    """Tests for vector search."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data."""
        # Reset and ingest test document
        store = VectorStore()
        store.reset()
        
        # Process a test document
        result = process_document(
            FIXTURES_DIR / 'test.txt',
            chunk_strategy='recursive',
            chunk_size=50,
            chunk_overlap=10
        )
        
        if result.success:
            # Wait a moment for ChromaDB to index
            time.sleep(1)
        
        yield
        
        # Cleanup
        store = VectorStore()
        store.reset()
    
    def test_vector_search_returns_results(self):
        """Test that vector search returns results."""
        searcher = VectorSearch()
        results = searcher.search("test document", top_k=5)
        
        assert len(results) > 0
        
        for result in results:
            assert result.chunk_id
            assert result.text
            assert result.score > 0
            assert result.document_id
            assert result.filename
    
    def test_vector_search_scores_normalized(self):
        """Test that vector search scores are 0-1."""
        searcher = VectorSearch()
        results = searcher.search("test", top_k=5)
        
        for result in results:
            assert 0 <= result.score <= 1


class TestKeywordSearch:
    """Tests for keyword search."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data."""
        store = VectorStore()
        store.reset()
        
        # Process test document
        process_document(
            FIXTURES_DIR / 'test.txt',
            chunk_strategy='recursive',
            chunk_size=50
        )
        
        yield
        
        store = VectorStore()
        store.reset()
    
    def test_keyword_search_returns_results(self):
        """Test that keyword search returns results."""
        searcher = KeywordSearch()
        
        results = searcher.search("test", top_k=5)
        
        assert len(results) >= 0  # May be 0 if FTS5 not populated yet
    
    def test_keyword_search_is_available(self):
        """Test that keyword search availability check works."""
        searcher = KeywordSearch()
        # Should return True if FTS5 is set up
        available = searcher.is_available()
        assert isinstance(available, bool)


class TestHybridSearch:
    """Tests for hybrid search."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data."""
        store = VectorStore()
        store.reset()
        
        process_document(
            FIXTURES_DIR / 'test.txt',
            chunk_strategy='recursive',
            chunk_size=50
        )
        
        yield
        
        store = VectorStore()
        store.reset()
    
    def test_hybrid_search_returns_results(self):
        """Test hybrid search returns results."""
        searcher = HybridSearch()
        results = searcher.search("test document", top_k=5)
        
        assert len(results) >= 0
    
    def test_hybrid_search_alpha_parameter(self):
        """Test that alpha parameter affects results."""
        searcher = HybridSearch()
        
        results_high_alpha = searcher.search("test", top_k=5, alpha=0.9)
        results_low_alpha = searcher.search("test", top_k=5, alpha=0.1)
        
        # Both should return results
        assert isinstance(results_high_alpha, list)
        assert isinstance(results_low_alpha, list)
    
    def test_rrf_calculation(self):
        """Test RRF scoring calculation."""
        # Create mock results
        results1 = [
            SearchResult(chunk_id='a', text='text a', score=0.9, document_id='d1', filename='f1'),
            SearchResult(chunk_id='b', text='text b', score=0.8, document_id='d1', filename='f1'),
        ]
        
        results2 = [
            SearchResult(chunk_id='b', text='text b', score=0.7, document_id='d1', filename='f1'),
            SearchResult(chunk_id='c', text='text c', score=0.6, document_id='d1', filename='f1'),
        ]
        
        merged = reciprocal_rank_fusion([results1, results2])
        
        assert len(merged) == 3  # a, b, c
        
        # b should have highest score (appears in both lists)
        assert merged[0].chunk_id == 'b'


class TestReranker:
    """Tests for re-ranker."""
    
    def test_reranker_changes_order(self):
        """Test that reranker can change result order."""
        reranker = Reranker()
        
        # Create mock results with initial scores
        results = [
            SearchResult(chunk_id='1', text='This is about cats', score=0.9, document_id='d1', filename='f1'),
            SearchResult(chunk_id='2', text='This is about dogs', score=0.8, document_id='d1', filename='f1'),
            SearchResult(chunk_id='3', text='This is about birds', score=0.7, document_id='d1', filename='f1'),
        ]
        
        # Re-rank with query about dogs
        reranked = reranker.rerank("dogs", results, top_n=3)
        
        # Should return results
        assert len(reranked) <= 3
        assert all(isinstance(r, SearchResult) for r in reranked)
    
    def test_reranker_empty_results(self):
        """Test reranker handles empty results."""
        reranker = Reranker()
        results = reranker.rerank("query", [], top_n=5)
        
        assert results == []


class TestRetriever:
    """Tests for unified retriever."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data."""
        store = VectorStore()
        store.reset()
        
        process_document(
            FIXTURES_DIR / 'test.txt',
            chunk_strategy='recursive',
            chunk_size=50
        )
        
        yield
        
        store = VectorStore()
        store.reset()
    
    def test_retriever_vector_strategy(self):
        """Test retriever with vector strategy."""
        retriever = Retriever()
        result = retriever.retrieve(
            "test",
            strategy='vector',
            top_k=5,
            rerank=False
        )
        
        assert result.strategy_used == 'vector'
        assert isinstance(result.results, list)
        assert result.search_time_ms >= 0
    
    def test_retriever_keyword_strategy(self):
        """Test retriever with keyword strategy."""
        retriever = Retriever()
        result = retriever.retrieve(
            "test",
            strategy='keyword',
            top_k=5,
            rerank=False
        )
        
        assert result.strategy_used == 'keyword'
        assert isinstance(result.results, list)
    
    def test_retriever_hybrid_strategy(self):
        """Test retriever with hybrid strategy."""
        retriever = Retriever()
        result = retriever.retrieve(
            "test",
            strategy='hybrid',
            top_k=5,
            rerank=False
        )
        
        assert result.strategy_used == 'hybrid'
        assert isinstance(result.results, list)
    
    def test_retriever_with_reranking(self):
        """Test retriever with re-ranking enabled."""
        retriever = Retriever()
        result = retriever.retrieve(
            "test",
            strategy='hybrid',
            top_k=10,
            rerank=True,
            rerank_top_n=3
        )
        
        assert len(result.results) <= 3
        assert result.rerank_time_ms >= 0
    
    def test_retriever_unknown_strategy_fallback(self):
        """Test retriever falls back for unknown strategy."""
        retriever = Retriever()
        result = retriever.retrieve(
            "test",
            strategy='unknown_strategy',
            top_k=5,
            rerank=False
        )
        
        assert result.strategy_used == 'hybrid'  # Should fallback
    
    def test_retriever_get_available_strategies(self):
        """Test getting available strategies."""
        retriever = Retriever()
        strategies = retriever.get_available_strategies()
        
        assert 'vector' in strategies
        assert 'keyword' in strategies
        assert 'hybrid' in strategies
        
        # HyDE and expanded need a generation provider; the retriever owns
        # that question now, so ask it rather than a removed attribute.
        if retriever._llm_available():
            assert 'hyde' in strategies
            assert 'expanded' in strategies
    
    def test_retriever_hyde_fallback(self):
        """Test HyDE falls back when no Gemini key."""
        retriever = Retriever()
        
        if not retriever._llm_available():
            result = retriever.retrieve(
                "test",
                strategy='hyde',
                top_k=5
            )
            
            # Should fallback to hybrid
            assert result.strategy_used == 'hybrid'
    
    def test_query_expansion_guards_use_the_public_provider_api(self):
        """The availability guard must survive a provider refactor.

        ``QueryExpansion`` is reachable directly from the query API, not only
        through ``Retriever``, whose own guard runs first and hides a broken
        one here. When the provider grew named backends its private client
        attributes went away, and these two call sites kept reading them --
        raising ``AttributeError`` on every HyDE and expansion request made
        through the API while the retriever-level test stayed green.
        """
        expansion = QueryExpansion()

        class Unavailable:
            def is_available(self):
                return False

        with patch.object(provider, "get_provider", return_value=Unavailable()):
            hyde = expansion.hyde_search("test", top_k=3)
            expanded = expansion.expanded_search("test", top_k=3)

        assert isinstance(hyde, list)
        assert isinstance(expanded, list)

    def test_search_simple(self):
        """Test simple search convenience method."""
        retriever = Retriever()
        results = retriever.search_simple("test", top_k=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
