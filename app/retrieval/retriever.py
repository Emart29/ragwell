"""Unified retriever - single entry point for all retrieval strategies."""
from app.retrieval.vector_search import VectorSearch
from app.retrieval.keyword_search import KeywordSearch
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.query_expand import QueryExpansion
from app.retrieval.reranker import Reranker
from app.retrieval.types import SearchResult, RetrievalResult
from app.llm.provider import get_provider
import time
from app.logging_config import get_logger

logger = get_logger(__name__)


class Retriever:
    """Unified retrieval interface supporting multiple strategies."""
    
    STRATEGIES = ['vector', 'keyword', 'hybrid', 'hyde', 'expanded']
    
    def __init__(self):
        """Initialize retriever with all search components."""
        self.vector_search = VectorSearch()
        self.keyword_search = KeywordSearch()
        self.hybrid_search = HybridSearch()
        self.query_expansion = QueryExpansion()
        self.reranker = Reranker()
        
        logger.info("Retriever initialized with strategies: %s", self.STRATEGIES)
    
    def retrieve(
        self,
        query: str,
        strategy: str = 'hybrid',
        top_k: int = 20,
        rerank: bool = True,
        rerank_top_n: int = 5,
        alpha: float = 0.7
    ) -> RetrievalResult:
        """Retrieve relevant chunks using specified strategy.
        
        Args:
            query: Search query
            strategy: Search strategy - 'vector', 'keyword', 'hybrid', 'hyde', 'expanded'
            top_k: Number of results to retrieve
            rerank: Whether to apply cross-encoder re-ranking
            rerank_top_n: Number of results to return after re-ranking
            alpha: Weight for vector search in hybrid mode (0-1)
            
        Returns:
            RetrievalResult with results and metadata
        """
        start_time = time.time()
        
        # Validate strategy
        if strategy not in self.STRATEGIES:
            logger.warning(f"Unknown strategy '{strategy}', falling back to 'hybrid'")
            strategy = 'hybrid'
        
        # Handle HyDE/expanded fallback when no LLM provider is available
        if strategy in ('hyde', 'expanded') and not self._llm_available():
            logger.warning(f"{strategy} unavailable (no LLM provider), falling back to 'hybrid'")
            strategy = 'hybrid'
        
        logger.info(f"Retrieving with strategy='{strategy}', query='{query[:50]}...'")
        
        # Execute search based on strategy
        search_start = time.time()
        
        if strategy == 'vector':
            results = self.vector_search.search(query, top_k=top_k)
        elif strategy == 'keyword':
            results = self.keyword_search.search(query, top_k=top_k)
        elif strategy == 'hybrid':
            results = self.hybrid_search.search(query, top_k=top_k, alpha=alpha)
        elif strategy == 'hyde':
            results = self.query_expansion.hyde_search(query, top_k=top_k)
        elif strategy == 'expanded':
            results = self.query_expansion.expanded_search(query, top_k=top_k)
        else:
            results = []
        
        search_time = time.time() - search_start
        total_candidates = len(results)
        
        logger.info(f"Search found {total_candidates} candidates in {search_time:.2f}s")
        
        rerank_time = 0.0
        if rerank and results:
            rerank_start = time.time()
            
            # Re-rank more than top_n to have good candidates
            rerank_candidates = min(len(results), rerank_top_n * 3)
            results_to_rerank = results[:rerank_candidates]
            
            results = self.reranker.rerank(
                query=query,
                results=results_to_rerank,
                top_n=rerank_top_n
            )
            
            rerank_time = time.time() - rerank_start
        
        total_time = time.time() - start_time
        
        # Log latency breakdown
        logger.info(
            f"Retrieval latency breakdown: "
            f"search={search_time*1000:.1f}ms, "
            f"rerank={rerank_time*1000:.1f}ms, "
            f"total={total_time*1000:.1f}ms"
        )
        
        return RetrievalResult(
            results=results,
            strategy_used=strategy,
            search_time_ms=round(search_time * 1000, 2),
            rerank_time_ms=round(rerank_time * 1000, 2),
            total_candidates_before_rerank=total_candidates
        )
    
    def _llm_available(self) -> bool:
        """Check if an LLM provider (Groq or Gemini) is available."""
        try:
            return get_provider().is_available()
        except Exception:
            return False

    def get_available_strategies(self) -> list[str]:
        """Get list of available strategies.

        Returns:
            List of strategy names
        """
        strategies = ['vector', 'keyword', 'hybrid']

        if self._llm_available():
            strategies.extend(['hyde', 'expanded'])

        return strategies
    
    def search_simple(
        self,
        query: str,
        top_k: int = 5
    ) -> list[SearchResult]:
        """Simple search with sensible defaults.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        result = self.retrieve(
            query=query,
            strategy='hybrid',
            top_k=top_k * 2,  # Get more for re-ranking
            rerank=True,
            rerank_top_n=top_k
        )
        return result.results


# Convenience function
def retrieve(
    query: str,
    strategy: str = 'hybrid',
    top_k: int = 20,
    rerank: bool = True,
    rerank_top_n: int = 5,
    alpha: float = 0.7
) -> RetrievalResult:
    """Convenience function for retrieval.
    
    Args:
        query: Search query
        strategy: Search strategy
        top_k: Number of results
        rerank: Whether to re-rank
        rerank_top_n: Results after re-ranking
        alpha: Hybrid weight
        
    Returns:
        RetrievalResult
    """
    retriever = Retriever()
    return retriever.retrieve(
        query=query,
        strategy=strategy,
        top_k=top_k,
        rerank=rerank,
        rerank_top_n=rerank_top_n,
        alpha=alpha
    )
