"""Hybrid search combining vector and keyword search with RRF."""
from app.retrieval.vector_search import VectorSearch
from app.retrieval.keyword_search import KeywordSearch
from app.retrieval.types import SearchResult
from typing import Optional
import logging


logger = logging.getLogger(__name__)


class HybridSearch:
    """Hybrid search using Reciprocal Rank Fusion (RRF).
    
    Combines vector semantic search with keyword (BM25) search
    using configurable alpha weighting.
    """
    
    RRF_K = 60  # RRF constant (standard value)
    
    def __init__(self):
        """Initialize hybrid search."""
        self.vector_search = VectorSearch()
        self.keyword_search = KeywordSearch()
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        alpha: float = 0.7
    ) -> list[SearchResult]:
        """Search using hybrid approach with RRF fusion.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for vector search (0-1), keyword gets (1-alpha)
            
        Returns:
            List of SearchResult objects sorted by RRF score
        """
        # Run both searches
        vector_results = self.vector_search.search(query, top_k=top_k * 2)
        keyword_results = self.keyword_search.search(query, top_k=top_k * 2)
        
        logger.info(
            f"Hybrid search: {len(vector_results)} vector, "
            f"{len(keyword_results)} keyword results"
        )
        
        # Check if we have any results
        if not vector_results and not keyword_results:
            return []
        
        # Deduplicate results by chunk_id
        all_results = {}
        
        for r in vector_results:
            all_results[r.chunk_id] = r
        
        for r in keyword_results:
            if r.chunk_id not in all_results:
                all_results[r.chunk_id] = r
        
        # Calculate RRF scores
        rrf_scores = self._calculate_rrf_scores(
            vector_results,
            keyword_results,
            alpha
        )
        
        # Apply RRF scores to results
        for chunk_id, rrf_score in rrf_scores.items():
            if chunk_id in all_results:
                all_results[chunk_id].score = rrf_score
        
        # Sort by RRF score (descending)
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        # Return top_k
        return sorted_results[:top_k]
    
    def _calculate_rrf_scores(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[SearchResult],
        alpha: float
    ) -> dict[str, float]:
        """Calculate RRF scores for all results.
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            alpha: Weight for vector search (keyword gets 1-alpha)
            
        Returns:
            Dictionary mapping chunk_id to RRF score
        """
        rrf_scores = {}
        
        # Create rank maps
        vector_ranks = {r.chunk_id: i + 1 for i, r in enumerate(vector_results)}
        keyword_ranks = {r.chunk_id: i + 1 for i, r in enumerate(keyword_results)}
        
        # Get all unique chunk IDs
        all_chunk_ids = set(vector_ranks.keys()) | set(keyword_ranks.keys())
        
        # Calculate RRF score for each chunk
        for chunk_id in all_chunk_ids:
            score = 0.0
            
            # Vector component
            if chunk_id in vector_ranks:
                rank = vector_ranks[chunk_id]
                score += alpha * (1.0 / (self.RRF_K + rank))
            
            # Keyword component
            if chunk_id in keyword_ranks:
                rank = keyword_ranks[chunk_id]
                score += (1 - alpha) * (1.0 / (self.RRF_K + rank))
            
            rrf_scores[chunk_id] = score
        
        return rrf_scores


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60
) -> list[SearchResult]:
    """Standalone RRF function for merging multiple result lists.
    
    Args:
        result_lists: List of result lists to merge
        k: RRF constant (default 60)
        
    Returns:
        Merged and re-ranked results
    """
    # Deduplicate and collect all results
    all_results = {}
    rank_maps = []
    
    for results in result_lists:
        rank_map = {r.chunk_id: i + 1 for i, r in enumerate(results)}
        rank_maps.append(rank_map)
        
        for r in results:
            if r.chunk_id not in all_results:
                all_results[r.chunk_id] = r
    
    # Calculate RRF scores
    for chunk_id in all_results:
        rrf_score = sum(
            1.0 / (k + rank_map.get(chunk_id, float('inf')))
            for rank_map in rank_maps
        )
        all_results[chunk_id].score = rrf_score
    
    # Sort by RRF score
    return sorted(
        all_results.values(),
        key=lambda x: x.score,
        reverse=True
    )
