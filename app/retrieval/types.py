"""Shared types for retrieval layer."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    """Search result from any retrieval strategy."""
    chunk_id: str
    text: str
    score: float
    document_id: str
    filename: str
    title: Optional[str] = None
    page_number: Optional[int] = None
    heading_context: Optional[str] = None
    
    def __hash__(self):
        """Make SearchResult hashable for deduplication."""
        return hash(self.chunk_id)
    
    def __eq__(self, other):
        """Equality based on chunk_id for deduplication."""
        if not isinstance(other, SearchResult):
            return False
        return self.chunk_id == other.chunk_id


@dataclass 
class RetrievalResult:
    """Complete retrieval result with metadata."""
    results: list[SearchResult]
    strategy_used: str
    search_time_ms: float
    rerank_time_ms: float
    total_candidates_before_rerank: int
