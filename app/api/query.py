"""Query API for document retrieval."""
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.retrieval.retriever import Retriever
from app.retrieval.types import RetrievalResult


router = APIRouter()
retriever = Retriever()


class QueryRequest(BaseModel):
    """Request for document query."""
    query: str = Field(..., min_length=3, max_length=1000, description="Search query (3-1000 characters)")
    strategy: str = Field('hybrid', description="Search strategy: vector, keyword, hybrid, hyde, expanded")
    top_k: int = Field(20, ge=1, le=100, description="Number of results")
    rerank: bool = Field(True, description="Apply cross-encoder re-ranking")
    alpha: float = Field(0.7, ge=0, le=1, description="Weight for vector search in hybrid mode")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "query": "What is machine learning?",
                "strategy": "hybrid",
                "top_k": 10,
                "rerank": True,
                "alpha": 0.7
            }]
        }
    }


class DocumentInfo(BaseModel):
    """Document information in result."""
    id: str
    filename: str
    title: Optional[str] = None


class ResultMetadata(BaseModel):
    """Metadata for search result."""
    page: Optional[int] = None
    heading: Optional[str] = None


class SearchResultItem(BaseModel):
    """Individual search result."""
    chunk_id: str
    text: str
    score: float
    document: DocumentInfo
    metadata: ResultMetadata


class QueryResponse(BaseModel):
    """Response for document query."""
    results: list[SearchResultItem]
    strategy: str
    search_time_ms: float
    rerank_time_ms: float
    total_candidates: int


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Search for relevant document chunks.
    
    Args:
        request: Query parameters
        
    Returns:
        Search results with relevance scores
    """
    # Clean and validate query
    query = request.query.strip()
    if not query or len(query) < 3:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 3 non-whitespace characters"
        )
    
    # Validate strategy
    valid_strategies = retriever.get_available_strategies()
    if request.strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy. Valid strategies: {valid_strategies}"
        )
    
    # Execute search
    result = retriever.retrieve(
        query=request.query,
        strategy=request.strategy,
        top_k=request.top_k,
        rerank=request.rerank,
        rerank_top_n=request.top_k,
        alpha=request.alpha
    )
    
    # Format response
    results = []
    for r in result.results:
        results.append(SearchResultItem(
            chunk_id=r.chunk_id,
            text=r.text,
            score=r.score,
            document=DocumentInfo(
                id=r.document_id,
                filename=r.filename,
                title=r.title
            ),
            metadata=ResultMetadata(
                page=r.page_number,
                heading=r.heading_context
            )
        ))
    
    return QueryResponse(
        results=results,
        strategy=result.strategy_used,
        search_time_ms=result.search_time_ms,
        rerank_time_ms=result.rerank_time_ms,
        total_candidates=result.total_candidates_before_rerank
    )
