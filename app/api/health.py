"""Health API for system status."""
from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from app.storage.database import DatabaseManager, engine
from app.storage.models import Document, Chunk
from app.storage.vector_store import VectorStore
from app.embeddings.embedder import Embedder


router = APIRouter()


class CacheStats(BaseModel):
    """Embedding cache statistics."""
    hits: int
    misses: int
    hit_rate: float


class HealthResponse(BaseModel):
    """System health response."""
    status: str
    total_documents: int
    total_chunks: int
    avg_quality_score: float
    embedding_cache: CacheStats
    chromadb_collection_count: int
    sqlite_connected: bool


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Get system health and statistics.
    
    Returns:
        System health information
    """
    # Check SQLite connection
    sqlite_connected = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            sqlite_connected = True
    except Exception:
        pass
    
    # Get document stats
    with DatabaseManager() as db:
        total_docs = db.query(Document).count()
        total_chunks = db.query(Chunk).count()
        
        # Calculate average quality
        chunks = db.query(Chunk).all()
        if chunks:
            quality_chunks = [c for c in chunks if c.quality_score]
            avg_quality = sum(c.quality_score for c in quality_chunks) / len(quality_chunks) if quality_chunks else 0.0
        else:
            avg_quality = 0.0
    
    # Get vector store stats
    vector_store = VectorStore()
    collection_count = vector_store.collection.count()
    
    # Get cache stats
    try:
        embedder = Embedder.get_instance()
        cache_stats = embedder.get_cache_stats()
    except Exception:
        cache_stats = {'cache_hits': 0, 'cache_misses': 0, 'hit_rate': 0.0}
    
    return HealthResponse(
        status="healthy",
        total_documents=total_docs,
        total_chunks=total_chunks,
        avg_quality_score=round(avg_quality, 3),
        embedding_cache=CacheStats(
            hits=cache_stats['cache_hits'],
            misses=cache_stats['cache_misses'],
            hit_rate=cache_stats['hit_rate']
        ),
        chromadb_collection_count=collection_count,
        sqlite_connected=sqlite_connected
    )
