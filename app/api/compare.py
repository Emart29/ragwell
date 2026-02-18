"""Chunking comparison API."""
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.storage.database import DatabaseManager
from app.storage.models import Document
from app.parsers import parse_file
from app.chunking.comparator import ChunkingComparator


router = APIRouter()


class ChunkDistribution(BaseModel):
    """Chunk size distribution."""
    bucket: str
    percentage: float


class ComparisonResult(BaseModel):
    """Chunking comparison result."""
    # Recursive stats
    recursive_chunk_count: int
    recursive_avg_size: float
    recursive_min_size: int
    recursive_max_size: int
    recursive_distribution: dict[str, float]
    recursive_sample_chunks: list[str]
    
    # Semantic stats
    semantic_chunk_count: int
    semantic_avg_size: float
    semantic_min_size: int
    semantic_max_size: int
    semantic_distribution: dict[str, float]
    semantic_sample_chunks: list[str]
    
    # Comparison
    size_difference_percent: float
    recommendation: str


@router.post("/compare/{document_id}", response_model=ComparisonResult)
async def compare_chunking(document_id: str):
    """Compare recursive vs semantic chunking strategies for a document.
    
    Args:
        document_id: Document ID to analyze
        
    Returns:
        Comparison of both chunking strategies
    """
    # Get document info
    with DatabaseManager() as db:
        doc = db.query(Document).filter(Document.id == document_id).first()
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # For comparison, we need to re-parse the original file
        # Check if we have an uploads directory with the file
        from app.config import settings
        import os
        
        upload_dir = Path("./data/uploads")
        
        # Try to find the file
        file_path = None
        for ext in ['.pdf', '.docx', '.txt', '.html', '.csv']:
            potential_path = upload_dir / f"{document_id}{ext}"
            if potential_path.exists():
                file_path = potential_path
                break
        
        if not file_path:
            raise HTTPException(
                status_code=404, 
                detail="Original file not found for comparison. File may have been cleaned up after processing."
            )
    
    try:
        # Re-parse the document
        parsed_doc = parse_file(file_path)
        
        # Run comparison
        comparator = ChunkingComparator(chunk_size=512, chunk_overlap=50)
        comparison = comparator.compare(parsed_doc)
        
        return ComparisonResult(
            recursive_chunk_count=comparison.recursive_chunk_count,
            recursive_avg_size=comparison.recursive_avg_size,
            recursive_min_size=comparison.recursive_min_size,
            recursive_max_size=comparison.recursive_max_size,
            recursive_distribution=comparison.recursive_distribution,
            recursive_sample_chunks=comparison.recursive_sample_chunks,
            semantic_chunk_count=comparison.semantic_chunk_count,
            semantic_avg_size=comparison.semantic_avg_size,
            semantic_min_size=comparison.semantic_min_size,
            semantic_max_size=comparison.semantic_max_size,
            semantic_distribution=comparison.semantic_distribution,
            semantic_sample_chunks=comparison.semantic_sample_chunks,
            size_difference_percent=comparison.size_difference_percent,
            recommendation=comparison.recommendation
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )
