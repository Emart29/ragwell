"""Documents API for document management."""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text

from app.storage.database import DatabaseManager
from app.storage.models import Document, Chunk
from app.storage.vector_store import VectorStore


router = APIRouter()


class DocumentListItem(BaseModel):
    """Document item in list."""
    id: str
    filename: str
    file_type: str
    file_size: int
    page_count: Optional[int]
    title: Optional[str]
    author: Optional[str]
    processing_status: str
    upload_time: str
    chunk_count: Optional[int] = None


class DocumentDetail(BaseModel):
    """Detailed document information."""
    id: str
    filename: str
    file_type: str
    file_size: int
    page_count: Optional[int]
    title: Optional[str]
    author: Optional[str]
    language: Optional[str]
    upload_time: str
    processing_status: str
    processing_error: Optional[str]
    metadata: Optional[dict]


class ChunkItem(BaseModel):
    """Chunk item in list."""
    id: str
    chunk_index: int
    text: str
    token_count: Optional[int]
    page_number: Optional[int]
    heading_context: Optional[str]
    quality_score: Optional[float]


class DocumentListResponse(BaseModel):
    """Response for document list."""
    documents: list[DocumentListItem]
    total: int
    page: int
    per_page: int
    pages: int


class ChunkListResponse(BaseModel):
    """Response for chunk list."""
    chunks: list[ChunkItem]
    total: int
    page: int
    per_page: int
    pages: int


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    file_type: Optional[str] = None
):
    """List all documents with pagination and filtering.
    
    Args:
        page: Page number (1-indexed)
        per_page: Items per page
        status: Filter by processing status
        file_type: Filter by file type
        
    Returns:
        Paginated list of documents
    """
    with DatabaseManager() as db:
        # Build query
        query = db.query(Document)
        
        if status:
            query = query.filter(Document.processing_status == status)
        
        if file_type:
            query = query.filter(Document.file_type == file_type)
        
        # Get total count
        total = query.count()
        
        # Get paginated results
        offset = (page - 1) * per_page
        docs = query.order_by(Document.upload_time.desc()).offset(offset).limit(per_page).all()
        
        # Format response with chunk counts
        documents = []
        for doc in docs:
            # Get chunk count for this document
            chunk_count = db.query(Chunk).filter(Chunk.document_id == doc.id).count() if doc.processing_status == "completed" else None
            
            documents.append(DocumentListItem(
                id=doc.id,
                filename=doc.filename,
                file_type=doc.file_type,
                file_size=doc.file_size,
                page_count=doc.page_count,
                title=doc.title,
                author=doc.author,
                processing_status=doc.processing_status,
                upload_time=doc.upload_time.isoformat() if doc.upload_time else None,
                chunk_count=chunk_count
            ))
        
        pages = (total + per_page - 1) // per_page
        
        return DocumentListResponse(
            documents=documents,
            total=total,
            page=page,
            per_page=per_page,
            pages=pages
        )


@router.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: str):
    """Get detailed information about a document.
    
    Args:
        document_id: Document ID
        
    Returns:
        Document details
    """
    with DatabaseManager() as db:
        doc = db.query(Document).filter(Document.id == document_id).first()
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentDetail(
            id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            file_size=doc.file_size,
            page_count=doc.page_count,
            title=doc.title,
            author=doc.author,
            language=doc.language,
            upload_time=doc.upload_time.isoformat() if doc.upload_time else None,
            processing_status=doc.processing_status,
            processing_error=doc.processing_error,
            metadata=doc.metadata_json
        )


@router.get("/documents/{document_id}/chunks", response_model=ChunkListResponse)
async def get_document_chunks(
    document_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100)
):
    """Get all chunks for a document.
    
    Args:
        document_id: Document ID
        page: Page number
        per_page: Items per page
        
    Returns:
        Paginated list of chunks
    """
    with DatabaseManager() as db:
        # Check document exists
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get total count
        total = db.query(Chunk).filter(Chunk.document_id == document_id).count()
        
        # Get paginated chunks
        offset = (page - 1) * per_page
        chunks = db.query(Chunk).filter(
            Chunk.document_id == document_id
        ).order_by(Chunk.chunk_index).offset(offset).limit(per_page).all()
        
        # Format response
        chunk_items = []
        for chunk in chunks:
            chunk_items.append(ChunkItem(
                id=chunk.id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                token_count=chunk.token_count,
                page_number=chunk.page_number,
                heading_context=chunk.heading_context,
                quality_score=chunk.quality_score
            ))
        
        pages = (total + per_page - 1) // per_page
        
        return ChunkListResponse(
            chunks=chunk_items,
            total=total,
            page=page,
            per_page=per_page,
            pages=pages
        )


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks.
    
    Args:
        document_id: Document ID to delete
        
    Returns:
        Success message
    """
    with DatabaseManager() as db:
        # Check document exists
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from vector store (ChromaDB + SQLite)
        vector_store = VectorStore()
        vector_store.delete_document(document_id)
        
        return {"message": "Document deleted successfully", "document_id": document_id}
