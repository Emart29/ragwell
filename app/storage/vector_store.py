"""ChromaDB vector storage for embeddings."""
from pathlib import Path
from typing import Optional
import uuid
import chromadb
from chromadb.config import Settings
from app.config import settings
from app.storage.database import DatabaseManager
from app.storage.models import Document, Chunk
import logging


logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-based vector storage for document chunks.

    Uses a singleton ChromaDB PersistentClient to avoid directory-lock
    contention when multiple Pipeline instances are created (e.g. in the
    API's sync ``wait=true`` mode).
    """

    COLLECTION_NAME = 'ragwell_chunks'

    _client = None  # shared singleton

    def __init__(self):
        """Initialize ChromaDB connection (client is a singleton)."""
        if VectorStore._client is None:
            VectorStore._client = chromadb.PersistentClient(
                path=settings.CHROMA_PATH,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        self.client = VectorStore._client

        # Get or create collection
        self._get_collection()

        logger.info(f"VectorStore initialized with collection: {self.COLLECTION_NAME}")
    
    def _get_collection(self):
        """Get or refresh ChromaDB collection."""
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Ragwell document chunks"}
        )
        return self.collection
    
    def store_document(
        self,
        parsed_doc: dict,
        metadata: dict,
        filename: str,
        file_type: str,
        file_size: int
    ) -> str:
        """Store document record in SQLite.
        
        Args:
            parsed_doc: Parsed document data
            metadata: Document metadata
            filename: Original filename
            file_type: File extension/type
            file_size: File size in bytes
            
        Returns:
            Document ID (UUID string)
        """
        document_id = str(uuid.uuid4())
        
        with DatabaseManager() as db:
            doc = Document(
                id=document_id,
                filename=filename,
                file_type=file_type,
                file_size=file_size,
                page_count=metadata.get('page_count'),
                title=metadata.get('title'),
                author=metadata.get('author'),
                language=metadata.get('language'),
                processing_status='processing',
                metadata_json=metadata
            )
            
            db.add(doc)
            db.commit()
        
        logger.info(f"Document stored: {document_id} ({filename})")
        return document_id
    
    def store_chunks(
        self,
        document_id: str,
        chunks: list,
        embeddings: list[list[float]],
        quality_scores: list[float]
    ) -> list[str]:
        """Store chunks in SQLite and ChromaDB.
        
        Args:
            document_id: Parent document ID
            chunks: List of Chunk objects
            embeddings: List of embedding vectors
            quality_scores: List of quality scores
            
        Returns:
            List of chunk IDs
        """
        chunk_ids = []
        
        # Prepare data for ChromaDB
        chroma_ids = []
        chroma_embeddings = []
        chroma_documents = []
        chroma_metadatas = []
        
        with DatabaseManager() as db:
            for i, (chunk, embedding, quality) in enumerate(
                zip(chunks, embeddings, quality_scores)
            ):
                chunk_id = str(uuid.uuid4())
                chunk_ids.append(chunk_id)
                
                # Store in SQLite
                db_chunk = Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    chunk_strategy='recursive',  # Or detect from chunk type
                    page_number=chunk.source_page,
                    heading_context=chunk.heading_context,
                    quality_score=quality,
                )
                
                db.add(db_chunk)
                
                # Prepare for ChromaDB
                chroma_ids.append(chunk_id)
                chroma_embeddings.append(embedding)
                chroma_documents.append(chunk.text)
                chroma_metadatas.append({
                    'document_id': document_id,
                    'chunk_index': chunk.chunk_index,
                    'page': chunk.source_page if chunk.source_page is not None else 0,
                    'heading': chunk.heading_context or '',
                    'token_count': chunk.token_count,
                    'quality_score': quality
                })
            
            db.commit()
        
        # Store in ChromaDB in batches
        batch_size = 100
        for i in range(0, len(chroma_ids), batch_size):
            end = min(i + batch_size, len(chroma_ids))
            try:
                self.collection.add(
                    ids=chroma_ids[i:end],
                    embeddings=chroma_embeddings[i:end],
                    documents=chroma_documents[i:end],
                    metadatas=chroma_metadatas[i:end]
                )
            except Exception as e:
                # If collection is missing (e.g. after reset in another process), refresh and retry
                if "NotFoundError" in str(e) or "not exist" in str(e).lower():
                    self._get_collection()
                    self.collection.add(
                        ids=chroma_ids[i:end],
                        embeddings=chroma_embeddings[i:end],
                        documents=chroma_documents[i:end],
                        metadatas=chroma_metadatas[i:end]
                    )
                else:
                    raise e
        
        # FTS5 table is populated via database triggers synchronized on chunks table.
        # Manual population is redundant and can cause schema/query issues.
        
        # Update document status
        with DatabaseManager() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.processing_status = 'completed'
                db.commit()
        
        logger.info(f"Stored {len(chunks)} chunks for document {document_id}")
        return chunk_ids
    
    def _populate_fts5(self, document_id: str, chunks: list, chunk_ids: list):
        """Populate FTS5 table for keyword search.
        
        Args:
            document_id: Parent document ID
            chunks: List of Chunk objects
            chunk_ids: List of chunk IDs
        """
        from sqlalchemy import text
        
        with DatabaseManager() as db:
            # Get rowids for the newly inserted chunks
            for chunk_id, chunk in zip(chunk_ids, chunks):
                # Check if already in FTS5 (triggers should handle it)
                existing = db.execute(
                    text("SELECT 1 FROM chunks_fts WHERE chunk_id = :chunk_id"),
                    {"chunk_id": chunk_id}
                ).fetchone()
                
                if not existing:
                    # Get the rowid from chunks table
                    row = db.execute(
                        text("SELECT rowid FROM chunks WHERE id = :chunk_id"),
                        {"chunk_id": chunk_id}
                    ).fetchone()
                    
                    if row:
                        # Insert into FTS5
                        db.execute(
                            text("""
                                INSERT INTO chunks_fts(text, chunk_id, document_id, rowid)
                                VALUES (:text, :chunk_id, :doc_id, :rowid)
                            """),
                            {
                                "text": chunk.text,
                                "chunk_id": chunk_id,
                                "doc_id": document_id,
                                "rowid": row.rowid
                            }
                        )
            
            db.commit()
    
    def search(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        filter_dict: Optional[dict] = None
    ) -> dict:
        """Search for similar chunks."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_dict,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as e:
            if "NotFoundError" in str(e) or "not exist" in str(e).lower():
                self._get_collection()
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=filter_dict,
                    include=['documents', 'metadatas', 'distances']
                )
            else:
                raise e
        
        return results
    
    def delete_document(self, document_id: str):
        """Delete document and all its chunks.
        
        Args:
            document_id: Document ID to delete
        """
        # Delete from SQLite
        with DatabaseManager() as db:
            db.query(Chunk).filter(Chunk.document_id == document_id).delete()
            db.query(Document).filter(Document.id == document_id).delete()
            db.commit()
        
        # Delete from ChromaDB
        try:
            self.collection.delete(
                where={'document_id': document_id}
            )
        except Exception as e:
            if "NotFoundError" in str(e) or "not exist" in str(e).lower():
                self._get_collection()
                self.collection.delete(
                    where={'document_id': document_id}
                )
            else:
                raise e
        
        logger.info(f"Deleted document: {document_id}")
    
    def get_stats(self) -> dict:
        """Get storage statistics.
        
        Returns:
            Dictionary with stats
        """
        with DatabaseManager() as db:
            total_docs = db.query(Document).count()
            total_chunks = db.query(Chunk).count()
            
            # Calculate average quality
            chunks = db.query(Chunk).all()
            if chunks:
                avg_quality = sum(
                    c.quality_score for c in chunks if c.quality_score
                ) / len([c for c in chunks if c.quality_score])
            else:
                avg_quality = 0.0
        
        return {
            'total_documents': total_docs,
            'total_chunks': total_chunks,
            'average_quality_score': round(avg_quality, 3),
            'collection_name': self.COLLECTION_NAME
        }
    
    def reset(self):
        """Reset the vector store (delete all data)."""
        # Delete and recreate collection
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass # Ignore if already gone
            
        self._get_collection()
        
        # Delete from SQLite
        from app.storage.database import create_fts5_tables
        with DatabaseManager() as db:
            db.query(Chunk).delete()
            db.query(Document).delete()
            db.commit()
            
        # Recreate FTS5 tables and triggers
        create_fts5_tables()
        
        logger.info("Vector store reset")
