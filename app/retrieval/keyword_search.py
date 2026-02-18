"""Keyword search using SQLite FTS5."""
from sqlalchemy import text
from app.storage.database import DatabaseManager
from app.storage.models import Document
from app.retrieval.types import SearchResult
from typing import Optional
import logging


logger = logging.getLogger(__name__)


class KeywordSearch:
    """Full-text keyword search using SQLite FTS5."""

    STOP_WORDS = {
        "the", "is", "a", "an", "in", "on", "of", "for", "to", "and", "or",
        "how", "did", "what", "were", "does", "are", "was", "has", "been",
        "being", "with", "from", "this", "that", "its", "can", "will",
    }
    
    def search(
        self,
        query: str,
        top_k: int = 20
    ) -> list[SearchResult]:
        """Search for chunks using full-text search.
        
        Uses SQLite FTS5 with BM25 ranking.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        results = []
        
        try:
            with DatabaseManager() as db:
                # FTS5 query with BM25 ranking
                # rank is a special FTS5 column for BM25 score (higher = more relevant)
                sql = text("""
                SELECT 
                    fts.id as chunk_id,
                    fts.document_id,
                    c.text,
                    c.page_number,
                    c.heading_context,
                    rank as bm25_score
                FROM chunks_fts fts
                JOIN chunks c ON fts.id = c.id
                WHERE chunks_fts MATCH :query
                ORDER BY rank
                LIMIT :limit
                """)
                
                # Aggressive sanitization for FTS5
                import re
                
                # 1. Strip "Information about:" prefix if present
                sanitized_query = re.sub(r'^Information about:\s*', '', query, flags=re.IGNORECASE).strip()
                
                # 2. Remove all punctuation except spaces and alphanumeric characters
                sanitized_query = re.sub(r"[^\w\s]", " ", sanitized_query)
                
                # 3. Normalize case and keep only meaningful words (>=3 chars, not stop words)
                words = []
                for token in sanitized_query.lower().split():
                    if len(token) < 3:
                        continue
                    if token in self.STOP_WORDS:
                        continue
                    words.append(token)
                
                if not words:
                    logger.warning(f"No meaningful keywords found in query: '{query}'")
                    return []
                
                # 4. Join with OR for FTS5 so chunks matching ANY term are returned
                # BM25 ranking will naturally score chunks with more matches higher
                sanitized_query = ' OR '.join(words)
                
                # 5. Add logging to show the sanitized query being sent to FTS5
                logger.info(f"FTS5 MATCH query: '{sanitized_query}'")
                
                # Execute search with named parameters
                rows = db.execute(sql, {"query": sanitized_query, "limit": top_k}).fetchall()
                
                if not rows:
                    return results
                
                # Get document metadata
                document_ids = set(row.document_id for row in rows)
                docs = db.query(Document).filter(Document.id.in_(document_ids)).all()
                doc_metadata = {d.id: {'filename': d.filename, 'title': d.title} for d in docs}
                
                # Create SearchResult objects
                for row in rows:
                    # Convert BM25 score to normalized 0-1 score
                    # BM25 score can be negative, so we normalize
                    # Higher (less negative) BM25 = more relevant
                    bm25_score = row.bm25_score or 0
                    # Normalize to 0-1 range (rough approximation)
                    # BM25 scores are typically -10 to 0, higher is better
                    normalized_score = min(1.0, max(0.0, 1.0 + bm25_score / 10))
                    
                    doc_info = doc_metadata.get(row.document_id, {'filename': '', 'title': None})
                    
                    result = SearchResult(
                        chunk_id=row.chunk_id,
                        text=row.text,
                        score=normalized_score,
                        document_id=row.document_id,
                        filename=doc_info['filename'],
                        title=doc_info['title'],
                        page_number=row.page_number,
                        heading_context=row.heading_context
                    )
                    
                    results.append(result)
                
                logger.info(f"Keyword search found {len(results)} results for query: {query[:50]}...")
                
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            # Return empty results on error rather than crashing
        
        return results
    
    def is_available(self) -> bool:
        """Check if FTS5 table exists and is populated.
        
        Returns:
            True if FTS5 is available and has data
        """
        try:
            with DatabaseManager() as db:
                # Check if table exists
                result = db.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
                ).fetchone()
                
                if not result:
                    return False
                
                # Check if it has data
                count = db.execute(text("SELECT COUNT(*) FROM chunks_fts")).fetchone()[0]
                return count > 0
        except Exception:
            return False
