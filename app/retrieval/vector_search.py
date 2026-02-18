"""Vector search using ChromaDB and embeddings."""
from app.embeddings.embedder import Embedder
from app.storage.vector_store import VectorStore
from app.storage.database import DatabaseManager
from app.storage.models import Document, Chunk
from app.retrieval.types import SearchResult
from typing import Optional
import logging


logger = logging.getLogger(__name__)


class VectorSearch:
    """Vector-based semantic search using ChromaDB."""
    
    def __init__(self):
        """Initialize vector search."""
        self.embedder = Embedder.get_instance()
        self.vector_store = VectorStore()
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_dict: Optional[dict] = None
    ) -> list[SearchResult]:
        """Search for similar chunks using vector similarity.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of SearchResult objects
        """
        # Embed the query using retrieval.query task for asymmetric search
        query_embedding = self.embedder.embed_query(query)
        
        # Query ChromaDB
        chroma_results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=top_k,
            filter_dict=filter_dict
        )
        
        # Process results
        results = []
        
        if not chroma_results or not chroma_results['ids']:
            return results
        
        # Extract results from ChromaDB response
        ids = chroma_results['ids'][0] if chroma_results['ids'] else []
        distances = chroma_results.get('distances', [[]])[0] if chroma_results.get('distances') else []
        documents = chroma_results.get('documents', [[]])[0] if chroma_results.get('documents') else []
        metadatas = chroma_results.get('metadatas', [[]])[0] if chroma_results.get('metadatas') else []
        
        # Get document metadata from SQLite
        document_ids = set()
        for metadata in metadatas:
            if metadata and 'document_id' in metadata:
                document_ids.add(metadata['document_id'])
        
        doc_metadata = {}
        if document_ids:
            with DatabaseManager() as db:
                docs = db.query(Document).filter(Document.id.in_(document_ids)).all()
                doc_metadata = {d.id: {'filename': d.filename, 'title': d.title} for d in docs}
        
        # Create SearchResult objects
        for i, chunk_id in enumerate(ids):
            if i >= len(distances):
                break
                
            # Convert distance to similarity score
            # ChromaDB uses cosine distance, so score = 1 - distance
            distance = distances[i] if i < len(distances) else 1.0
            score = max(0.0, 1.0 - distance)
            
            text = documents[i] if i < len(documents) else ""
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            document_id = metadata.get('document_id', '')
            doc_info = doc_metadata.get(document_id, {'filename': '', 'title': None})
            
            result = SearchResult(
                chunk_id=chunk_id,
                text=text,
                score=score,
                document_id=document_id,
                filename=doc_info['filename'],
                title=doc_info['title'],
                page_number=metadata.get('page'),
                heading_context=metadata.get('heading')
            )
            
            results.append(result)
        
        logger.info(f"Vector search found {len(results)} results for query: {query[:50]}...")
        return results
