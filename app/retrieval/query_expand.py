"""Query expansion using LLM (Groq or Gemini)."""
from app.embeddings.embedder import Embedder
from app.retrieval.vector_search import VectorSearch
from app.retrieval.hybrid_search import reciprocal_rank_fusion
from app.retrieval.types import SearchResult
from app.config import settings
from app.llm import provider
from typing import Optional
import logging


from app.logging_config import get_logger

logger = get_logger(__name__)


class QueryExpansion:
    """Query expansion using LLM (Google Gemini) for better retrieval.
    
    Provides HyDE (Hypothetical Document Embeddings) and multi-query
    expansion. Falls back gracefully if no API key is configured.
    """
    
    def __init__(self):
        """Initialize query expansion."""
        self.vector_search = VectorSearch()
        self.embedder = Embedder.get_instance()
    
    
    def hyde_search(
        self,
        query: str,
        top_k: int = 20
    ) -> list[SearchResult]:
        """HyDE (Hypothetical Document Embeddings) search.
        
        1. Generate a hypothetical document that would answer the query
        2. Embed that document
        3. Search using that embedding
        
        Args:
            query: Original query
            top_k: Number of results
            
        Returns:
            Search results
        """
        if not provider.get_provider().groq_client and not provider.get_provider().gemini_model:
            logger.warning("LLM providers unavailable, falling back to standard vector search")
            return self.vector_search.search(query, top_k=top_k)
        
        try:
            # Generate hypothetical document
            hypothetical_doc = self._generate_hypothetical_document(query)
            logger.info(f"HyDE generated: {hypothetical_doc[:100]}...")
            
            # Embed the hypothetical document
            hyde_embedding = self.embedder.embed(hypothetical_doc)
            
            # Search with HyDE embedding
            from app.storage.vector_store import VectorStore
            store = VectorStore()
            
            chroma_results = store.search(hyde_embedding, n_results=top_k)
            
            # Process results (similar to vector_search.py)
            results = self._process_chroma_results(chroma_results)
            
            logger.info(f"HyDE search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"HyDE search failed: {e}, falling back to standard search")
            return self.vector_search.search(query, top_k=top_k)
    
    def _generate_hypothetical_document(self, query: str) -> str:
        """Generate hypothetical document using LLM.
        
        Args:
            query: User query
            
        Returns:
            Generated text that would answer the query
        """
        prompt = f"""Write a short paragraph (2-3 sentences) that would answer the following question:

Question: {query}

Write the answer as if it were a passage from a relevant document:"""
        
        hypothetical_doc = provider.generate(prompt)
        if not hypothetical_doc:
            logger.warning("LLM HyDE generation failed")
            raise ValueError("LLM HyDE generation failed")
            
        return hypothetical_doc
    
    def expanded_search(
        self,
        query: str,
        top_k: int = 20
    ) -> list[SearchResult]:
        """Multi-query expansion search.
        
        1. Generate 3 alternative phrasings of the query
        2. Run vector search for each
        3. Merge results using RRF
        
        Args:
            query: Original query
            top_k: Number of results
            
        Returns:
            Merged search results
        """
        if not provider.get_provider().groq_client and not provider.get_provider().gemini_model:
            logger.warning("LLM providers unavailable, falling back to standard vector search")
            return self.vector_search.search(query, top_k=top_k)
        
        try:
            # Generate query variations
            variations = self._generate_query_variations(query)
            logger.info(f"Generated {len(variations)} query variations")
            
            # Search each variation
            all_results = []
            for var in variations:
                results = self.vector_search.search(var, top_k=top_k)
                all_results.append(results)
            
            # Also include original query
            all_results.append(self.vector_search.search(query, top_k=top_k))
            
            # Merge using RRF
            merged = reciprocal_rank_fusion(all_results)
            
            logger.info(f"Expanded search found {len(merged)} results")
            return merged[:top_k]
            
        except Exception as e:
            logger.error(f"Expanded search failed: {e}, falling back to standard search")
            return self.vector_search.search(query, top_k=top_k)
    
    def _generate_query_variations(self, query: str) -> list[str]:
        """Generate alternative phrasings of the query using LLM.
        
        Args:
            query: Original query
            
        Returns:
            List of alternative phrasings
        """
        prompt = f"""Given the search query: "{query}"

Generate 3 alternative ways to phrase this query that might retrieve similar documents.
Each should be different but capture the same intent.

Return exactly 3 variations, one per line, no numbering:"""
        
        response_text = provider.generate(prompt)
        if not response_text:
            logger.warning("LLM query variations generation failed")
            raise ValueError("LLM query variations generation failed")
        
        # Parse variations
        variations = []
        for line in response_text.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('-'):
                # Remove leading numbers/bullets
                if '. ' in line[:5]:
                    line = line.split('. ', 1)[1]
                variations.append(line)
        
        return variations[:3]  # Ensure max 3
    
    def _process_chroma_results(self, chroma_results: dict) -> list[SearchResult]:
        """Process ChromaDB results into SearchResult objects.
        
        Args:
            chroma_results: Raw ChromaDB results
            
        Returns:
            List of SearchResult objects
        """
        from app.storage.database import DatabaseManager
        from app.storage.models import Document
        
        results = []
        
        if not chroma_results or not chroma_results['ids']:
            return results
        
        ids = chroma_results['ids'][0] if chroma_results['ids'] else []
        distances = chroma_results.get('distances', [[]])[0] if chroma_results.get('distances') else []
        documents = chroma_results.get('documents', [[]])[0] if chroma_results.get('documents') else []
        metadatas = chroma_results.get('metadatas', [[]])[0] if chroma_results.get('metadatas') else []
        
        # Get document metadata
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
        
        return results
