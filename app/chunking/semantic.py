"""Semantic boundary-based text chunking."""
from dataclasses import dataclass
from typing import Optional
import re
import numpy as np
from app.chunking.recursive import Chunk, count_tokens
from app.parsers.base import ParsedDocument
from app.config import settings


@dataclass
class SemanticChunk(Chunk):
    """A semantic chunk with boundary information."""
    boundary_score: float = 0.0  # Similarity score at boundary


class SemanticChunker:
    """Semantic boundary-based text chunker.
    
    Uses sentence embeddings to detect topic shifts and split
    at semantic boundaries.
    """
    
    def __init__(
        self,
        max_tokens: int = None,
        min_tokens: int = 50,
        similarity_threshold: float = 0.3,
        model_name: str = None
    ):
        """Initialize the semantic chunker.
        
        Args:
            max_tokens: Maximum tokens per chunk
            min_tokens: Minimum tokens before allowing split
            similarity_threshold: Threshold below which to split
            model_name: Embedding model name
        """
        self.max_tokens = max_tokens if max_tokens is not None else settings.CHUNK_SIZE
        self.min_tokens = min_tokens
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name if model_name is not None else settings.EMBEDDING_MODEL
        
        # Shared embedder instance - lazy loaded
        self._embedder = None
    
    @property
    def embedder(self):
        """Lazy load embedder instance."""
        if self._embedder is None:
            from app.embeddings.embedder import Embedder
            self._embedder = Embedder.get_instance()
        return self._embedder
    
    def chunk_document(self, parsed_doc: ParsedDocument) -> list[Chunk]:
        """Chunk a parsed document using semantic boundaries.
        
        Args:
            parsed_doc: ParsedDocument from parser
            
        Returns:
            List of Chunk objects
        """
        full_text = parsed_doc.text
        
        # Split into sentences
        sentences = self._split_into_sentences(full_text)
        
        if len(sentences) <= 1:
            # Not enough sentences for semantic chunking
            return self._fallback_chunk(parsed_doc)
        
        # Get page boundaries and heading context
        page_boundaries = self._get_page_boundaries(parsed_doc)
        heading_map = self._get_heading_map(parsed_doc)
        
        # Calculate sentence boundaries (character positions)
        sentence_boundaries = self._get_sentence_boundaries(full_text, sentences)
        
        # Get embeddings for all sentences
        sentence_embeddings = self.embedder.embed_batch(sentences)
        
        # Calculate similarities between consecutive sentences
        similarities = self._calculate_similarities(sentence_embeddings)
        
        # Find split points
        split_indices = self._find_split_points(
            sentences,
            similarities,
            sentence_boundaries
        )
        
        # Create chunks
        chunks = []
        start_idx = 0
        
        for end_idx in split_indices:
            chunk_text = ' '.join(sentences[start_idx:end_idx])
            start_char = sentence_boundaries[start_idx][0]
            end_char = sentence_boundaries[end_idx - 1][1]
            
            token_count = count_tokens(chunk_text)
            source_page = self._get_page_for_position(start_char, page_boundaries)
            heading_context = self._get_heading_for_position(start_char, heading_map)
            
            chunk = Chunk(
                text=chunk_text,
                chunk_index=len(chunks),
                start_char=start_char,
                end_char=end_char,
                token_count=token_count,
                source_page=source_page,
                heading_context=heading_context
            )
            
            chunks.append(chunk)
            start_idx = end_idx
        
        # Add final chunk
        if start_idx < len(sentences):
            chunk_text = ' '.join(sentences[start_idx:])
            start_char = sentence_boundaries[start_idx][0]
            end_char = sentence_boundaries[-1][1]
            
            token_count = count_tokens(chunk_text)
            source_page = self._get_page_for_position(start_char, page_boundaries)
            heading_context = self._get_heading_for_position(start_char, heading_map)
            
            chunk = Chunk(
                text=chunk_text,
                chunk_index=len(chunks),
                start_char=start_char,
                end_char=end_char,
                token_count=token_count,
                source_page=source_page,
                heading_context=heading_context
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting on punctuation
        # Could be improved with spaCy or NLTK
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_sentence_boundaries(
        self,
        full_text: str,
        sentences: list[str]
    ) -> list[tuple[int, int]]:
        """Get character boundaries for each sentence.
        
        Args:
            full_text: Full document text
            sentences: List of sentences
            
        Returns:
            List of (start_char, end_char) tuples
        """
        boundaries = []
        current_pos = 0
        
        for sentence in sentences:
            start = full_text.find(sentence, current_pos)
            if start == -1:
                start = current_pos
            end = start + len(sentence)
            boundaries.append((start, end))
            current_pos = end
        
        return boundaries
    
    def _calculate_similarities(self, embeddings: list) -> list[float]:
        """Calculate cosine similarity between consecutive embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            List of similarity scores (one less than embeddings)
        """
        similarities = []
        
        for i in range(len(embeddings) - 1):
            emb1 = embeddings[i]
            emb2 = embeddings[i + 1]
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
            similarities.append(float(similarity))
        
        return similarities
    
    def _find_split_points(
        self,
        sentences: list[str],
        similarities: list[float],
        sentence_boundaries: list[tuple[int, int]]
    ) -> list[int]:
        """Find indices where to split based on similarity and token limits.
        
        Args:
            sentences: List of sentences
            similarities: Similarity scores between sentences
            sentence_boundaries: Character boundaries for sentences
            
        Returns:
            List of end indices for chunks
        """
        split_indices = []
        current_chunk_tokens = 0
        current_chunk_start = 0
        
        for i in range(len(sentences)):
            sentence = sentences[i]
            sentence_tokens = count_tokens(sentence)
            
            # Check if adding this sentence exceeds max tokens
            if current_chunk_tokens + sentence_tokens > self.max_tokens:
                if current_chunk_start < i:
                    split_indices.append(i)
                current_chunk_start = i
                current_chunk_tokens = sentence_tokens
                continue
            
            current_chunk_tokens += sentence_tokens
            
            # Check for semantic boundary (but respect min tokens)
            if i < len(similarities):
                similarity = similarities[i]
                
                # Check if we've reached minimum tokens
                if current_chunk_tokens >= self.min_tokens:
                    # Check for topic shift
                    if similarity < self.similarity_threshold:
                        split_indices.append(i + 1)
                        current_chunk_start = i + 1
                        current_chunk_tokens = 0
        
        return split_indices
    
    def _fallback_chunk(self, parsed_doc: ParsedDocument) -> list[Chunk]:
        """Fallback to recursive chunking when semantic fails.
        
        Args:
            parsed_doc: Parsed document
            
        Returns:
            Chunks from recursive chunker
        """
        from app.chunking.recursive import RecursiveChunker
        chunker = RecursiveChunker(chunk_size=self.max_tokens)
        return chunker.chunk_document(parsed_doc)
    
    def _get_page_boundaries(self, parsed_doc: ParsedDocument) -> list[tuple[int, int]]:
        """Calculate character boundaries for each page."""
        boundaries = []
        current_pos = 0
        
        for page_text in parsed_doc.pages:
            start = current_pos
            end = current_pos + len(page_text)
            boundaries.append((start, end))
            current_pos = end + 2
        
        return boundaries
    
    def _get_heading_map(self, parsed_doc: ParsedDocument) -> list[tuple[int, str]]:
        """Create a map of character positions to headings."""
        heading_map = []
        
        outline = parsed_doc.metadata.get('heading_outline', [])
        if outline:
            full_text = parsed_doc.text
            for heading in outline:
                heading_text = heading.get('text', '')
                if heading_text:
                    pos = full_text.find(heading_text)
                    if pos >= 0:
                        heading_map.append((pos, heading_text))
        
        heading_map.sort(key=lambda x: x[0])
        return heading_map
    
    def _get_page_for_position(
        self,
        char_pos: int,
        page_boundaries: list[tuple[int, int]]
    ) -> Optional[int]:
        """Determine which page a character position falls on."""
        for page_num, (start, end) in enumerate(page_boundaries, 1):
            if start <= char_pos < end:
                return page_num
        return None
    
    def _get_heading_for_position(
        self,
        char_pos: int,
        heading_map: list[tuple[int, str]]
    ) -> Optional[str]:
        """Get the last heading before a character position."""
        last_heading = None
        for pos, heading in heading_map:
            if pos <= char_pos:
                last_heading = heading
            else:
                break
        return last_heading
