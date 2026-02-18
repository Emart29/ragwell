"""Recursive character-based text chunking."""
from dataclasses import dataclass
from typing import Optional
import tiktoken
from app.parsers.base import ParsedDocument
from app.config import settings


@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    source_page: Optional[int] = None
    heading_context: Optional[str] = None


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens using tiktoken.
    
    Args:
        text: Text to count tokens for
        model: Tiktoken model name (default: cl100k_base)
        
    Returns:
        Number of tokens
    """
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))


class RecursiveChunker:
    """Recursive character-based text chunker.
    
    Uses LangChain's RecursiveCharacterTextSplitter with tiktoken
    for accurate token counting.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        model_name: str = "cl100k_base"
    ):
        """Initialize the recursive chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            model_name: Tiktoken model for counting
        """
        self.chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
        self.model_name = model_name
        self._splitter = None  # Lazy loaded
    
    @property
    def splitter(self):
        """Lazy load the LangChain splitter."""
        if self._splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=lambda x: count_tokens(x, self.model_name),
                separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
            )
        return self._splitter
    
    def chunk_document(self, parsed_doc: ParsedDocument) -> list[Chunk]:
        """Chunk a parsed document.
        
        Args:
            parsed_doc: ParsedDocument from parser
            
        Returns:
            List of Chunk objects with metadata
        """
        chunks = []
        full_text = parsed_doc.text
        
        # Get page boundaries for tracking
        page_boundaries = self._get_page_boundaries(parsed_doc)
        
        # Get heading context map
        heading_map = self._get_heading_map(parsed_doc)
        
        # Split text using LangChain
        text_chunks = self.splitter.split_text(full_text)
        
        current_pos = 0
        for idx, text_chunk in enumerate(text_chunks):
            # Find position in original text
            start_char = full_text.find(text_chunk, current_pos)
            if start_char == -1:
                start_char = current_pos
            end_char = start_char + len(text_chunk)
            
            # Get source page
            source_page = self._get_page_for_position(start_char, page_boundaries)
            
            # Get heading context
            heading_context = self._get_heading_for_position(start_char, heading_map)
            
            # Count tokens
            token_count = count_tokens(text_chunk, self.model_name)
            
            chunk = Chunk(
                text=text_chunk,
                chunk_index=idx,
                start_char=start_char,
                end_char=end_char,
                token_count=token_count,
                source_page=source_page,
                heading_context=heading_context
            )
            
            chunks.append(chunk)
            current_pos = end_char - self.chunk_overlap
        
        return chunks
    
    def _get_page_boundaries(self, parsed_doc: ParsedDocument) -> list[tuple[int, int]]:
        """Calculate character boundaries for each page.
        
        Returns:
            List of (start_char, end_char) tuples for each page
        """
        boundaries = []
        current_pos = 0
        
        for page_text in parsed_doc.pages:
            start = current_pos
            end = current_pos + len(page_text)
            boundaries.append((start, end))
            current_pos = end + 2  # Account for '\n\n' separator
        
        return boundaries
    
    def _get_page_for_position(
        self,
        char_pos: int,
        page_boundaries: list[tuple[int, int]]
    ) -> Optional[int]:
        """Determine which page a character position falls on.
        
        Args:
            char_pos: Character position in full text
            page_boundaries: List of (start, end) for each page
            
        Returns:
            Page number (1-indexed) or None
        """
        for page_num, (start, end) in enumerate(page_boundaries, 1):
            if start <= char_pos < end:
                return page_num
        return None
    
    def _get_heading_map(self, parsed_doc: ParsedDocument) -> list[tuple[int, str]]:
        """Create a map of character positions to headings.
        
        Returns:
            List of (char_position, heading_text) tuples
        """
        heading_map = []
        
        # Try to get from metadata
        outline = parsed_doc.metadata.get('heading_outline', [])
        if outline:
            full_text = parsed_doc.text
            for heading in outline:
                heading_text = heading.get('text', '')
                if heading_text:
                    pos = full_text.find(heading_text)
                    if pos >= 0:
                        heading_map.append((pos, heading_text))
        
        # Sort by position
        heading_map.sort(key=lambda x: x[0])
        return heading_map
    
    def _get_heading_for_position(
        self,
        char_pos: int,
        heading_map: list[tuple[int, str]]
    ) -> Optional[str]:
        """Get the last heading before a character position.
        
        Args:
            char_pos: Character position
            heading_map: List of (position, heading) tuples
            
        Returns:
            Most recent heading text or None
        """
        last_heading = None
        for pos, heading in heading_map:
            if pos <= char_pos:
                last_heading = heading
            else:
                break
        return last_heading
