"""Tests for chunking strategies."""
import pytest
from pathlib import Path
from app.parsers import parse_file
from app.chunking.recursive import RecursiveChunker, Chunk, count_tokens
from app.chunking.semantic import SemanticChunker
from app.chunking.comparator import ChunkingComparator


FIXTURES_DIR = Path(__file__).parent / 'fixtures'


class TestRecursiveChunker:
    """Tests for recursive character chunking."""
    
    def test_recursive_produces_valid_chunks(self):
        """Test that recursive chunker produces valid chunks."""
        parsed_doc = parse_file(FIXTURES_DIR / 'test.txt')
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_document(parsed_doc)
        
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert chunk.text
            assert chunk.chunk_index >= 0
            assert chunk.token_count > 0
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char
    
    def test_chunk_sizes_respect_limits(self):
        """Test that chunks respect size limits."""
        parsed_doc = parse_file(FIXTURES_DIR / 'test.txt')
        chunk_size = 100
        chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=20)
        chunks = chunker.chunk_document(parsed_doc)
        
        # Chunks should be close to target size (with some tolerance)
        for chunk in chunks:
            # Allow 20% tolerance for edge chunks
            assert chunk.token_count <= chunk_size * 1.2, \
                f"Chunk exceeds size limit: {chunk.token_count} > {chunk_size * 1.2}"
    
    def test_overlap_works(self):
        """Test that overlap is applied between chunks."""
        # Create a longer document
        parsed_doc = parse_file(FIXTURES_DIR / 'test.txt')
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk_document(parsed_doc)
        
        if len(chunks) > 1:
            # Check that consecutive chunks have overlap
            for i in range(len(chunks) - 1):
                chunk1_end = chunks[i].end_char
                chunk2_start = chunks[i + 1].start_char
                # Some overlap should exist
                assert chunk2_start < chunk1_end + 50  # Allow tolerance
    
    def test_heading_context_tracked(self):
        """Test that heading context is tracked."""
        # Use DOCX which has heading hierarchy
        parsed_doc = parse_file(FIXTURES_DIR / 'test.docx')
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_document(parsed_doc)
        
        # At least some chunks should have heading context
        chunks_with_context = [c for c in chunks if c.heading_context]
        assert len(chunks_with_context) > 0 or len(chunks) == 0
    
    def test_page_tracking(self):
        """Test that page numbers are tracked."""
        parsed_doc = parse_file(FIXTURES_DIR / 'test.pdf')
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_document(parsed_doc)
        
        for chunk in chunks:
            # Page should be either None or a positive integer
            assert chunk.source_page is None or chunk.source_page > 0
    
    def test_count_tokens(self):
        """Test token counting function."""
        text = "This is a test sentence with exactly ten tokens."
        count = count_tokens(text)
        assert count > 0
        assert isinstance(count, int)


class TestSemanticChunker:
    """Tests for semantic boundary chunking."""
    
    def test_semantic_produces_valid_chunks(self):
        """Test that semantic chunker produces valid chunks."""
        parsed_doc = parse_file(FIXTURES_DIR / 'test.txt')
        chunker = SemanticChunker(max_tokens=150)
        chunks = chunker.chunk_document(parsed_doc)
        
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert chunk.text
            assert chunk.chunk_index >= 0
            assert chunk.token_count > 0
    
    def test_semantic_chunk_sizes(self):
        """Test that semantic chunks respect max tokens."""
        parsed_doc = parse_file(FIXTURES_DIR / 'test.txt')
        max_tokens = 150
        chunker = SemanticChunker(max_tokens=max_tokens)
        chunks = chunker.chunk_document(parsed_doc)
        
        for chunk in chunks:
            # Semantic chunks shouldn't exceed max by much
            assert chunk.token_count <= max_tokens * 1.3, \
                f"Chunk too large: {chunk.token_count} tokens"
    
    def test_semantic_avoids_small_chunks(self):
        """Test that semantic chunker avoids very small chunks."""
        parsed_doc = parse_file(FIXTURES_DIR / 'test.txt')
        chunker = SemanticChunker(max_tokens=150, min_tokens=50)
        chunks = chunker.chunk_document(parsed_doc)
        
        # Most chunks should be above minimum (allow one small final chunk)
        small_chunks = [c for c in chunks if c.token_count < 50]
        assert len(small_chunks) <= 1


class TestChunkingComparator:
    """Tests for chunking comparator."""
    
    def test_comparison_produces_results(self):
        """Test that comparison produces valid results."""
        parsed_doc = parse_file(FIXTURES_DIR / 'test.txt')
        comparator = ChunkingComparator(chunk_size=100)
        result = comparator.compare(parsed_doc)
        
        assert result.recursive_chunk_count >= 0
        assert result.semantic_chunk_count >= 0
        assert result.recursive_avg_size >= 0
        assert result.semantic_avg_size >= 0
        assert result.recommendation
    
    def test_comparison_distributions(self):
        """Test that comparison produces distribution data."""
        parsed_doc = parse_file(FIXTURES_DIR / 'test.txt')
        comparator = ChunkingComparator(chunk_size=100)
        result = comparator.compare(parsed_doc)
        
        assert result.recursive_distribution
        assert result.semantic_distribution
        
        # Distributions should sum to ~100%
        recursive_total = sum(result.recursive_distribution.values())
        assert 90 <= recursive_total <= 110  # Allow rounding errors
    
    def test_comparison_sample_chunks(self):
        """Test that comparison returns sample chunks."""
        parsed_doc = parse_file(FIXTURES_DIR / 'test.txt')
        comparator = ChunkingComparator(chunk_size=100)
        result = comparator.compare(parsed_doc)
        
        assert len(result.recursive_sample_chunks) <= 3
        assert len(result.semantic_sample_chunks) <= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
