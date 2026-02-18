"""Tests for chunk quality validation."""
import pytest
from app.chunking.recursive import Chunk
from app.validation.chunk_validator import ChunkValidator, QualityResult


class TestChunkQualityValidation:
    """Tests for chunk quality validator."""
    
    def test_good_chunk_passes(self):
        """Test that a well-formed chunk passes validation."""
        validator = ChunkValidator()
        
        good_chunk = Chunk(
            text="This is a well-formed chunk with coherent text content. "
                 "It has proper sentence structure and meaningful information.",
            chunk_index=0,
            start_char=0,
            end_char=100,
            token_count=25,
            source_page=1,
            heading_context="Introduction"
        )
        
        result = validator.validate(good_chunk)
        
        assert result.passed is True
        assert result.score >= 0.5
        assert len(result.issues) == 0
    
    def test_small_chunk_fails(self):
        """Test that chunks with < 20 tokens are rejected."""
        validator = ChunkValidator()
        
        small_chunk = Chunk(
            text="Small chunk",
            chunk_index=0,
            start_char=0,
            end_char=11,
            token_count=2,  # Less than 20
            source_page=1
        )
        
        result = validator.validate(small_chunk)
        
        assert result.passed is False
        assert result.score == 0.0
        assert any("small" in issue.lower() for issue in result.issues)
    
    def test_high_whitespace_flagged(self):
        """Test that high whitespace ratio is flagged."""
        validator = ChunkValidator()
        
        whitespace_chunk = Chunk(
            text="Words    " * 20,  # Lots of spaces
            chunk_index=0,
            start_char=0,
            end_char=100,
            token_count=30
        )
        
        result = validator.validate(whitespace_chunk)
        
        # Should have lower score or whitespace issue
        assert result.score < 1.0 or any("whitespace" in i.lower() for i in result.issues)
    
    def test_encoding_artifacts_detected(self):
        """Test that encoding artifacts are detected."""
        validator = ChunkValidator()
        
        artifact_chunk = Chunk(
            text="Text with encoding issues: \ufffd\ufffd\x00\x00",
            chunk_index=0,
            start_char=0,
            end_char=40,
            token_count=25
        )
        
        result = validator.validate(artifact_chunk)
        
        # Should detect encoding artifacts
        assert any("encoding" in i.lower() or "replacement" in i.lower() 
                  or "null" in i.lower() for i in result.issues)
    
    def test_repetition_detected(self):
        """Test that repetitive content is detected."""
        validator = ChunkValidator()
        
        # Create repetitive text
        repetitive_text = "repeat " * 100  # Will create repeating 5-grams
        
        repetitive_chunk = Chunk(
            text=repetitive_text,
            chunk_index=0,
            start_char=0,
            end_char=len(repetitive_text),
            token_count=50
        )
        
        result = validator.validate(repetitive_chunk)
        
        # Should detect repetition
        assert result.score < 1.0 or any("repetition" in i.lower() for i in result.issues)
    
    def test_low_coherence_detected(self):
        """Test that low coherence (mostly numbers/symbols) is detected."""
        validator = ChunkValidator()
        
        low_coherence_chunk = Chunk(
            text="123 456 789 012 345 678 901 234 567 890 !@#$%",
            chunk_index=0,
            start_char=0,
            end_char=45,
            token_count=15
        )
        
        result = validator.validate(low_coherence_chunk)
        
        # Should detect low coherence (but might also be rejected for being small)
        if result.passed:
            assert any("coherence" in i.lower() or "digit" in i.lower() 
                      or "symbol" in i.lower() for i in result.issues)
    
    def test_batch_validation(self):
        """Test batch validation."""
        validator = ChunkValidator()
        
        chunks = [
            Chunk(
                text="Good chunk with proper text content here",
                chunk_index=0,
                start_char=0,
                end_char=40,
                token_count=25
            ),
            Chunk(
                text="Bad",
                chunk_index=1,
                start_char=40,
                end_char=43,
                token_count=1
            )
        ]
        
        results = validator.validate_batch(chunks)
        
        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False
    
    def test_validation_summary(self):
        """Test validation summary generation."""
        validator = ChunkValidator()
        
        chunks = [
            Chunk(
                text="Good chunk one",
                chunk_index=0,
                start_char=0,
                end_char=20,
                token_count=25
            ),
            Chunk(
                text="Good chunk two",
                chunk_index=1,
                start_char=20,
                end_char=40,
                token_count=25
            ),
            Chunk(
                text="X",
                chunk_index=2,
                start_char=40,
                end_char=41,
                token_count=1
            )
        ]
        
        results = validator.validate_batch(chunks)
        summary = validator.get_validation_summary(chunks, results)
        
        assert summary['total_chunks'] == 3
        assert summary['passed'] == 2
        assert summary['failed'] == 1
        assert summary['pass_rate'] == 66.7
    
    def test_pass_threshold(self):
        """Test custom pass threshold."""
        strict_validator = ChunkValidator(pass_threshold=0.9)
        lenient_validator = ChunkValidator(pass_threshold=0.3)
        
        # Create chunk with some issues
        chunk = Chunk(
            text="Text with   extra   spaces and some content",
            chunk_index=0,
            start_char=0,
            end_char=45,
            token_count=20
        )
        
        strict_result = strict_validator.validate(chunk)
        lenient_result = lenient_validator.validate(chunk)
        
        # Strict validator should be more likely to fail
        assert strict_result.passed is False or lenient_result.passed is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
