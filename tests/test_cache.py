"""Tests for embedding cache."""
import pytest
import tempfile
import shutil
from pathlib import Path
from app.embeddings.cache import EmbeddingCache


class TestEmbeddingCache:
    """Tests for embedding cache."""
    
    def test_cache_stores_and_retrieves(self):
        """Test that cache stores and retrieves embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache"
            cache = EmbeddingCache(str(cache_path))
            
            text = "Test text for embedding"
            model_name = "test-model"
            embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            # Store
            cache.set(text, model_name, embedding)
            
            # Retrieve
            cached = cache.get(text, model_name)
            
            assert cached is not None
            assert cached == embedding
    
    def test_cache_returns_none_for_missing(self):
        """Test that cache returns None for missing entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache"
            cache = EmbeddingCache(str(cache_path))
            
            cached = cache.get("nonexistent text", "nonexistent-model")
            
            assert cached is None
    
    def test_cache_differentiates_by_text(self):
        """Test that different texts have different cache keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache"
            cache = EmbeddingCache(str(cache_path))
            
            model_name = "test-model"
            embedding1 = [0.1, 0.2, 0.3]
            embedding2 = [0.4, 0.5, 0.6]
            
            cache.set("Text one", model_name, embedding1)
            cache.set("Text two", model_name, embedding2)
            
            assert cache.get("Text one", model_name) == embedding1
            assert cache.get("Text two", model_name) == embedding2
    
    def test_cache_differentiates_by_model(self):
        """Test that different models have different cache keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache"
            cache = EmbeddingCache(str(cache_path))
            
            text = "Same text"
            embedding1 = [0.1, 0.2, 0.3]
            embedding2 = [0.4, 0.5, 0.6]
            
            cache.set(text, "model-a", embedding1)
            cache.set(text, "model-b", embedding2)
            
            assert cache.get(text, "model-a") == embedding1
            assert cache.get(text, "model-b") == embedding2
    
    def test_cache_stats(self):
        """Test cache statistics tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache"
            cache = EmbeddingCache(str(cache_path))
            
            model_name = "test-model"
            embedding = [0.1, 0.2, 0.3]
            
            # Cache miss
            cache.get("text", model_name)
            
            # Store
            cache.set("text", model_name, embedding)
            
            # Cache hit
            cache.get("text", model_name)
            
            stats = cache.get_stats()
            
            assert stats['cache_hits'] == 1
            assert stats['cache_misses'] == 1
            assert stats['hit_rate'] == 0.5
    
    def test_cache_get_or_embed(self):
        """Test get_or_embed convenience method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache"
            cache = EmbeddingCache(str(cache_path))
            
            texts = ["Text one", "Text two", "Text three"]
            model_name = "test-model"
            
            def mock_embed(texts_to_embed):
                return [[0.1] * 384 for _ in texts_to_embed]
            
            # First call - all misses
            embeddings1, stats1 = cache.get_or_embed(texts, model_name, mock_embed)
            
            assert len(embeddings1) == 3
            assert stats1['cache_misses'] == 3
            assert stats1['cache_hits'] == 0
            
            # Second call - all hits
            embeddings2, stats2 = cache.get_or_embed(texts, model_name, mock_embed)
            
            assert len(embeddings2) == 3
            assert stats2['cache_hits'] == 3
            # Stats are cumulative in the same object instance
            assert stats2['cache_misses'] == 3
    
    def test_cache_clear(self):
        """Test cache clearing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache"
            cache = EmbeddingCache(str(cache_path))
            
            cache.set("text", "model", [0.1, 0.2])
            assert cache.get("text", "model") is not None
            
            cache.clear()
            
            # After clear, entry should be gone
            assert cache.get("text", "model") is None
    
    def test_cache_size(self):
        """Test cache size method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache"
            cache = EmbeddingCache(str(cache_path))
            
            assert cache.size() == 0
            
            cache.set("text1", "model", [0.1])
            cache.set("text2", "model", [0.2])
            
            assert cache.size() == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
