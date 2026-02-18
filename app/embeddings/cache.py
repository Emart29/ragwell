"""In-memory embedding cache.

Kept only as a lightweight compatibility layer. The active embedding path uses
Jina API directly and does not depend on local persistence.
"""
import hashlib
from typing import Optional


class EmbeddingCache:
    """Simple process-local cache for text embeddings."""

    def __init__(self, cache_path: str = None):
        self.cache_path = cache_path
        self._memory_cache: dict[str, list[float]] = {}
        self._hits = 0
        self._misses = 0

    def _get_key(self, text: str, model_name: str) -> str:
        key_data = f"{text}||{model_name}".encode("utf-8")
        return hashlib.sha256(key_data).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[list[float]]:
        key = self._get_key(text, model_name)
        if key in self._memory_cache:
            self._hits += 1
            return self._memory_cache[key]
        self._misses += 1
        return None

    def set(self, text: str, model_name: str, embedding: list[float]):
        key = self._get_key(text, model_name)
        self._memory_cache[key] = embedding

    def get_or_embed(self, texts: list[str], model_name: str, embed_func: callable) -> tuple[list[list[float]], dict]:
        results = []
        to_embed = []
        to_embed_indices = []

        for i, text in enumerate(texts):
            cached = self.get(text, model_name)
            if cached is not None:
                results.append((i, cached))
            else:
                to_embed.append(text)
                to_embed_indices.append(i)

        if to_embed:
            new_embeddings = embed_func(to_embed)
            for i, text in enumerate(to_embed):
                embedding = new_embeddings[i]
                self.set(text, model_name, embedding)
                results.append((to_embed_indices[i], embedding))

        results.sort(key=lambda x: x[0])
        embeddings = [emb for _, emb in results]
        return embeddings, self.get_stats()

    def get_stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "total_requests": total,
            "hit_rate": round(hit_rate, 4),
        }

    def clear(self):
        self._memory_cache.clear()
        self._hits = 0
        self._misses = 0

    def size(self) -> int:
        return len(self._memory_cache)
