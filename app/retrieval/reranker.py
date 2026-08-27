from app.config import settings
"""Re-ranker using Jina Reranker API.

Replaces local CrossEncoder to remove sentence-transformers dependency.
Uses the same JINA_API_KEY (free tier includes reranking).
"""
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from app.retrieval.types import SearchResult
from app.logging_config import get_logger

logger = get_logger(__name__)

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
REQUEST_TIMEOUT = (10, 30)  # (connect timeout, read timeout)


def _create_session() -> requests.Session:
    """Create a session with retry strategy and connection pooling."""
    session = requests.Session()
    
    # Retry strategy for transient errors
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"]
    )
    
    # Mount adapters for both http and https
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=5,
        pool_maxsize=5
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


class Reranker:
    """Jina-based re-ranker for refining search results."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.api_key = settings.JINA_API_KEY
        self.model = "jina-reranker-v2-base-multilingual"
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._session = _create_session()
        self._initialized = True

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int = 5,
    ) -> list[SearchResult]:
        """Re-rank search results using Jina Reranker API.

        Falls back to returning the original top_n results if the API
        is unavailable or the key is missing.
        """
        if not results:
            return []
        if not self.api_key:
            logger.warning("JINA_API_KEY not set — skipping rerank")
            return results[:top_n]

        documents = [r.text for r in results]

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(results)),
        }

        try:
            resp = self._session.post(
                JINA_RERANK_URL,
                headers=self._headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            reranked: list[SearchResult] = []
            for item in data["results"]:
                idx = item["index"]
                result = results[idx]
                result.score = float(item["relevance_score"])
                reranked.append(result)

            logger.info("Re-ranking complete, returning top %d", len(reranked))
            return reranked

        except requests.exceptions.Timeout as e:
            logger.error("Jina reranker timeout: %s", e)
            return results[:top_n]
        except requests.exceptions.RequestException as e:
            logger.error("Jina reranker error: %s", e)
            return results[:top_n]


def rerank_results(
    query: str,
    results: list[SearchResult],
    top_n: int = 5,
) -> list[SearchResult]:
    """Convenience function to re-rank results."""
    return Reranker().rerank(query, results, top_n)
