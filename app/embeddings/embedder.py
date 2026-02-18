"""Text embedding using Jina Embeddings API.

Replaces local sentence-transformers to avoid OOM crashes on Windows.
Free tier: 10M tokens at https://jina.ai/embeddings/
"""
import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.logging_config import get_logger

logger = get_logger(__name__)

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
REQUEST_TIMEOUT = (10, 60)  # (connect timeout, read timeout)
MAX_RETRIES = 5  # Maximum retries for rate limiting
RETRY_DELAY = 1.0  # Initial retry delay in seconds
MAX_RETRY_DELAY = 30.0  # Maximum retry delay in seconds


def _create_session() -> requests.Session:
    """Create a session with retry strategy and connection pooling."""
    session = requests.Session()
    
    # Retry strategy for transient errors (but not 429 - we handle that separately)
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST", "GET"]
    )
    
    # Mount adapters for both http and https
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


class Embedder:
    """Jina Embeddings API client.

    Uses asymmetric embedding:
    - ``retrieval.passage`` for document chunks (ingestion)
    - ``retrieval.query`` for search queries (retrieval)
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "Embedder":
        return cls()

    def __init__(self):
        if self._initialized:
            return
        self.api_key = os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "JINA_API_KEY not set. Get a free key at https://jina.ai/embeddings/"
            )
        self.model = "jina-embeddings-v3"
        self.dimensions = 384  # match existing ChromaDB collection
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._session = _create_session()
        self._initialized = True
        logger.info("Jina Embedder initialized (model=%s, dims=%d)", self.model, self.dimensions)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Embed a single text (passage task)."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using ``retrieval.passage`` task.
        
        Includes exponential backoff for rate limiting (429 errors).
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        batch_size = 50  # Reduced batch size to avoid rate limits

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.info(
                "Embedding batch %d/%d (%d texts) via Jina API",
                i // batch_size + 1,
                (len(texts) + batch_size - 1) // batch_size,
                len(batch),
            )

            payload = {
                "model": self.model,
                "input": batch,
                "dimensions": self.dimensions,
                "task": "retrieval.passage",
            }

            # Exponential backoff for rate limiting
            for attempt in range(MAX_RETRIES):
                try:
                    resp = self._session.post(
                        JINA_API_URL, 
                        headers=self._headers, 
                        json=payload, 
                        timeout=REQUEST_TIMEOUT
                    )
                    
                    # Handle rate limiting with exponential backoff
                    if resp.status_code == 429:
                        retry_after = float(resp.headers.get("Retry-After", RETRY_DELAY))
                        delay = min(retry_after * (2 ** attempt), MAX_RETRY_DELAY)
                        logger.warning(
                            "Rate limited by Jina API (attempt %d/%d). Waiting %.1fs...",
                            attempt + 1, MAX_RETRIES, delay
                        )
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(delay)
                            continue
                        else:
                            raise requests.exceptions.RequestException(
                                f"Rate limit exceeded after {MAX_RETRIES} retries"
                            )
                    
                    resp.raise_for_status()
                    data = resp.json()
                    embeddings = [
                        item["embedding"]
                        for item in sorted(data["data"], key=lambda x: x["index"])
                    ]
                    all_embeddings.extend(embeddings)
                    break  # Success, exit retry loop
                    
                except requests.exceptions.Timeout as e:
                    logger.warning("Jina API timeout (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, e)
                    if attempt == MAX_RETRIES - 1:
                        raise
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                    
                except requests.exceptions.RequestException as e:
                    if "429" in str(e):
                        # Already handled above, but catch just in case
                        delay = min(RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                        logger.warning("Rate limited, waiting %.1fs...", delay)
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(delay)
                            continue
                    logger.error("Jina API error (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, e)
                    raise

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query using ``retrieval.query`` task."""
        payload = {
            "model": self.model,
            "input": [query],
            "dimensions": self.dimensions,
            "task": "retrieval.query",
        }

        try:
            resp = self._session.post(
                JINA_API_URL, 
                headers=self._headers, 
                json=payload, 
                timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except requests.exceptions.RequestException as e:
            logger.error("Jina API error in embed_query: %s", e)
            raise

    # ------------------------------------------------------------------
    # Compat helpers (used by pipeline.py)
    # ------------------------------------------------------------------

    def get_cache_stats(self) -> dict:
        return {"cache_hits": 0, "cache_misses": 0, "total_requests": 0, "hit_rate": 0.0}

    def reset_cache_stats(self):
        pass
