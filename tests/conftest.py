"""Shared test configuration.

The suite splits into two halves that a clone should be able to tell apart:
tests that exercise logic, and tests that need a live embedding or generation
API. Without this, the second half *fails* on a fresh clone with no keys, and a
reader who has just cloned the repository sees twenty-nine red lines and
concludes the project is broken rather than that they have not set a key.

Anything needing a key is skipped with a reason naming the key. Skips are
visible in the summary, failures are not distinguishable from real ones.
"""

from __future__ import annotations

import pytest

from app.config import settings

#: Test modules that cannot run without an embedding key: they ingest, embed,
#: or search for real. Named rather than detected, because guessing from
#: imports would silently start skipping a test that stopped needing a key.
NEEDS_EMBEDDING_KEY = {
    "test_retrieval",
    "test_pipeline",
    "test_chunking",
    "test_real_world",
    "test_quality",
    "test_cache",
}


def pytest_collection_modifyitems(config, items):
    """Skip live-API tests when their key is absent, rather than failing them."""
    if settings.JINA_API_KEY:
        return

    skip = pytest.mark.skip(
        reason=(
            "needs JINA_API_KEY: this test embeds or searches for real. "
            "Get a free key at https://jina.ai/embeddings/"
        )
    )
    for item in items:
        if item.module.__name__.split(".")[-1] in NEEDS_EMBEDDING_KEY:
            item.add_marker(skip)
