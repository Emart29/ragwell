"""Tests for the answer API.

These run against FastAPI's test client with the retrieval and generation
layers stubbed. What is under test is the contract the endpoint offers — that a
decline is a valid answer rather than an error, that a provider outage is not
reported as an empty corpus, and that every claim carries the evidence needed to
check it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.answer.contract import Claim, GroundedAnswer, InsufficientEvidence
from app.answer.generate import GenerationResult
from app.main import app
from app.retrieval.types import RetrievalResult, SearchResult

CHUNK = SearchResult(
    chunk_id="chunk-1",
    text="The Maximum Deposit Insurance Coverage was reviewed upward to N5,000,000.",
    score=0.9,
    document_id="doc",
    filename="2024-Annual-Report.pdf",
    page_number=27,
    heading_context="Deposit guarantee",
)


def grounded():
    return GroundedAnswer(claims=[Claim(
        text="The coverage limit is N5,000,000 per depositor.",
        chunk_ids=["chunk-1"],
        quote="reviewed upward to N5,000,000",
    )])


@pytest.fixture
def client(monkeypatch):
    """A client whose retrieval and generation are scripted."""
    from app.api import answer as module

    state = {"answer": grounded(), "ok": True, "error": ""}

    def fake_retrieve(query, strategy="hybrid", top_k=8, **_):
        return RetrievalResult(
            results=[CHUNK], strategy_used=strategy, search_time_ms=1.0,
            rerank_time_ms=0.0, total_candidates_before_rerank=1,
        )

    class FakeGenerator:
        provider = type("P", (), {"name": "fake", "model": "fake-1"})()

        def generate(self, question, results):
            return GenerationResult(
                ok=state["ok"],
                answer=state["answer"] if state["ok"] else None,
                question=question,
                retrieved=list(results),
                provider="fake",
                model="fake-1",
                error=state["error"],
                prompt_tokens=100,
                latency_ms=12.0,
            )

    monkeypatch.setattr(module.retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(module, "_generator", lambda provider: FakeGenerator())
    test_client = TestClient(app)
    test_client.state = state
    return test_client


class TestAnswering:
    def test_a_claim_carries_its_evidence(self, client):
        """A claim without a chunk id and a quote cannot be checked, which is
        the whole point of the endpoint."""
        body = client.post("/api/answer", json={"question": "What is the limit?"}).json()
        assert body["kind"] == "grounded"
        claim = body["claims"][0]
        assert claim["chunk_ids"] == ["chunk-1"]
        assert claim["quote"]

    def test_the_cited_chunk_is_returned_with_its_location(self, client):
        """So a reader can find the passage in the source document."""
        body = client.post("/api/answer", json={"question": "What is the limit?"}).json()
        chunk = body["chunks"][0]
        assert chunk["filename"] == "2024-Annual-Report.pdf"
        assert chunk["page"] == 27

    def test_the_quote_is_verified_by_default(self, client):
        body = client.post("/api/answer", json={"question": "What is the limit?"}).json()
        assert body["claims"][0]["quote_verified"] is True

    def test_verification_can_be_turned_off(self, client):
        """Unverified is None, not False: not checked and checked-and-absent
        are different states."""
        body = client.post(
            "/api/answer", json={"question": "What is the limit?", "verify": False}
        ).json()
        assert body["claims"][0]["quote_verified"] is None

    def test_confidence_is_returned_with_its_reasons(self, client):
        body = client.post("/api/answer", json={"question": "What is the limit?"}).json()
        assert 0.0 <= body["confidence"] <= 1.0
        assert isinstance(body["confidence_notes"], list)


class TestDeclining:
    def test_a_decline_is_a_valid_answer_not_an_error(self, client):
        """A corpus that cannot answer is a finding. Returning 4xx would make
        it indistinguishable from a bad request."""
        client.state["answer"] = InsufficientEvidence(
            searched_for="Zambia", missing="no chunk mentions Zambia"
        )
        response = client.post("/api/answer", json={"question": "What about Zambia?"})
        assert response.status_code == 200
        body = response.json()
        assert body["kind"] == "insufficient_evidence"
        assert body["claims"] == []
        assert "Zambia" in body["missing"]


class TestFailure:
    def test_a_provider_outage_is_a_502_not_an_empty_answer(self, client):
        """Returning 200 with no claims would make a model outage look like a
        corpus that says nothing."""
        client.state.update(ok=False, error="provider is down", answer=None)
        response = client.post("/api/answer", json={"question": "What is the limit?"})
        assert response.status_code == 502
        assert "provider is down" in response.json()["detail"]

    def test_a_short_question_is_rejected(self, client):
        assert client.post("/api/answer", json={"question": "hi"}).status_code == 422

    def test_an_unknown_provider_is_refused_by_name(self):
        from app.api import answer as module

        with pytest.raises(ValueError) as exc:
            module._generator("not_a_provider")
        assert "not_a_provider" in str(exc.value)


class TestStoredAnswers:
    def test_an_answer_can_be_fetched_again(self, client):
        created = client.post("/api/answer", json={"question": "What is the limit?"}).json()
        fetched = client.get(f"/api/answer/{created['answer_id']}").json()
        assert fetched["answer_id"] == created["answer_id"]
        assert fetched["claims"] == created["claims"]

    def test_an_unknown_id_is_a_404(self, client):
        assert client.get("/api/answer/nosuchid").status_code == 404

    def test_a_decline_has_nothing_to_verify(self, client):
        client.state["answer"] = InsufficientEvidence(
            searched_for="x", missing="nothing here"
        )
        created = client.post("/api/answer", json={"question": "What about Zambia?"}).json()
        response = client.post(f"/api/answer/{created['answer_id']}/verify")
        assert response.status_code == 400
        assert "no citations" in response.json()["detail"]

    def test_the_store_does_not_grow_without_bound(self, client):
        from app.api.answer import MAX_STORED_ANSWERS, _answers

        for _ in range(5):
            client.post("/api/answer", json={"question": "What is the limit?"})
        assert len(_answers) <= MAX_STORED_ANSWERS
