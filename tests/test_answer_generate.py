"""Tests for grounded answer generation.

Most of these use a scripted provider rather than a live one. The behaviour
under test is what the generator does with a model's output — dropping claims
that cite chunks nobody retrieved, flagging quotes that are not in their chunk,
declining rather than returning an empty answer — and driving that from real
generations would make the tests depend on a model choosing to misbehave.
"""

from __future__ import annotations

import json

import pytest

from app.answer.contract import GroundedAnswer, InsufficientEvidence
from app.answer.generate import AnswerGenerator, format_chunks
from app.llm.provider import GenerationError
from app.retrieval.types import SearchResult

CHUNK_A = SearchResult(
    chunk_id="chunk-a",
    text="The merger completed on 14 March 2019 after regulatory approval.",
    score=0.9,
    document_id="doc-1",
    filename="report.pdf",
    page_number=3,
    heading_context="Corporate history",
)
CHUNK_B = SearchResult(
    chunk_id="chunk-b",
    text="The combined company employed 4,200 people.",
    score=0.8,
    document_id="doc-1",
    filename="report.pdf",
    page_number=4,
)


def answer_json(claims) -> str:
    return json.dumps({"answer": {"kind": "grounded", "claims": claims}})


def decline_json(missing="nothing on that") -> str:
    return json.dumps({
        "answer": {
            "kind": "insufficient_evidence",
            "searched_for": "the question",
            "missing": missing,
        }
    })


@pytest.fixture
def generator(monkeypatch):
    """A generator whose provider returns whatever the test scripts."""
    gen = AnswerGenerator()
    gen._scripted = None

    def fake_call(prompt):
        if isinstance(gen._scripted, Exception):
            raise gen._scripted
        return gen._scripted

    monkeypatch.setattr(gen, "_call", fake_call)
    return gen


class TestPromptConstruction:
    def test_chunk_ids_appear_beside_their_text(self):
        """Citing must be a copying task. A model asked to recall an id it saw
        in a header will eventually write a plausible-looking one instead."""
        rendered = format_chunks([CHUNK_A, CHUNK_B])
        assert "[chunk-a]" in rendered
        assert "[chunk-b]" in rendered
        assert "The merger completed" in rendered

    def test_the_location_is_shown_so_a_citation_is_checkable(self):
        rendered = format_chunks([CHUNK_A])
        assert "report.pdf" in rendered
        assert "p.3" in rendered
        assert "Corporate history" in rendered


class TestGrounding:
    def test_a_well_cited_answer_survives_intact(self, generator):
        generator._scripted = answer_json([{
            "text": "The merger completed on 14 March 2019.",
            "chunk_ids": ["chunk-a"],
            "quote": "The merger completed on 14 March 2019",
        }])
        result = generator.generate("When?", [CHUNK_A, CHUNK_B])
        assert result.ok
        assert result.claim_count == 1
        assert not result.dropped_claims
        assert not result.unverified_claims

    def test_a_claim_citing_an_unretrieved_chunk_is_dropped(self, generator):
        """The model can only have got that id by inventing it."""
        generator._scripted = answer_json([
            {
                "text": "The merger completed on 14 March 2019.",
                "chunk_ids": ["chunk-a"],
                "quote": "The merger completed on 14 March 2019",
            },
            {
                "text": "Revenue tripled the following year.",
                "chunk_ids": ["chunk-z"],
                "quote": "Revenue tripled",
            },
        ])
        result = generator.generate("When?", [CHUNK_A, CHUNK_B])
        assert result.claim_count == 1
        assert len(result.dropped_claims) == 1
        assert result.dropped_claims[0]["cited"] == ["chunk-z"]

    def test_a_partly_grounded_claim_is_kept_and_narrowed(self, generator):
        """Half a citation is still a citation; the invented half is recorded."""
        generator._scripted = answer_json([{
            "text": "The merger completed on 14 March 2019.",
            "chunk_ids": ["chunk-a", "chunk-z"],
            "quote": "The merger completed on 14 March 2019",
        }])
        result = generator.generate("When?", [CHUNK_A, CHUNK_B])
        assert result.claim_count == 1
        assert result.answer.claims[0].chunk_ids == ["chunk-a"]
        assert result.dropped_claims[0]["cited"] == ["chunk-z"]

    def test_an_answer_with_nothing_grounded_becomes_a_decline(self, generator):
        """Returning an empty answer would push the problem downstream."""
        generator._scripted = answer_json([{
            "text": "Something entirely invented.",
            "chunk_ids": ["chunk-z"],
            "quote": "invented",
        }])
        result = generator.generate("When?", [CHUNK_A])
        assert result.ok
        assert result.declined
        assert "not retrieved" in result.answer.missing

    def test_a_quote_absent_from_its_chunk_is_flagged_not_dropped(self, generator):
        """Whether that is fabrication or paraphrase is the verification
        layer's question; this layer records it."""
        generator._scripted = answer_json([{
            "text": "The merger completed in spring 2019.",
            "chunk_ids": ["chunk-a"],
            "quote": "the merger wrapped up in the spring",
        }])
        result = generator.generate("When?", [CHUNK_A])
        assert result.claim_count == 1
        assert len(result.unverified_claims) == 1
        assert result.quote_verified_rate == 0.0

    def test_the_verified_rate_counts_only_surviving_claims(self, generator):
        generator._scripted = answer_json([
            {
                "text": "The merger completed on 14 March 2019.",
                "chunk_ids": ["chunk-a"],
                "quote": "The merger completed on 14 March 2019",
            },
            {
                "text": "It employed many people.",
                "chunk_ids": ["chunk-b"],
                "quote": "employed thousands of people",
            },
        ])
        result = generator.generate("When?", [CHUNK_A, CHUNK_B])
        assert result.claim_count == 2
        assert result.quote_verified_rate == 0.5


class TestDeclining:
    def test_a_decline_is_passed_through(self, generator):
        generator._scripted = decline_json("no chunk gives a date")
        result = generator.generate("When?", [CHUNK_A])
        assert result.ok
        assert result.declined
        assert result.claim_count == 0

    def test_no_retrieval_declines_rather_than_erroring(self, generator):
        """An empty corpus result is a finding, not a failure."""
        result = generator.generate("When?", [])
        assert result.ok
        assert result.declined
        assert "no chunks" in result.answer.missing


class TestFailureIsLoud:
    def test_a_provider_failure_is_reported_not_swallowed(self, generator):
        generator._scripted = GenerationError("provider is down")
        result = generator.generate("When?", [CHUNK_A])
        assert not result.ok
        assert "provider is down" in result.error
        assert result.answer is None

    def test_malformed_json_is_reported_with_the_raw_output(self, generator):
        generator._scripted = "this is not json at all"
        result = generator.generate("When?", [CHUNK_A])
        assert not result.ok
        assert "not valid JSON" in result.error
        assert result.raw_output == "this is not json at all"

    def test_json_violating_the_contract_is_reported(self, generator):
        """An uncited claim is rejected even when the provider let it through."""
        generator._scripted = answer_json([
            {"text": "Uncited.", "chunk_ids": [], "quote": "x"}
        ])
        result = generator.generate("When?", [CHUNK_A])
        assert not result.ok
        assert "contract" in result.error

    def test_the_summary_says_what_happened(self, generator):
        generator._scripted = answer_json([
            {
                "text": "The merger completed on 14 March 2019.",
                "chunk_ids": ["chunk-a"],
                "quote": "The merger completed on 14 March 2019",
            },
            {"text": "Invented.", "chunk_ids": ["chunk-z"], "quote": "x"},
        ])
        summary = generator.generate("When?", [CHUNK_A]).summary()
        assert "1 claims" in summary
        assert "dropped" in summary


class TestProvenance:
    def test_the_result_names_the_provider_and_model(self, generator):
        generator._scripted = decline_json()
        result = generator.generate("When?", [CHUNK_A])
        assert result.provider
        assert result.model

    def test_the_retrieved_chunks_are_kept_for_reconstruction(self, generator):
        generator._scripted = decline_json()
        result = generator.generate("When?", [CHUNK_A, CHUNK_B])
        assert [r.chunk_id for r in result.retrieved] == ["chunk-a", "chunk-b"]


class TestTransientLimitsAreRetried:
    """A tokens-per-minute ceiling is a pause, not a failure.

    The retry helper existed but was wired only into the long-context arm, so
    the retrieval arm lost whole benchmark cells to an 8,000 TPM limit that
    clears in eight seconds — and the loss looked like the model being unable
    to answer.
    """

    def test_the_generator_retries_a_rate_limit(self, generator, monkeypatch):
        calls = []

        def flaky(prompt):
            calls.append(1)
            if len(calls) < 2:
                raise GenerationError("429 rate limit reached on tokens per minute")
            return answer_json([{
                "text": "Refunds take 5 to 7 business days.",
                "chunk_ids": ["chunk-a"],
                "quote": "The merger completed on 14 March 2019",
            }])

        monkeypatch.setattr(generator, "_call", flaky)
        monkeypatch.setattr("app.config.settings.GENERATION_RETRIES", 3)
        result = generator.generate("When?", [CHUNK_A])
        assert result.ok
        assert len(calls) == 2

    def test_a_permanent_failure_is_not_retried(self, generator, monkeypatch):
        calls = []

        def broken(prompt):
            calls.append(1)
            raise GenerationError("400 invalid schema")

        monkeypatch.setattr(generator, "_call", broken)
        result = generator.generate("When?", [CHUNK_A])
        assert not result.ok
        assert len(calls) == 1
