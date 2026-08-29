"""Tests for the long-context arm.

The comparison this project publishes is only worth anything if both arms got a
fair attempt, so most of these protect that fairness: the same contract, the
same grounding rules, tokens counted rather than estimated, and a corpus that
does not fit recorded as not fitting rather than quietly truncated.
"""

from __future__ import annotations

import pytest

from app.answer.contract import Claim, GroundedAnswer, InsufficientEvidence
from app.answer.schema import for_gemini, gemini_answer_schema, strict_answer_schema
from app.llm.provider import (
    MAX_HONOURED_DELAY,
    GenerationError,
    is_transient,
    requested_delay,
    retry_with_backoff,
)
from app.longcontext.stuff import RESERVED_TOKENS, CorpusFit, LongContextAnswerer
from app.retrieval.types import SearchResult


def chunk(chunk_id="c1", text="Refunds take 5 to 7 business days.") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id, text=text, score=0.0,
        document_id="doc", filename="policy.txt",
    )


class TestGeminiSchemaTranslation:
    """Groq and Gemini accept different subsets, in opposite directions."""

    def test_one_of_becomes_any_of(self):
        """Gemini's Schema type has any_of and no one_of at all."""
        rendered = str(gemini_answer_schema())
        assert "anyOf" in rendered
        assert "oneOf" not in rendered

    def test_additional_properties_is_removed(self):
        """Gemini returns "Cannot find field" for it; Groq strict mode requires
        it on every object. The same contract needs opposite treatment."""
        assert "additionalProperties" not in str(gemini_answer_schema())

    def test_groq_still_gets_what_groq_requires(self):
        """Translating for one provider must not break the other."""
        groq_schema, _ = strict_answer_schema()
        assert "additionalProperties" in str(groq_schema)
        assert "oneOf" in str(groq_schema)

    def test_the_branches_survive_translation(self):
        branches = gemini_answer_schema()["properties"]["answer"]["anyOf"]
        assert len(branches) == 2
        assert all("properties" in b for b in branches)

    def test_nested_objects_are_translated_too(self):
        nested = {"type": "object", "properties": {
            "inner": {"type": "object", "additionalProperties": False,
                      "oneOf": [{"type": "string"}]},
        }}
        out = for_gemini(nested)
        assert "additionalProperties" not in str(out)
        assert "anyOf" in str(out)


class TestCorpusFit:
    def test_a_fitting_corpus_reports_utilisation(self):
        fit = CorpusFit(chunks=10, tokens=1000, limit=100_000, fits=True)
        assert fit.utilisation == 0.01
        assert "fits" in fit.describe()

    def test_an_oversized_corpus_says_so_loudly(self):
        fit = CorpusFit(chunks=9999, tokens=2_000_000, limit=1_000_000, fits=False)
        assert "DOES NOT FIT" in fit.describe()

    def test_head_room_is_left_for_the_answer(self):
        """A corpus sized to the exact window leaves nothing to reply with."""
        assert RESERVED_TOKENS > 0


class TestRefusingToTruncate:
    def test_an_oversized_corpus_fails_rather_than_truncating(self, monkeypatch):
        """Truncating would answer a different question from a different corpus
        and report it as this one."""
        answerer = LongContextAnswerer.__new__(LongContextAnswerer)
        answerer.provider = type("P", (), {"name": "gemini", "model": "test"})()
        answerer._window = 1000
        monkeypatch.setattr(
            answerer, "measure_fit",
            lambda chunks: CorpusFit(chunks=len(chunks), tokens=99_999,
                                     limit=1000, fits=False),
        )
        result, fit = LongContextAnswerer.answer(answerer, "q", [chunk()])
        assert not result.ok
        assert "does not fit" in result.error
        assert not fit.fits

    def test_the_window_falls_back_low_not_high(self, monkeypatch):
        """Overstating the window turns "did not fit" into a truncated request
        nobody labelled."""
        answerer = LongContextAnswerer.__new__(LongContextAnswerer)
        answerer.provider = type("P", (), {"name": "gemini", "model": "nope"})()
        answerer._window = None
        monkeypatch.setattr(
            "app.longcontext.stuff.settings.GEMINI_API_KEY", "invalid"
        )
        assert answerer.context_limit() <= 128_000


class TestGrounding:
    """The same rule the retrieval arm applies, and it bites harder here."""

    def _answerer(self):
        answerer = LongContextAnswerer.__new__(LongContextAnswerer)
        answerer.provider = type("P", (), {"name": "gemini", "model": "test"})()
        return answerer

    def test_a_claim_citing_an_absent_id_is_dropped(self):
        """Every chunk was supplied, so an unknown id is unambiguously
        invented rather than possibly filtered out."""
        from app.answer.generate import GenerationResult

        record = GenerationResult(ok=True, question="q")
        answer = GroundedAnswer(claims=[
            Claim(text="Invented.", chunk_ids=["nope"], quote="x"),
        ])
        grounded = LongContextAnswerer._ground(
            self._answerer(), answer, [chunk()], record
        )
        assert isinstance(grounded, InsufficientEvidence)
        assert record.dropped_claims
        assert "not in the corpus" in record.dropped_claims[0]["reason"]

    def test_a_valid_claim_survives(self):
        from app.answer.generate import GenerationResult

        record = GenerationResult(ok=True, question="q")
        answer = GroundedAnswer(claims=[
            Claim(text="Refunds take 5 to 7 business days.",
                  chunk_ids=["c1"], quote="take 5 to 7 business days"),
        ])
        grounded = LongContextAnswerer._ground(
            self._answerer(), answer, [chunk()], record
        )
        assert isinstance(grounded, GroundedAnswer)
        assert not record.unverified_claims


class TestTransientRetry:
    @pytest.mark.parametrize(
        "error", ["503 UNAVAILABLE high demand", "429 rate limit exceeded",
                  "500 internal error", "deadline exceeded"],
    )
    def test_capacity_errors_are_retried(self, error):
        assert is_transient(error)

    @pytest.mark.parametrize(
        "error", ["400 invalid schema", "404 model not found",
                  "Cannot find field additional_properties"],
    )
    def test_permanent_errors_are_not(self, error):
        """A rejected schema fails identically however many times it is sent."""
        assert not is_transient(error)

    def test_it_succeeds_after_a_transient_failure(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 2:
                raise GenerationError("503 UNAVAILABLE")
            return "ok"

        assert retry_with_backoff(flaky, attempts=3, label="test") == "ok"
        assert len(calls) == 2

    def test_it_gives_up_on_a_permanent_failure_immediately(self):
        calls = []

        def broken():
            calls.append(1)
            raise GenerationError("400 invalid schema")

        with pytest.raises(GenerationError):
            retry_with_backoff(broken, attempts=4, label="test")
        assert len(calls) == 1

    def test_it_stops_after_the_attempt_limit(self):
        calls = []

        def always_down():
            calls.append(1)
            raise GenerationError("503 UNAVAILABLE")

        with pytest.raises(GenerationError):
            retry_with_backoff(always_down, attempts=2, label="test")
        assert len(calls) == 2


class TestBothArmsOnOneModel:
    """The comparison is retrieval against stuffing. If one arm runs on Groq
    and the other on Gemini, the headline number conflates that with one model
    against another, and nothing separates the two afterwards."""

    def test_the_generator_has_a_gemini_path(self):
        from app.answer.generate import AnswerGenerator

        generator = AnswerGenerator(provider_name="gemini")
        assert generator.provider.name == "gemini"

    def test_gemini_gets_a_schema_it_accepts(self):
        """Groq requires additionalProperties on every object; Gemini rejects
        the key outright. The same contract needs both translations."""
        from app.answer.schema import gemini_answer_schema, strict_answer_schema

        groq_schema, _ = strict_answer_schema()
        assert "additionalProperties" in str(groq_schema)
        assert "additionalProperties" not in str(gemini_answer_schema())


class TestHonouringTheRequestedDelay:
    """Providers say how long to wait, and the number is usually right.

    Guessing instead is how a run loses cells to a limit that would have
    cleared: an exponential backoff topping out at eight seconds gives up on a
    ceiling whose own error message asks for twenty-seven, and the loss then
    looks like the model being unable to answer.
    """

    def test_a_named_delay_is_read(self):
        assert requested_delay("Please retry in 27.117175799s.") == pytest.approx(27.117, abs=0.01)

    def test_the_structured_field_is_read_too(self):
        assert requested_delay("'retryDelay': '13s'") == 13.0

    def test_no_named_delay_falls_back_to_backoff(self):
        assert requested_delay("429 rate limit reached") is None

    def test_an_absurd_delay_is_capped(self):
        """A wait measured in hours is a daily quota, and waiting will not fix
        it. Better to fail visibly than to block."""
        assert requested_delay("Please retry in 4000s.") == MAX_HONOURED_DELAY

    def test_the_retry_waits_what_was_asked(self, monkeypatch):
        waits = []
        monkeypatch.setattr("app.llm.provider.time.sleep", waits.append)
        calls = []

        def limited():
            calls.append(1)
            if len(calls) < 2:
                raise GenerationError("429 exhausted. Please retry in 27s.")
            return "ok"

        assert retry_with_backoff(limited, attempts=3, label="test") == "ok"
        assert 27 <= waits[0] <= 29
