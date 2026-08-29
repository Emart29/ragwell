"""Tests for the fallback policy.

Two properties. The threshold must stay unset until it is measured, and the
layer must behave sensibly in that state rather than inventing a value. And when
a threshold does exist, escalation must only keep an answer that actually scored
better — a system that gets less reliable the harder it tries is worse than one
that gives up.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.answer.contract import Claim, GroundedAnswer, InsufficientEvidence
from app.answer.fallback import Action, FallbackPolicy
from app.answer.generate import GenerationResult
from app.retrieval.types import SearchResult

STRONG = SearchResult(
    chunk_id="policy",
    text="Refunds are issued to the original payment method and take 5 to 7 business days.",
    score=0.95,
    document_id="doc",
    filename="policy.txt",
)
WEAK = SearchResult(
    chunk_id="vague",
    text="Various operational matters are handled by the relevant team.",
    score=0.08,
    document_id="doc",
    filename="policy.txt",
)


def claim(**overrides) -> Claim:
    base = dict(
        text="Refunds take 5 to 7 business days.",
        chunk_ids=["policy"],
        quote="take 5 to 7 business days",
    )
    return Claim(**{**base, **overrides})


@dataclass
class FakeRetrieval:
    results: list


class FakeRetriever:
    """Returns scripted chunks and records how it was called."""

    def __init__(self, per_call):
        self._per_call = list(per_call)
        self.calls = []

    def retrieve(self, query, strategy="hybrid", top_k=5, **_):
        self.calls.append({"strategy": strategy, "top_k": top_k})
        results = self._per_call[min(len(self.calls) - 1, len(self._per_call) - 1)]
        return FakeRetrieval(results=results)


class FakeGenerator:
    """Returns a scripted answer per call."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.calls = 0

    def generate(self, question, results):
        answer = self._answers[min(self.calls, len(self._answers) - 1)]
        self.calls += 1
        return GenerationResult(
            ok=True,
            answer=answer,
            question=question,
            retrieved=list(results),
            provider="fake",
            model="fake",
        )


def good_answer():
    return GroundedAnswer(claims=[claim()])


def thin_answer():
    return GroundedAnswer(claims=[
        claim(text="Various matters are handled by a team.",
              chunk_ids=["vague"], quote="handled by the relevant team")
    ])


class TestUnmeasuredThreshold:
    """The default state: no threshold, because none has been measured."""

    def test_it_does_not_invent_a_threshold(self):
        policy = FallbackPolicy(FakeRetriever([[STRONG]]), FakeGenerator([good_answer()]))
        assert policy.threshold is None

    def test_it_records_the_score_without_judging_it(self):
        """With no threshold there is nothing to compare against, so claiming
        it "would have escalated" would be inventing the missing number."""
        retriever = FakeRetriever([[WEAK]])
        policy = FallbackPolicy(retriever, FakeGenerator([thin_answer()]))
        record = policy.answer("How long do refunds take?")
        assert record.observed_only
        assert "no threshold has been measured" in record.attempts[-1].note
        assert record.confidence.score > 0

    def test_it_spends_no_extra_requests(self):
        """Escalating on a threshold nobody measured would spend requests to
        satisfy a guess."""
        retriever = FakeRetriever([[WEAK]])
        generator = FakeGenerator([thin_answer()])
        FallbackPolicy(retriever, generator).answer("How long do refunds take?")
        assert len(retriever.calls) == 1
        assert generator.calls == 1

    def test_the_summary_says_it_is_only_observing(self):
        policy = FallbackPolicy(FakeRetriever([[WEAK]]), FakeGenerator([thin_answer()]))
        assert "observing only" in policy.answer("How long?").summary()


class TestEscalation:
    def test_a_confident_answer_is_accepted_without_retrying(self):
        retriever = FakeRetriever([[STRONG]])
        policy = FallbackPolicy(retriever, FakeGenerator([good_answer()]), threshold=0.3)
        record = policy.answer("How long do refunds take?")
        assert record.final_action is Action.ACCEPTED
        assert not record.escalated
        assert len(retriever.calls) == 1

    def test_a_weak_answer_triggers_a_requery(self):
        retriever = FakeRetriever([[WEAK], [STRONG]])
        policy = FallbackPolicy(
            retriever, FakeGenerator([thin_answer(), good_answer()]), threshold=0.6
        )
        record = policy.answer("How long do refunds take?")
        assert Action.REQUERIED in [a.action for a in record.attempts]
        assert retriever.calls[1]["strategy"] == "expanded"

    def test_widening_asks_for_more_chunks(self):
        retriever = FakeRetriever([[WEAK], [WEAK], [STRONG]])
        policy = FallbackPolicy(
            retriever,
            FakeGenerator([thin_answer(), thin_answer(), good_answer()]),
            threshold=0.9,
            widen_to=20,
        )
        policy.answer("How long do refunds take?")
        assert retriever.calls[2]["top_k"] == 20

    def test_a_worse_retry_does_not_replace_a_better_answer(self):
        """A system that gets less reliable the harder it tries is worse than
        one that gives up. The retries here score lower, so the confidence
        carried forward stays the best one seen."""
        retriever = FakeRetriever([[STRONG], [WEAK], [WEAK]])
        policy = FallbackPolicy(
            retriever,
            FakeGenerator([good_answer(), thin_answer(), thin_answer()]),
            threshold=0.99,
        )
        record = policy.answer("How long do refunds take?")
        first = record.attempts[0].confidence
        assert record.confidence.score == first
        assert all(a.confidence <= first for a in record.attempts)

    def test_exhausting_every_path_declines(self):
        retriever = FakeRetriever([[WEAK]])
        policy = FallbackPolicy(
            retriever, FakeGenerator([thin_answer()]), threshold=0.99
        )
        record = policy.answer("How long do refunds take?")
        assert record.declined
        assert record.final_action is Action.DECLINED
        assert "below the" in record.answer.missing

    def test_the_decline_names_the_score_it_fell_short_of(self):
        policy = FallbackPolicy(
            FakeRetriever([[WEAK]]), FakeGenerator([thin_answer()]), threshold=0.95
        )
        record = policy.answer("How long?")
        assert "0.95" in record.answer.missing


class TestAlreadyDeclined:
    def test_a_first_pass_decline_is_not_escalated(self):
        """The model said the corpus does not contain this. Retrying to make it
        say otherwise is how a system talks itself into an answer."""
        declined = InsufficientEvidence(searched_for="x", missing="no chunk says")
        retriever = FakeRetriever([[STRONG]])
        policy = FallbackPolicy(
            retriever, FakeGenerator([declined]), threshold=0.9
        )
        record = policy.answer("What is the CEO name?")
        assert record.final_action is Action.ALREADY_DECLINED
        assert len(retriever.calls) == 1


class TestRecording:
    def test_every_attempt_is_kept(self):
        retriever = FakeRetriever([[WEAK], [WEAK], [WEAK]])
        policy = FallbackPolicy(
            retriever,
            FakeGenerator([thin_answer(), thin_answer(), thin_answer()]),
            threshold=0.99,
        )
        record = policy.answer("How long?")
        actions = [a.action for a in record.attempts]
        assert Action.REQUERIED in actions
        assert Action.WIDENED in actions
        assert Action.DECLINED in actions

    def test_the_summary_shows_the_path(self):
        retriever = FakeRetriever([[WEAK], [STRONG]])
        policy = FallbackPolicy(
            retriever, FakeGenerator([thin_answer(), good_answer()]), threshold=0.6
        )
        assert "->" in policy.answer("How long?").summary()

    def test_each_attempt_records_the_strategy_it_used(self):
        retriever = FakeRetriever([[WEAK], [STRONG]])
        policy = FallbackPolicy(
            retriever, FakeGenerator([thin_answer(), good_answer()]), threshold=0.6
        )
        record = policy.answer("How long?")
        assert {a.strategy for a in record.attempts} >= {"hybrid", "expanded"}
