"""Tests for citation verification.

The claim this project makes is that citations can be checked, not merely
emitted, and that most of the checking is free. These tests protect both halves:
the cheap rungs must catch what they claim to catch, and the expensive rung must
never quietly report "supported" when it has malfunctioned.
"""

from __future__ import annotations

import pytest

from app.answer.contract import Claim, GroundedAnswer
from app.answer.verify import (
    OVERLAP_THRESHOLD,
    CitationVerifier,
    Verdict,
    content_words,
    lexical_overlap,
    parse_entailment,
)
from app.answer.verify_eval import CASES, Agreement
from app.retrieval.types import SearchResult

POLICY_TEXT = (
    "Customers may request a refund within 30 days of purchase. Refunds are "
    "issued to the original payment method and take 5 to 7 business days."
)
POLICY = SearchResult(
    chunk_id="policy",
    text=POLICY_TEXT,
    score=0.9,
    document_id="doc",
    filename="policy.txt",
)
UNRELATED = SearchResult(
    chunk_id="hours",
    text="Support operates Monday to Friday, 9am to 6pm UK time.",
    score=0.5,
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


def answer(*claims) -> GroundedAnswer:
    return GroundedAnswer(claims=list(claims) or [claim()])


@pytest.fixture
def verifier():
    """A verifier with no judge, so only the free rungs run."""
    return CitationVerifier(judge_provider=None)


class TestQuoteContainment:
    """Rung one: free, deterministic, and the one the argument rests on."""

    def test_a_real_quote_passes(self, verifier):
        report = verifier.verify(answer(), [POLICY])
        assert report.checks[0].quote_found is True

    def test_an_invented_quote_is_caught_without_a_model(self, verifier):
        report = verifier.verify(
            answer(claim(quote="refunds are processed within one hour")), [POLICY]
        )
        assert report.checks[0].quote_found is False
        assert report.caught_for_free == 1

    def test_a_quote_from_the_wrong_chunk_is_caught(self, verifier):
        report = verifier.verify(answer(claim(chunk_ids=["hours"])), [POLICY, UNRELATED])
        assert report.checks[0].quote_found is False

    def test_a_missing_chunk_leaves_the_rung_unanswered(self, verifier):
        """None, not False: a chunk that was never supplied is not evidence
        that the quote was invented."""
        report = verifier.verify(answer(claim(chunk_ids=["absent"])), [POLICY])
        assert report.checks[0].quote_found is None
        assert report.caught_for_free == 0


class TestLexicalOverlap:
    """Rung two: catches a genuine quote supporting an overreaching claim."""

    def test_a_claim_drawn_from_the_chunk_overlaps_heavily(self):
        assert lexical_overlap("Refunds take 5 to 7 business days.", POLICY_TEXT) > 0.8

    def test_an_unrelated_claim_barely_overlaps(self):
        assert lexical_overlap("The office is in Berlin.", POLICY_TEXT) < 0.3

    def test_stopwords_do_not_manufacture_overlap(self):
        """Two texts sharing only "the" and "of" share nothing."""
        assert lexical_overlap("It is on the of a", POLICY_TEXT) == 0.0

    def test_content_words_exclude_filler(self):
        assert content_words("the refund of a purchase") == {"refund", "purchase"}

    def test_overlap_is_measured_against_the_claim_not_the_chunk(self):
        """A chunk is far longer than a claim; a symmetric measure would score
        a short true claim as barely related."""
        short_true = lexical_overlap("Refunds take 5 to 7 business days.", POLICY_TEXT)
        assert short_true > OVERLAP_THRESHOLD


class TestCheapVerification:
    def test_both_free_rungs_must_pass(self, verifier):
        report = verifier.verify(answer(), [POLICY])
        assert report.checks[0].cheaply_verified
        assert report.cheaply_verified_rate == 1.0

    def test_a_real_quote_with_an_overreaching_claim_is_not_cheaply_verified(
        self, verifier
    ):
        """The quote is genuine and the claim has wandered off it."""
        overreaching = claim(
            text="The company operates a fleet of delivery vehicles in Berlin.",
            quote="take 5 to 7 business days",
        )
        check = verifier.verify(answer(overreaching), [POLICY]).checks[0]
        assert check.quote_found is True
        assert not check.cheaply_verified


class TestDecorativeCitations:
    def test_a_missing_quote_marks_it_decorative_without_a_judge(self, verifier):
        report = verifier.verify(answer(claim(quote="invented text")), [POLICY])
        assert report.checks[0].decorative
        assert report.decorative_rate == 1.0

    def test_a_judged_contradiction_is_decorative(self, verifier):
        report = verifier.verify(answer(), [POLICY])
        report.checks[0].entailment = Verdict.CONTRADICTED
        assert report.checks[0].decorative

    def test_a_judged_support_overrides_a_missing_quote(self, verifier):
        """A correct paraphrase fails string containment and is still sound.
        The expensive rung exists precisely for this."""
        report = verifier.verify(answer(claim(quote="not in the text")), [POLICY])
        report.checks[0].entailment = Verdict.SUPPORTED
        assert not report.checks[0].decorative


class TestJudgeParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("supported\nIt says so directly.", Verdict.SUPPORTED),
            ("contradicted\nThe passage says weeks.", Verdict.CONTRADICTED),
            ("unrelated\nNothing about refunds.", Verdict.UNRELATED),
            ("SUPPORTED", Verdict.SUPPORTED),
            ("  supported  ", Verdict.SUPPORTED),
        ],
    )
    def test_it_reads_a_verdict(self, raw, expected):
        assert parse_entailment(raw)[0] is expected

    def test_it_finds_a_verdict_buried_in_prose(self):
        verdict, _ = parse_entailment(
            "Looking at the passage carefully, I would say this is supported."
        )
        assert verdict is Verdict.SUPPORTED

    def test_an_unreadable_reply_is_unchecked_not_supported(self):
        """A judge that reads as "supported" when it malfunctions is worse than
        no judge at all."""
        assert parse_entailment("I'm not sure how to answer that.")[0] is Verdict.UNCHECKED

    def test_an_empty_reply_is_unchecked(self):
        assert parse_entailment("")[0] is Verdict.UNCHECKED
        assert parse_entailment("   \n ")[0] is Verdict.UNCHECKED

    def test_contradicted_wins_over_a_stray_mention_of_supported(self):
        """"not supported, contradicted" must not read as supported."""
        assert parse_entailment("contradicted — not supported by the text")[0] is (
            Verdict.CONTRADICTED
        )


class TestReporting:
    def test_an_unrun_judge_is_not_counted_as_agreement(self, verifier):
        report = verifier.verify(answer(), [POLICY])
        assert not report.judge_ran
        assert report.entailed_rate == 0.0

    def test_rates_are_zero_rather_than_dividing_by_zero(self, verifier):
        report = verifier.verify(
            GroundedAnswer(claims=[claim()]), []
        )
        assert report.decorative_rate == 0.0
        assert "no claims" not in report.summary()

    def test_the_summary_names_the_free_findings(self, verifier):
        report = verifier.verify(answer(claim(quote="invented")), [POLICY])
        assert "decorative" in report.summary()


class TestJudgeValidationSet:
    """The judge's own agreement rate must exist before its numbers are used."""

    def test_the_set_covers_all_three_verdicts(self):
        labels = {c.label for c in CASES}
        assert labels == {Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.UNRELATED}

    def test_every_case_says_what_it_tests(self):
        """A case whose purpose cannot be stated usually measures nothing."""
        assert all(c.tests for c in CASES)

    def test_it_includes_the_plausible_but_absent_case(self):
        """The failure that matters: true of the world, absent from the passage."""
        assert any(c.id == "absent_but_plausible" for c in CASES)

    def test_a_judge_with_false_supports_is_not_trustworthy(self):
        """Waving through a bad citation is worse than being strict, because
        catching bad citations is the entire point."""
        agreement = Agreement(
            total=8, agreed=7, false_supports=["absent_but_plausible"],
            false_rejects=[], unchecked=[],
        )
        assert agreement.rate > 0.75
        assert not agreement.trustworthy

    def test_a_strict_judge_can_still_be_trustworthy(self):
        agreement = Agreement(
            total=8, agreed=6, false_supports=[], false_rejects=["a", "b"],
            unchecked=[],
        )
        assert agreement.trustworthy

    def test_the_report_says_whether_to_quote_it(self):
        bad = Agreement(total=8, agreed=4, false_supports=["x"], false_rejects=[], unchecked=[])
        assert "unreliable" in bad.report()
