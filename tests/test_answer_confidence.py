"""Tests for confidence scoring and calibration.

Two properties matter here. Confidence must be computed from things the system
can observe, not from the model's opinion of itself. And the score must be
honest about being uncalibrated until someone has plotted it against measured
accuracy — a confidence nobody has checked is a number that sorts answers
without saying what the order is worth.
"""

from __future__ import annotations

import pytest

from app.answer.calibration import (
    Bin,
    CalibrationCurve,
    ScoredOutcome,
    build_curve,
)
from app.answer.confidence import (
    ConfidenceBreakdown,
    coverage,
    retrieval_strength,
    score_answer,
    source_agreement,
)
from app.answer.contract import Claim, GroundedAnswer, InsufficientEvidence
from app.answer.verify import CitationVerifier
from app.retrieval.types import SearchResult

POLICY = SearchResult(
    chunk_id="policy",
    text="Refunds are issued to the original payment method and take 5 to 7 business days.",
    score=0.9,
    document_id="doc",
    filename="policy.txt",
)
HOURS = SearchResult(
    chunk_id="hours",
    text="Support operates Monday to Friday, 9am to 6pm UK time.",
    score=0.8,
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


class TestSignals:
    def test_retrieval_strength_follows_the_top_score(self):
        """A strong first hit with a weak tail is a good retrieval; averaging
        would punish it for returning extra candidates."""
        strong = retrieval_strength([POLICY, SearchResult(
            chunk_id="x", text="", score=0.05, document_id="d", filename="f")])
        assert strong > 0.9

    def test_nothing_retrieved_scores_zero(self):
        assert retrieval_strength([]) == 0.0

    def test_coverage_notices_an_unaddressed_question(self):
        answer = GroundedAnswer(claims=[claim()])
        engaged = coverage("How long do refunds take?", answer)
        ignored = coverage("What is the shipping cost to Germany?", answer)
        assert engaged > ignored

    def test_agreement_penalises_a_single_source(self):
        """One chunk supporting everything is a single point of failure."""
        one_source = GroundedAnswer(claims=[claim(), claim(text="Refunds go to the card.")])
        two_sources = GroundedAnswer(claims=[
            claim(),
            claim(text="Support runs weekdays.", chunk_ids=["hours"],
                  quote="Monday to Friday"),
        ])
        assert source_agreement(two_sources) > source_agreement(one_source)


class TestScoring:
    def test_a_well_supported_answer_scores_high(self):
        answer = GroundedAnswer(claims=[claim()])
        report = CitationVerifier(judge_provider=None).verify(answer, [POLICY])
        result = score_answer("How long do refunds take?", answer, [POLICY], report)
        assert result.score > 0.6

    def test_a_failed_citation_pulls_the_score_down(self):
        good = GroundedAnswer(claims=[claim()])
        bad = GroundedAnswer(claims=[claim(quote="entirely invented text")])
        verifier = CitationVerifier(judge_provider=None)
        high = score_answer("How long do refunds take?", good, [POLICY],
                            verifier.verify(good, [POLICY]))
        low = score_answer("How long do refunds take?", bad, [POLICY],
                           verifier.verify(bad, [POLICY]))
        assert low.score < high.score

    def test_an_unverified_answer_does_not_score_as_a_verified_one(self):
        """A missing check is not a passing one."""
        answer = GroundedAnswer(claims=[claim()])
        report = CitationVerifier(judge_provider=None).verify(answer, [POLICY])
        verified = score_answer("How long do refunds take?", answer, [POLICY], report)
        unverified = score_answer("How long do refunds take?", answer, [POLICY], None)
        assert unverified.score <= verified.score
        assert any("not verified" in n for n in unverified.notes)

    def test_a_decline_is_not_scored_as_a_bad_answer(self):
        """Otherwise "correctly declined" and "answered badly" become
        indistinguishable downstream."""
        declined = InsufficientEvidence(searched_for="x", missing="no chunk says")
        result = score_answer("x", declined, [POLICY])
        assert result.score == 0.0
        assert any("declined" in n for n in result.notes)

    def test_self_reported_confidence_is_recorded_not_used(self):
        """It is another generation from the same model. Whether it correlates
        with being right is a measurement, not an assumption."""
        answer = GroundedAnswer(claims=[claim()])
        with_claim = score_answer("How long?", answer, [POLICY], self_reported=0.99)
        without = score_answer("How long?", answer, [POLICY])
        assert with_claim.score == without.score
        assert with_claim.self_reported == 0.99

    def test_the_breakdown_says_why_the_score_is_low(self):
        answer = GroundedAnswer(claims=[claim()])
        result = score_answer("What is the shipping cost to Germany?", answer, [])
        assert result.notes
        assert "weak retrieval" in result.explain()

    def test_the_score_stays_within_bounds(self):
        answer = GroundedAnswer(claims=[claim()])
        result = score_answer("How long do refunds take?", answer, [POLICY])
        assert 0.0 <= result.score <= 1.0


class TestCalibration:
    def _outcomes(self, spec) -> list[ScoredOutcome]:
        out = []
        for confidence, right, wrong in spec:
            out += [ScoredOutcome(confidence, True) for _ in range(right)]
            out += [ScoredOutcome(confidence, False) for _ in range(wrong)]
        return out

    def test_a_calibrated_score_reports_a_small_error(self):
        curve = build_curve(self._outcomes([
            (0.85, 17, 3),   # 85% predicted, 85% measured
            (0.65, 13, 7),   # 65% predicted, 65% measured
        ]))
        assert curve.expected_calibration_error < 0.05
        assert curve.is_calibrated

    def test_an_overconfident_score_is_caught(self):
        curve = build_curve(self._outcomes([(0.9, 8, 12)]))
        assert curve.expected_calibration_error > 0.3
        assert not curve.is_calibrated
        assert "not a probability" in curve.report()

    def test_a_thin_bin_is_flagged_rather_than_quoted(self):
        """A bin of three swings on one answer, and printing it beside a full
        one invites a comparison the data cannot support."""
        curve = build_curve(self._outcomes([(0.85, 2, 1)]))
        assert "n too small" in curve.report()
        assert not curve.bins[-2].is_measurable or curve.total == 3

    def test_declines_are_excluded(self):
        """A correct decline is not a correct answer; counting it as one lets a
        system that declines everything report perfect calibration."""
        outcomes = [
            ScoredOutcome(0.9, True),
            ScoredOutcome(0.0, True, declined=True),
        ]
        assert build_curve(outcomes).total == 1

    def test_a_threshold_comes_with_what_it_costs(self):
        curve = build_curve(self._outcomes([
            (0.95, 19, 1),
            (0.85, 15, 5),
            (0.45, 5, 15),
        ]))
        threshold, cost = curve.threshold_for(0.85)
        assert threshold < 1.0
        assert "declines" in cost
        assert "wrong" in cost

    def test_an_unreachable_target_says_so(self):
        curve = build_curve(self._outcomes([(0.9, 10, 10)]))
        threshold, cost = curve.threshold_for(0.99)
        assert threshold == 1.0
        assert "no threshold reaches" in cost

    def test_no_data_does_not_invent_a_threshold(self):
        threshold, cost = build_curve([]).threshold_for(0.9)
        assert threshold == 1.0
        assert "no bin" in cost

    def test_the_gap_names_the_direction(self):
        over = Bin(low=0.8, high=0.9, outcomes=[
            ScoredOutcome(0.85, False) for _ in range(10)
        ])
        assert over.gap > 0  # positive means overconfident


class TestCalibrationPersistence:
    """A saved run keeps the raw outcomes, not only the binned curve.

    Bin edges are a presentation choice. A reader who disagrees with mine
    should be able to re-bin the same data rather than re-run the measurement,
    and a change to the binning should apply to old runs instead of leaving
    them frozen in whatever shape they were written with.
    """

    def _outcomes(self):
        return [
            ScoredOutcome(0.85, True, question_id="a"),
            ScoredOutcome(0.85, False, question_id="b"),
            ScoredOutcome(0.45, False, question_id="c"),
            ScoredOutcome(0.0, True, question_id="d", declined=True),
        ]

    def test_a_run_survives_a_round_trip(self, tmp_path):
        from app.calibration_io import load_outcomes, save_outcomes

        outcomes = self._outcomes()
        path = tmp_path / "calibration.json"
        save_outcomes(outcomes, build_curve(outcomes), path, {"repeats": 3})

        restored, curve = load_outcomes(path)
        assert len(restored) == len(outcomes)
        assert curve.total == 3  # the decline is excluded

    def test_the_raw_outcomes_are_kept(self, tmp_path):
        import json

        from app.calibration_io import save_outcomes

        outcomes = self._outcomes()
        path = tmp_path / "calibration.json"
        save_outcomes(outcomes, build_curve(outcomes), path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert len(payload["outcomes"]) == 4
        assert {o["question_id"] for o in payload["outcomes"]} == {"a", "b", "c", "d"}

    def test_the_curve_is_rebuilt_rather_than_read(self, tmp_path):
        """So a later change to the bin edges applies to a run saved today."""
        from app.calibration_io import load_outcomes, save_outcomes

        outcomes = self._outcomes()
        path = tmp_path / "calibration.json"
        save_outcomes(outcomes, build_curve(outcomes), path)
        _, curve = load_outcomes(path)
        assert sum(b.count for b in curve.bins) == curve.total
