"""Tests for answer scoring.

The property these protect: the five measurements stay separate. A single
"accuracy" figure would let a system that declines every question look good, and
a system that answers every question look thorough. Both are failures and they
are opposite ones.
"""

from __future__ import annotations

import pytest

from app.answer.contract import Claim, GroundedAnswer, InsufficientEvidence
from eval.questions import Category, Difficulty, Question
from eval.score import (
    AnswerScore,
    Scoreboard,
    contains_fact,
    numbers_in,
    score_answer,
)


def question(**overrides) -> Question:
    base = dict(
        id="q1",
        text="What is the coverage limit?",
        category=Category.MISLEADING,
        difficulty=Difficulty.HARD,
        source_files=("2024-Annual-Report",),
        must_contain=("5,000,000",),
        must_not_contain=("N500,000",),
        tests="x",
    )
    return Question(**{**base, **overrides})


def answer(text="The limit is N5,000,000 per depositor.", chunk_ids=("c1",)):
    return GroundedAnswer(claims=[
        Claim(text=text, chunk_ids=list(chunk_ids), quote="N5,000,000")
    ])


def declined():
    return InsufficientEvidence(searched_for="x", missing="no report states it")


class TestFactMatching:
    def test_an_exact_figure_matches(self):
        assert contains_fact("The limit is N5,000,000.", "5,000,000")

    def test_separators_do_not_decide_it(self):
        """5000000 and 5,000,000 are the same answer."""
        assert contains_fact("The limit is 5000000 naira.", "5,000,000")

    def test_words_instead_of_digits_still_match(self):
        assert contains_fact("The limit is 5 million naira.", "5,000,000")

    def test_a_different_number_does_not_match(self):
        """This corpus is entirely numbers, so a near miss is a wrong answer."""
        assert not contains_fact("The limit is N500,000.", "5,000,000")

    def test_case_and_spacing_are_forgiven(self):
        assert contains_fact("paid within  FOUR   DAYS", "four days")

    def test_a_decimal_figure_matches(self):
        assert contains_fact("rose to N20.54 trillion", "20.54")


class TestCorrectness:
    def test_a_correct_answer_scores_correct(self):
        score = score_answer(question(), answer())
        assert score.correct
        assert not score.missing_facts
        assert not score.trapped_on

    def test_a_missing_fact_is_named(self):
        score = score_answer(question(), answer("The limit was reviewed upward."))
        assert not score.correct
        assert score.missing_facts == ["5,000,000"]

    def test_walking_into_the_trap_is_recorded_separately(self):
        """Stating a superseded figure is a different failure from omitting the
        right one, and the article reports it as its own rate."""
        score = score_answer(question(), answer("The limit is N500,000 per depositor."))
        assert not score.correct
        assert score.trapped_on == ["N500,000"]

    def test_stating_both_figures_still_fails(self):
        """An answer containing the right number and the superseded one has not
        resolved the contradiction, it has repeated it."""
        score = score_answer(
            question(),
            answer("The limit rose from N500,000 to N5,000,000."),
        )
        assert score.trapped_on
        assert not score.correct


class TestDeclining:
    def test_declining_an_absent_question_is_correct(self):
        score = score_answer(question(category=Category.ABSENT, must_contain=(),
                                      must_not_contain=(), source_files=()), declined())
        assert score.declined_correctly
        assert score.correct is None

    def test_correctness_is_not_recorded_for_an_absent_question(self):
        """There is no content to be correct about, and scoring it as a pass
        would let declining everything look like answering everything."""
        score = score_answer(question(category=Category.ABSENT, must_contain=(),
                                      must_not_contain=(), source_files=()), declined())
        assert score.correct is None

    def test_answering_an_absent_question_is_the_headline_failure(self):
        score = score_answer(
            question(category=Category.ABSENT, must_contain=(),
                     must_not_contain=(), source_files=()),
            answer("Ethiopia's scheme covers 100,000 birr."),
        )
        assert score.answered_when_it_should_not_have
        assert score.correct is False

    def test_declining_an_answerable_question_is_a_miss(self):
        score = score_answer(question(), declined())
        assert score.declined_wrongly
        assert score.correct is False


class TestHallucination:
    def test_a_claim_citing_nothing_retrieved_is_counted(self):
        score = score_answer(question(), answer(chunk_ids=("ghost",)),
                             retrieved_ids={"c1", "c2"})
        assert score.hallucinated_claims == 1
        assert score.hallucination_rate == 1.0

    def test_a_grounded_claim_is_not(self):
        score = score_answer(question(), answer(chunk_ids=("c1",)),
                             retrieved_ids={"c1", "c2"})
        assert score.hallucinated_claims == 0

    def test_the_check_is_skipped_rather_than_assumed_clean(self):
        """Without the retrieved ids there is nothing to check against, and
        reporting zero would be a claim rather than a measurement."""
        score = score_answer(question(), answer(chunk_ids=("ghost",)))
        assert score.hallucinated_claims == 0
        assert score.total_claims == 1


class TestScoreboard:
    def _board(self) -> Scoreboard:
        absent = question(category=Category.ABSENT, must_contain=(),
                          must_not_contain=(), source_files=())
        return Scoreboard(scores=[
            score_answer(question(id="a"), answer()),                    # correct
            score_answer(question(id="b"), answer("It is N500,000.")),   # trapped
            score_answer(absent, declined()),                            # good decline
            score_answer(question(id="d", category=Category.ABSENT, must_contain=(),
                                  must_not_contain=(), source_files=()),
                         answer("Zambia covers 50,000 kwacha.")),        # overreach
        ])

    def test_correctness_counts_only_answerable_questions(self):
        assert self._board().correctness == 0.5

    def test_overreach_is_reported_on_its_own(self):
        """The number this corpus exists to produce."""
        assert self._board().overreach_rate == 0.5

    def test_trap_rate_is_reported_on_its_own(self):
        assert self._board().trap_rate == 0.5

    def test_decline_precision_and_recall_are_both_reported(self):
        """A system declining exactly once, correctly, scores 100% precision
        and is useless. Recall is what catches that."""
        board = self._board()
        assert board.decline_precision == 1.0
        assert board.decline_recall == 0.5

    def test_a_system_that_declines_everything_does_not_look_good(self):
        board = Scoreboard(scores=[
            score_answer(question(id=f"q{i}"), declined()) for i in range(4)
        ])
        assert board.correctness == 0.0
        assert board.decline_precision == 0.0

    def test_empty_rates_do_not_divide_by_zero(self):
        board = Scoreboard()
        assert board.correctness == 0.0
        assert board.overreach_rate == 0.0
        assert board.hallucination_rate == 0.0

    def test_the_report_separates_the_categories(self):
        rendered = self._board().report()
        assert "overreach" in rendered
        assert "trap rate" in rendered
        assert "misleading" in rendered


class TestNumbersAreComparedAsNumbers:
    """The bug this replaced would have inverted the corpus's central result.

    Strip the separators from N5,000,000 and it contains "500000", so a
    substring check scored every correct answer to the coverage question as
    having stated the superseded figure instead.
    """

    def test_the_current_limit_is_not_read_as_the_superseded_one(self):
        assert not contains_fact("The limit is N5,000,000 per depositor.", "N500,000")

    def test_the_superseded_limit_still_matches_itself(self):
        assert contains_fact("The limit is N500,000.", "N500,000")

    def test_an_answer_stating_both_matches_both(self):
        text = "The limit rose from N500,000 to N5,000,000."
        assert contains_fact(text, "N500,000")
        assert contains_fact(text, "5,000,000")

    def test_scale_words_are_applied(self):
        assert contains_fact("rose to N20.54 trillion", "20.54")
        assert contains_fact("the limit is 5 million naira", "5,000,000")

    def test_a_longer_number_does_not_contain_a_shorter_one(self):
        """266,548,151 must not read as containing 548 or 151."""
        assert not contains_fact("266,548,151 accounts", "548")
        assert contains_fact("266,548,151 accounts", "266,548,151")

    def test_numbers_in_reads_values_not_strings(self):
        assert numbers_in("N5,000,000") == {5_000_000.0}
        assert 20_540_000_000_000.0 in numbers_in("N20.54 trillion")


class TestWordFactsRespectBoundaries:
    def test_a_hyphenated_neighbour_does_not_match(self):
        """"twenty-four days" is a different duration from "four days"."""
        assert not contains_fact("paid within twenty-four days", "four days")

    def test_trailing_punctuation_does_not_prevent_a_match(self):
        assert contains_fact("paid within four days.", "four days")
