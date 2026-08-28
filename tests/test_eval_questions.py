"""Tests for the evaluation question set.

These check the labels against the corpus rather than checking the system. A
wrong label is worse than a bug: a bug shows up as a failure, and a wrong label
shows up as a confident number that is silently incorrect.

The corpus is four NDIC annual reports, which are not committed — they are
roughly 450MB and are reconstructed by ``eval/fetch_corpus.py`` from a manifest.
Tests needing the text skip when it is absent rather than failing, so a clone
without the corpus still has a green suite; the label checks are the ones that
matter and they say plainly when they could not run.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from eval.fetch_corpus import DOCUMENTS, FETCHED
from eval.questions import (
    CORE_QUESTIONS,
    CURRENT_REPORT,
    QUESTIONS,
    QUESTIONS_BY_ID,
    REPORTS,
    Category,
    Difficulty,
    by_category,
)

EXTRACTED = pathlib.Path(__file__).resolve().parent.parent / "eval" / "extracted"


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


@pytest.fixture(scope="module")
def corpus() -> dict[str, str]:
    if not EXTRACTED.exists() or not list(EXTRACTED.glob("*.txt")):
        pytest.skip("corpus text absent; run eval/fetch_corpus.py and extract")
    return {
        path.stem: normalise(path.read_text(encoding="utf-8"))
        for path in EXTRACTED.glob("*.txt")
    }


class TestTheSetIsUsable:
    def test_every_category_is_represented(self):
        """A set without ABSENT and MISLEADING measures retrieval and nothing
        else, and a system that always answers scores perfectly on it."""
        for category in Category:
            assert by_category(category), category

    def test_ids_are_unique(self):
        assert len(QUESTIONS_BY_ID) == len(QUESTIONS)

    def test_every_question_states_what_it_tests(self):
        for question in QUESTIONS:
            assert question.tests, question.id

    def test_difficulty_varies(self):
        assert {q.difficulty for q in QUESTIONS} == set(Difficulty)

    def test_the_hard_categories_carry_real_weight(self):
        hard = by_category(Category.ABSENT) + by_category(Category.MISLEADING)
        assert len(hard) >= len(QUESTIONS) / 3


class TestTheCoreSubset:
    """Used at the largest corpus sizes, where the long-context arm sends the
    whole corpus per question and cost scales with the product of the two."""

    def test_it_is_smaller_than_the_full_set(self):
        assert 0 < len(CORE_QUESTIONS) < len(QUESTIONS)

    def test_it_keeps_every_category(self):
        """Dropping ABSENT or MISLEADING at the expensive sizes would make the
        top of the sweep measure something easier than the bottom."""
        covered = {q.category for q in CORE_QUESTIONS}
        assert covered == set(Category)

    def test_it_is_not_just_the_easy_ones(self):
        assert any(q.difficulty is Difficulty.HARD for q in CORE_QUESTIONS)


class TestLabelsMatchTheCorpus:
    def test_every_named_source_exists(self, corpus):
        for question in QUESTIONS:
            for stem in question.source_files:
                assert stem in corpus, f"{question.id} names {stem}"

    def test_required_facts_are_in_the_named_source(self, corpus):
        """A required fact absent from its own source would fail every correct
        answer, and the failure would look like the system's."""
        for question in QUESTIONS:
            for fact in question.must_contain:
                found = any(
                    fact.lower() in corpus[stem] for stem in question.source_files
                )
                assert found, f"{question.id}: {fact!r} not in {question.source_files}"

    def test_every_trap_exists_somewhere_in_the_corpus(self, corpus):
        """A must_not_contain value appearing nowhere traps nothing, and makes
        the question look harder than it is."""
        whole = " ".join(corpus.values())
        for question in QUESTIONS:
            for trap in question.must_not_contain:
                assert trap.lower() in whole, f"{question.id}: {trap!r} traps nothing"

    def test_a_trap_is_never_also_a_required_fact(self):
        for question in QUESTIONS:
            overlap = {t.lower() for t in question.must_contain} & {
                t.lower() for t in question.must_not_contain
            }
            assert not overlap, f"{question.id}: {overlap}"


class TestAbsentQuestions:
    def test_they_declare_no_source(self):
        for question in by_category(Category.ABSENT):
            assert not question.source_files, question.id

    def test_they_require_no_facts(self):
        for question in by_category(Category.ABSENT):
            assert not question.must_contain, question.id

    def test_declining_is_correct_only_for_them(self):
        for question in QUESTIONS:
            assert question.should_decline is (
                question.category is Category.ABSENT
            ), question.id

    @pytest.mark.parametrize(
        "term", ["ethiopia", "zambia", "sovereign wealth", "insurance premium tax"]
    )
    def test_the_absent_subjects_really_are_absent(self, corpus, term):
        """Checked rather than assumed. An ABSENT question whose subject turns
        out to be in the corpus would score a correct answer as a hallucination."""
        whole = " ".join(corpus.values())
        assert term not in whole, f"{term!r} is in the corpus after all"


class TestMisleadingQuestions:
    """These were not planted. The reports contradict each other because each
    states the position as it stood, which is how real document sets behave."""

    def test_each_one_has_a_trap(self):
        for question in by_category(Category.MISLEADING):
            assert question.must_not_contain, question.id

    def test_the_coverage_review_really_did_change_the_figure(self, corpus):
        """The central trap of this corpus: three of four reports state a limit
        that the 2024 review superseded."""
        current = corpus[CURRENT_REPORT]
        assert "n5,000,000" in current
        superseded = [
            stem for stem in corpus if stem != CURRENT_REPORT and "n500,000" in corpus[stem]
        ]
        assert len(superseded) >= 2, "the trap depends on older reports stating N500,000"

    def test_the_older_reports_state_the_old_limit_confidently(self, corpus):
        """If they hedged, the question would be easy. They do not."""
        for stem, text in corpus.items():
            if stem == CURRENT_REPORT and "n500,000" in text:
                continue
        assert "n500,000" in corpus[REPORTS[2022]]

    def test_the_dmb_count_genuinely_differs_across_years(self, corpus):
        assert "32 dmbs" in corpus[REPORTS[2021]]
        assert "35 dmbs" in corpus[REPORTS[2022]]


class TestSynthesisQuestions:
    def test_at_least_one_spans_two_reports(self):
        """Otherwise nothing measures whether retrieval returns both halves,
        which is where top-k truncation bites."""
        spanning = [
            q for q in by_category(Category.SYNTHESIS) if len(q.source_files) > 1
        ]
        assert spanning


class TestTheManifest:
    def test_every_document_has_a_checksum(self):
        """A regulator that replaces a PDF would otherwise change the corpus
        under a benchmark claiming to be reproducible."""
        for document in DOCUMENTS:
            assert document.sha256, document.filename

    def test_the_fetch_date_is_recorded(self):
        """The corpus is only the corpus as of a date."""
        assert FETCHED

    def test_the_questions_and_the_manifest_agree(self):
        manifest_stems = {d.filename.removesuffix(".pdf") for d in DOCUMENTS}
        assert set(REPORTS.values()) == manifest_stems
