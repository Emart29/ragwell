"""The hand-labelled questions both arms answer.

Labelled against four NDIC annual reports — 2020, 2021, 2022 and 2024 — which
are real regulatory documents rather than a corpus written for this benchmark.
That distinction matters most for one category:

**The MISLEADING questions are not planted.** A synthetic corpus needs traps
built into it deliberately, and a trap you designed is a trap you know the shape
of. These reports contradict each other the way real document sets do, because
each states the current position and every earlier one states the position as it
then stood. The clearest case: in 2024 the Maximum Deposit Insurance Coverage
for a Deposit Money Bank rose from N500,000 to N5,000,000. Three of the four
reports in this corpus therefore state a figure that is now wrong, and they state
it as confidently as the 2024 report states the right one. Asking "what is the
coverage limit" against this corpus is a genuinely hard retrieval problem that
nobody had to invent.

Four categories, failing in different ways:

* ``DIRECT`` — the answer sits in one report. Measures retrieval and reading.
* ``SYNTHESIS`` — needs two facts, often from different reports.
* ``ABSENT`` — the corpus does not contain the answer, so declining is correct.
* ``MISLEADING`` — an earlier report answers it differently and confidently.

Answers are stored as facts a correct response must contain, not prose to match
word for word: scoring against an exact string would measure phrasing, and both
arms are free to phrase things differently.

Every label in this file was read out of the extracted text and checked back
against it by ``tests/test_eval_questions.py``. A wrong label is worse than a
bug — a bug shows up as a failure, a wrong label shows up as a confident number
that is silently incorrect.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

#: The reports, newest first. The filenames are the stems of the downloaded
#: PDFs; ``fetch_corpus.py`` writes exactly these.
REPORTS = {
    2024: "2024-Annual-Report",
    2022: "2022-Annual-Report",
    2021: "2021-Annual-Report",
    2020: "NDIC-2020-Annual-Report",
}

#: The report holding the current position. A question about "now" is labelled
#: against this one, and the others are what make it hard.
CURRENT_REPORT = REPORTS[2024]


class Category(str, Enum):
    DIRECT = "direct"
    SYNTHESIS = "synthesis"
    ABSENT = "absent"
    MISLEADING = "misleading"


class Difficulty(str, Enum):
    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"


@dataclass(frozen=True)
class Question:
    """One question and what a correct answer contains."""

    id: str
    text: str
    category: Category
    difficulty: Difficulty
    #: Report stems the answer is drawn from. Used to check retrieval found the
    #: right documents, separately from whether the answer was right.
    source_files: tuple[str, ...] = ()
    #: Facts a correct answer must contain, as substrings sought after
    #: whitespace and case normalisation. Short and factual.
    must_contain: tuple[str, ...] = ()
    #: Facts marking a wrong answer — usually the superseded figure.
    must_not_contain: tuple[str, ...] = ()
    #: Why this question is in the set.
    tests: str = ""
    #: Included in the reduced set used at the largest corpus sizes, where
    #: sending the whole corpus per question is the dominant cost.
    core: bool = False

    @property
    def should_decline(self) -> bool:
        return self.category is Category.ABSENT


QUESTIONS: list[Question] = [
    # ---------------------------------------------------------------- DIRECT
    Question(
        id="mdic_dmb_2024",
        text=(
            "What is the maximum deposit insurance coverage per depositor per "
            "Deposit Money Bank following the 2024 review?"
        ),
        category=Category.DIRECT,
        difficulty=Difficulty.EASY,
        source_files=(REPORTS[2024],),
        must_contain=("5,000,000",),
        tests="the headline figure of the corpus; a system failing this is broken",
        core=True,
    ),
    Question(
        id="dmb_insured_deposits_2024",
        text="What were the insured deposits of Deposit Money Banks at end-December 2024?",
        category=Category.DIRECT,
        difficulty=Difficulty.MODERATE,
        source_files=(REPORTS[2024],),
        must_contain=("20.54",),
        tests="a figure stated once, in trillions, beside the prior year's",
        core=True,
    ),
    Question(
        id="fully_insured_accounts_2024",
        text="How many fully insured bank accounts were there at end-December 2024?",
        category=Category.DIRECT,
        difficulty=Difficulty.MODERATE,
        source_files=(REPORTS[2024],),
        must_contain=("266,548,151",),
        tests="a long exact number, where a near-miss is a wrong answer",
    ),
    Question(
        id="heritage_payout_speed",
        text=(
            "How quickly did the NDIC begin paying verified depositors after "
            "Heritage Bank's licence was revoked?"
        ),
        category=Category.DIRECT,
        difficulty=Difficulty.MODERATE,
        source_files=(REPORTS[2024],),
        must_contain=("four days",),
        tests="a fact expressed in words rather than digits",
    ),
    Question(
        id="mfb_limit_2024",
        text=(
            "What is the maximum deposit insurance coverage per depositor per "
            "Microfinance Bank after the 2024 review?"
        ),
        category=Category.DIRECT,
        difficulty=Difficulty.MODERATE,
        source_files=(REPORTS[2024],),
        must_contain=("2,000,000",),
        must_not_contain=("200,000",),
        tests=(
            "three limits changed at once and each has a different value; "
            "retrieving the paragraph is not enough, it must be read"
        ),
        core=True,
    ),

    # ------------------------------------------------------------- SYNTHESIS
    Question(
        id="coverage_effect_on_deposits",
        text=(
            "By how much did the insured deposits of Deposit Money Banks rise "
            "in 2024, and what caused the rise?"
        ),
        category=Category.SYNTHESIS,
        difficulty=Difficulty.HARD,
        source_files=(REPORTS[2024],),
        must_contain=("168.19",),
        tests=(
            "requires the percentage and its stated cause, which is the "
            "coverage review; either half alone is an incomplete answer"
        ),
        core=True,
    ),
    Question(
        id="limits_by_institution_type",
        text=(
            "After the 2024 review, how does the coverage limit differ between "
            "a Deposit Money Bank and a Primary Mortgage Bank?"
        ),
        category=Category.SYNTHESIS,
        difficulty=Difficulty.HARD,
        source_files=(REPORTS[2024],),
        must_contain=("5,000,000", "2,000,000"),
        tests="two figures from one passage that a summariser tends to collapse into one",
        core=True,
    ),
    Question(
        id="dmb_count_change",
        text=(
            "How did the number of Deposit Money Banks in operation change "
            "between 2021 and 2022?"
        ),
        category=Category.SYNTHESIS,
        difficulty=Difficulty.HARD,
        source_files=(REPORTS[2021], REPORTS[2022]),
        must_contain=("32", "35"),
        tests=(
            "the only question needing a fact from two different reports; "
            "this is where top-k truncation bites"
        ),
    ),

    # ---------------------------------------------------------------- ABSENT
    Question(
        id="ethiopia_scheme",
        text="What does the report say about Ethiopia's deposit insurance scheme?",
        category=Category.ABSENT,
        difficulty=Difficulty.MODERATE,
        tests=(
            "the reports discuss several African schemes but never Ethiopia, so "
            "a plausible-sounding answer is fabricated"
        ),
        core=True,
    ),
    Question(
        id="sovereign_wealth",
        text="How much of the Deposit Insurance Fund is held in sovereign wealth instruments?",
        category=Category.ABSENT,
        difficulty=Difficulty.HARD,
        tests=(
            "the corpus discusses the fund at length and never mentions "
            "sovereign wealth; the surrounding detail invites an answer"
        ),
        core=True,
    ),
    Question(
        id="premium_tax",
        text="What is the insurance premium tax rate applied to insured institutions?",
        category=Category.ABSENT,
        difficulty=Difficulty.HARD,
        tests=(
            "premiums are discussed in detail but no such tax exists here; the "
            "vocabulary overlap makes retrieval return confident-looking chunks"
        ),
    ),
    Question(
        id="zambia_comparison",
        text="How does Nigeria's coverage limit compare with Zambia's?",
        category=Category.ABSENT,
        difficulty=Difficulty.MODERATE,
        tests="Zambia appears nowhere; comparison questions invite invention",
    ),

    # ------------------------------------------------------------ MISLEADING
    Question(
        id="mdic_current",
        text="What is the maximum deposit insurance coverage for a depositor in a Deposit Money Bank?",
        category=Category.MISLEADING,
        difficulty=Difficulty.HARD,
        source_files=(REPORTS[2024],),
        must_contain=("5,000,000",),
        must_not_contain=("N500,000",),
        tests=(
            "the central trap of this corpus and it was not planted: three of "
            "four reports state N500,000 with equal confidence, and only the "
            "2024 one is current"
        ),
        core=True,
    ),
    Question(
        id="mfb_limit_unqualified",
        text="What is the deposit insurance limit for microfinance bank depositors?",
        category=Category.MISLEADING,
        difficulty=Difficulty.HARD,
        source_files=(REPORTS[2024],),
        must_contain=("2,000,000",),
        must_not_contain=("N200,000",),
        tests="the superseded N200,000 appears in three reports and reads as settled",
        core=True,
    ),
    Question(
        id="dmb_count_now",
        text="How many Deposit Money Banks are in operation?",
        category=Category.MISLEADING,
        difficulty=Difficulty.MODERATE,
        source_files=(REPORTS[2024],),
        must_contain=("35",),
        must_not_contain=("32 DMBs",),
        tests="the 2021 report says 32 and states it as plainly as the later ones say 35",
    ),
    Question(
        id="dmb_deposits_prior_year",
        text="What were the insured deposits of Deposit Money Banks at end-December 2023?",
        category=Category.MISLEADING,
        difficulty=Difficulty.HARD,
        source_files=(REPORTS[2024],),
        must_contain=("7.66",),
        must_not_contain=("20.54",),
        tests=(
            "the 2024 figure sits in the same sentence and is the more "
            "prominent number; answering it is a subtle, checkable error"
        ),
        core=True,
    ),
]

QUESTIONS_BY_ID = {q.id: q for q in QUESTIONS}

#: Used at the largest corpus sizes, where the long-context arm sends the whole
#: corpus per question and cost scales with the product of the two. Chosen to
#: keep every category represented rather than to keep the easy ones.
CORE_QUESTIONS = [q for q in QUESTIONS if q.core]


def by_category(category: Category) -> list[Question]:
    return [q for q in QUESTIONS if q.category is category]


def category_counts() -> dict[str, int]:
    return {c.value: len(by_category(c)) for c in Category}
