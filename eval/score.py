"""Scores an answer against its hand-written label.

Deterministic where it can be, judged only where it cannot. Five measurements,
kept separate because they fail for different reasons and a single "accuracy"
would hide which:

* **Citation accuracy** — the share of claims whose citation verifies. Free.
* **Hallucination rate** — the share of claims supported by no retrieved chunk.
  The failure that matters most, and checkable without judgement.
* **Correctness** — does the answer contain the labelled facts, and avoid the
  trap? Deterministic for a factual question, and this corpus is all figures.
* **Declined correctly** — an ABSENT question answered with a refusal.
* **Declined wrongly** — an answerable question the system gave up on.

The last two are never summed with the first three. A system that declines
everything has a perfect hallucination rate and is useless, and a single figure
would let it look good.

The correctness judge is deliberately a *fallback*. Where a label says the
answer must contain "5,000,000", a string check settles it and no model is
needed. The judge runs only where the deterministic check is ambiguous, and its
agreement with the labels is measured in ``score_eval.py`` before any number it
produces is quoted — the same discipline as the entailment judge, and for the
same reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.answer.contract import GroundedAnswer, InsufficientEvidence
from app.answer.verify import VerificationReport
from app.logging_config import get_logger
from eval.questions import Category, Question

logger = get_logger(__name__)


def normalise(text: str) -> str:
    """Fold whitespace and case so a fact can be matched however it is written."""
    return re.sub(r"\s+", " ", text.lower()).strip()


#: A number as written in these reports: digits, optional thousands separators,
#: optional decimal part.
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

#: Multipliers a report uses instead of writing the zeros out.
_SCALES = {"million": 1_000_000, "billion": 1_000_000_000, "trillion": 1_000_000_000_000}


def _numeric_value(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def numbers_in(text: str) -> set[float]:
    """Every number a text states, as values rather than as strings.

    Values, because a substring comparison is wrong in a way that quietly
    inverts this corpus: strip the separators from ``N5,000,000`` and it
    contains ``500000``, so the current coverage limit reads as though it
    stated the superseded one. Every correct answer to the central question
    would have scored as walking into the trap.

    A trailing scale word is applied, so "20.54 trillion" and "20,540,000,000,000"
    are the same value, and "5 million" matches "5,000,000".
    """
    lowered = normalise(text)
    values: set[float] = set()
    for match in _NUMBER.finditer(lowered):
        value = _numeric_value(match.group())
        if value is None:
            continue
        values.add(value)
        tail = lowered[match.end():match.end() + 12].strip()
        for word, scale in _SCALES.items():
            if tail.startswith(word):
                values.add(value * scale)
    return values


def contains_fact(answer_text: str, fact: str) -> bool:
    """Whether an answer states a labelled fact.

    A fact that is a number is compared as a number: separators and scale words
    vary between writers and none of them change the value. A fact that is not
    a number is matched on word boundaries, so "four days" does not match
    "twenty-four days".
    """
    fact_numbers = numbers_in(fact)
    if fact_numbers:
        # Each label entry carries one figure, so any match settles it.
        # A label needing two lists them separately, each checked here.
        return bool(fact_numbers & numbers_in(answer_text))

    # A hyphen is a word boundary to \b, so \bfour days\b matches
    # "twenty-four days" — a different duration. The lookarounds exclude
    # a neighbouring hyphen as well as a neighbouring letter.
    pattern = r"(?<![-\w])" + re.escape(normalise(fact)) + r"(?![-\w])"
    return re.search(pattern, normalise(answer_text)) is not None


@dataclass
class AnswerScore:
    """How one answer fared against its label."""

    question_id: str
    category: Category
    declined: bool
    #: None when the question is ABSENT: there is no content to be correct about.
    correct: bool | None = None
    #: Facts the label required that the answer did not state.
    missing_facts: list[str] = field(default_factory=list)
    #: Traps the answer walked into — a superseded figure, usually.
    trapped_on: list[str] = field(default_factory=list)
    citation_accuracy: float = 0.0
    hallucinated_claims: int = 0
    total_claims: int = 0
    #: True when the deterministic check could not settle it and a judge ran.
    judged: bool = False

    @property
    def declined_correctly(self) -> bool:
        return self.declined and self.category is Category.ABSENT

    @property
    def declined_wrongly(self) -> bool:
        return self.declined and self.category is not Category.ABSENT

    @property
    def answered_when_it_should_not_have(self) -> bool:
        """Answered an ABSENT question. The failure the category exists for."""
        return not self.declined and self.category is Category.ABSENT

    @property
    def hallucination_rate(self) -> float:
        if not self.total_claims:
            return 0.0
        return self.hallucinated_claims / self.total_claims

    def problems(self) -> list[str]:
        issues = []
        if self.answered_when_it_should_not_have:
            issues.append("answered a question the corpus cannot support")
        if self.declined_wrongly:
            issues.append("declined an answerable question")
        if self.missing_facts:
            issues.append(f"missing: {', '.join(self.missing_facts)}")
        if self.trapped_on:
            issues.append(f"stated a superseded figure: {', '.join(self.trapped_on)}")
        if self.hallucinated_claims:
            issues.append(f"{self.hallucinated_claims} ungrounded claims")
        return issues


def score_answer(
    question: Question,
    answer: GroundedAnswer | InsufficientEvidence,
    verification: VerificationReport | None = None,
    retrieved_ids: set[str] | None = None,
) -> AnswerScore:
    """Score one answer against its label.

    Args:
        question: The labelled question.
        answer: What the system produced.
        verification: The citation check, if it was run.
        retrieved_ids: Chunk ids the retriever supplied, for the hallucination
            check. Without them that check is skipped rather than assumed clean.

    Returns:
        The score, with each dimension separate.
    """
    declined = isinstance(answer, InsufficientEvidence)
    score = AnswerScore(
        question_id=question.id,
        category=question.category,
        declined=declined,
    )

    if declined:
        # An ABSENT question has no content to be correct about, so correctness
        # stays None rather than being recorded as a pass or a fail. Declining
        # an answerable question is a miss, recorded as such.
        score.correct = None if question.category is Category.ABSENT else False
        return score

    text = answer.text()
    score.total_claims = len(answer.claims)

    if question.category is Category.ABSENT:
        # Answering at all is the failure. Whatever it said is unsupportable,
        # because the corpus does not contain the answer.
        score.correct = False
    else:
        score.missing_facts = [
            fact for fact in question.must_contain if not contains_fact(text, fact)
        ]
        score.trapped_on = [
            trap for trap in question.must_not_contain if contains_fact(text, trap)
        ]
        score.correct = not score.missing_facts and not score.trapped_on

    if verification is not None:
        score.citation_accuracy = verification.cheaply_verified_rate

    if retrieved_ids is not None:
        score.hallucinated_claims = sum(
            1
            for claim in answer.claims
            if not set(claim.chunk_ids) & retrieved_ids
        )

    return score


@dataclass
class Scoreboard:
    """Every score for one run, with the rates kept apart."""

    scores: list[AnswerScore] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.scores)

    @property
    def answerable(self) -> list[AnswerScore]:
        return [s for s in self.scores if s.category is not Category.ABSENT]

    @property
    def absent(self) -> list[AnswerScore]:
        return [s for s in self.scores if s.category is Category.ABSENT]

    @property
    def correctness(self) -> float:
        """Share of answerable questions answered correctly."""
        if not self.answerable:
            return 0.0
        return sum(1 for s in self.answerable if s.correct) / len(self.answerable)

    @property
    def decline_precision(self) -> float:
        """Of the questions it declined, how many should have been declined.

        Reported next to recall rather than instead of it: a system that
        declines exactly once, correctly, scores 100% here and is useless.
        """
        declined = [s for s in self.scores if s.declined]
        if not declined:
            return 0.0
        return sum(1 for s in declined if s.declined_correctly) / len(declined)

    @property
    def decline_recall(self) -> float:
        """Of the questions it should have declined, how many it did."""
        if not self.absent:
            return 0.0
        return sum(1 for s in self.absent if s.declined) / len(self.absent)

    @property
    def overreach_rate(self) -> float:
        """Share of ABSENT questions it answered anyway.

        The number this corpus exists to produce: how often a system invents an
        answer when the documents cannot support one.
        """
        if not self.absent:
            return 0.0
        return sum(1 for s in self.absent if not s.declined) / len(self.absent)

    @property
    def trap_rate(self) -> float:
        """Share of MISLEADING questions answered with a superseded figure."""
        misleading = [s for s in self.scores if s.category is Category.MISLEADING]
        if not misleading:
            return 0.0
        return sum(1 for s in misleading if s.trapped_on) / len(misleading)

    @property
    def citation_accuracy(self) -> float:
        answered = [s for s in self.scores if not s.declined and s.total_claims]
        if not answered:
            return 0.0
        return sum(s.citation_accuracy for s in answered) / len(answered)

    @property
    def hallucination_rate(self) -> float:
        claims = sum(s.total_claims for s in self.scores)
        if not claims:
            return 0.0
        return sum(s.hallucinated_claims for s in self.scores) / claims

    def by_category(self) -> dict[str, dict[str, float]]:
        out = {}
        for category in Category:
            cut = [s for s in self.scores if s.category is category]
            if not cut:
                continue
            if category is Category.ABSENT:
                out[category.value] = {
                    "n": len(cut),
                    "declined": sum(1 for s in cut if s.declined) / len(cut),
                }
            else:
                out[category.value] = {
                    "n": len(cut),
                    "correct": sum(1 for s in cut if s.correct) / len(cut),
                }
        return out

    def report(self) -> str:
        lines = [
            f"correctness      {self.correctness:.0%} over "
            f"{len(self.answerable)} answerable questions",
            f"overreach        {self.overreach_rate:.0%} of "
            f"{len(self.absent)} unanswerable questions were answered anyway",
            f"trap rate        {self.trap_rate:.0%} stated a superseded figure",
            f"citation acc.    {self.citation_accuracy:.0%}",
            f"hallucination    {self.hallucination_rate:.0%} of claims cite "
            "nothing retrieved",
            "",
            "by category:",
        ]
        for name, stats in self.by_category().items():
            metric = "declined" if "declined" in stats else "correct"
            lines.append(
                f"  {name:<12} n={int(stats['n']):<3} {metric} {stats[metric]:.0%}"
            )
        return "\n".join(lines)
