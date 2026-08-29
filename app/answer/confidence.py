"""Scores how much an answer should be trusted, from observable evidence.

Not from asking the model how sure it is. A self-reported confidence is another
generation, subject to the same failure the rest of this project measures, and
it is available at exactly the moment the model is most likely to be wrong for
the same reason it produced a bad answer. It is included as one input here, and
whether it correlates with being right is itself a measurement rather than an
assumption.

Four signals, all of them things the system can see for itself:

* **Retrieval strength** — how well the top chunks scored. Weak retrieval means
  the answer is built on whatever happened to come back.
* **Citation verification** — the share of claims whose citation actually holds,
  from the free rungs of the verification ladder.
* **Coverage** — how much of the question the claims address. An answer that
  ignores half the question is confident about half a question.
* **Agreement** — whether independent chunks support the same claims. One chunk
  supporting everything is a single point of failure.

The number these produce is meaningless until it is calibrated. A confidence
score nobody has plotted against measured accuracy is decoration, so
``calibration.py`` exists and the threshold in the fallback layer is read off
that curve rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.answer.contract import GroundedAnswer, InsufficientEvidence
from app.answer.verify import VerificationReport, content_words
from app.logging_config import get_logger
from app.retrieval.types import SearchResult

logger = get_logger(__name__)

#: How the signals combine. Verification is weighted highest because it is the
#: only one measuring whether the answer is actually supported rather than
#: whether the conditions for a good answer were present.
WEIGHTS = {
    "retrieval": 0.25,
    "verification": 0.40,
    "coverage": 0.20,
    "agreement": 0.15,
}

#: Retrieval score treated as strong. Scores are not comparable across
#: strategies — a reranker score and a cosine similarity mean different things —
#: so this normalises rather than thresholds, and the calibration curve is what
#: gives the result meaning.
STRONG_RETRIEVAL_SCORE = 0.7


@dataclass
class ConfidenceBreakdown:
    """The score, and every signal that produced it.

    Kept separate rather than collapsed into one number, because a low score
    with strong retrieval and failed verification is a different problem from a
    low score with nothing retrieved, and the caller can act on the difference.
    """

    score: float
    retrieval: float = 0.0
    verification: float = 0.0
    coverage: float = 0.0
    agreement: float = 0.0
    #: What the model said about itself, when it was asked. Recorded and not
    #: used, until calibration says whether it is worth anything.
    self_reported: float | None = None
    notes: list[str] = field(default_factory=list)

    def explain(self) -> str:
        parts = [
            f"retrieval {self.retrieval:.2f}",
            f"verification {self.verification:.2f}",
            f"coverage {self.coverage:.2f}",
            f"agreement {self.agreement:.2f}",
        ]
        line = f"confidence {self.score:.2f} — " + ", ".join(parts)
        if self.notes:
            line += "\n  " + "\n  ".join(self.notes)
        return line


def retrieval_strength(results: list[SearchResult]) -> float:
    """How well the retrieved chunks scored, as a 0-1 signal.

    Uses the top score rather than the mean: a strong first hit with weak tail
    is a good retrieval, and averaging would punish it for returning extra
    candidates the reranker sensibly ranked low.
    """
    if not results:
        return 0.0
    top = max(r.score for r in results)
    return max(0.0, min(top / STRONG_RETRIEVAL_SCORE, 1.0))


def coverage(question: str, answer: GroundedAnswer) -> float:
    """Share of the question's content words the answer engages with.

    Crude on purpose. It is not measuring whether the answer is correct — that
    is what the labels are for — only whether it addressed what was asked. An
    answer that ignores half the question is confident about half a question.
    """
    asked = content_words(question)
    if not asked:
        return 1.0
    answered = content_words(answer.text())
    return len(asked & answered) / len(asked)


def source_agreement(answer: GroundedAnswer) -> float:
    """Whether the answer rests on more than one source.

    One chunk supporting every claim is a single point of failure: if that
    chunk was retrieved in error, the whole answer goes with it. Scored as the
    share of distinct chunks cited relative to the number of claims, capped at
    one, so citing two chunks across two claims scores full marks and citing
    one chunk across four does not.
    """
    if not answer.claims:
        return 0.0
    distinct = len(answer.cited_chunk_ids)
    return min(distinct / len(answer.claims), 1.0)


def score_answer(
    question: str,
    answer: GroundedAnswer | InsufficientEvidence,
    results: list[SearchResult],
    verification: VerificationReport | None = None,
    self_reported: float | None = None,
) -> ConfidenceBreakdown:
    """Score an answer from what can be observed about it.

    Args:
        question: The question asked.
        answer: The answer produced.
        results: Chunks the retriever returned.
        verification: The citation check, if it has been run. Without it the
            verification signal is unavailable rather than assumed good.
        self_reported: Any confidence the model volunteered. Recorded, not used.

    Returns:
        The score with its components, so a low score can be acted on rather
        than merely noticed.
    """
    breakdown = ConfidenceBreakdown(score=0.0, self_reported=self_reported)

    if isinstance(answer, InsufficientEvidence):
        # A decline is not a low-confidence answer; it is the absence of one.
        # Scoring it low would make "correctly declined" and "answered badly"
        # indistinguishable downstream.
        breakdown.notes.append("declined; confidence does not apply")
        return breakdown

    breakdown.retrieval = retrieval_strength(results)
    breakdown.coverage = coverage(question, answer)
    breakdown.agreement = source_agreement(answer)

    if verification is None:
        # Redistribute rather than substituting a default. A missing check is
        # not a passing one, and assuming otherwise would let an unverified
        # answer score as high as a verified one.
        breakdown.notes.append("citations not verified; score omits that signal")
        usable = {k: v for k, v in WEIGHTS.items() if k != "verification"}
        total = sum(usable.values())
        breakdown.score = sum(
            getattr(breakdown, name) * weight / total
            for name, weight in usable.items()
        )
    else:
        breakdown.verification = verification.cheaply_verified_rate
        if verification.judge_ran:
            # Where the judge ran, trust it over the string check: a correct
            # paraphrase fails containment and is still sound.
            breakdown.verification = max(
                breakdown.verification, verification.entailed_rate
            )
        breakdown.score = sum(
            getattr(breakdown, name) * weight for name, weight in WEIGHTS.items()
        )

    if breakdown.retrieval < 0.5:
        breakdown.notes.append("weak retrieval; the answer rests on poor matches")
    if breakdown.coverage < 0.5:
        breakdown.notes.append("much of the question is unaddressed")
    if breakdown.agreement < 0.5:
        breakdown.notes.append("most claims rest on a single chunk")

    breakdown.score = round(max(0.0, min(breakdown.score, 1.0)), 4)
    return breakdown
