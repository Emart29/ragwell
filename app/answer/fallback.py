"""What to do when the first answer is not good enough.

A system that answers regardless of confidence has no use for a confidence
score. This is the part that acts on it, in order of preference:

1. **Re-query** — reformulate the question and retrieve again. Ragwell already
   has query expansion and HyDE, so this costs a retrieval rather than new code.
2. **Widen** — more chunks, and a different strategy. Cheap, and often the whole
   problem: the answer was in the corpus and outside the top five.
3. **Decline** — return ``InsufficientEvidence`` naming what was missing.

The threshold is deliberately unset. It is a measured value, read off the
calibration curve in ``calibration.py`` once there are enough scored outcomes to
plot one, and a number guessed before that measurement would be presented later
as though it had been derived. Until it is configured, this layer runs in
``observe`` mode: it scores every answer, records the score, accepts it, and
changes nothing. It does not claim it *would* have escalated — with no threshold
there is nothing to compare against, and saying otherwise would invent the very
number that is missing.

Every attempt is recorded. "How often does it decline?" is a number the article
needs, and a system that never declines is not being honest about what it knows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.answer.confidence import ConfidenceBreakdown, score_answer
from app.answer.contract import GroundedAnswer, InsufficientEvidence
from app.answer.generate import AnswerGenerator, GenerationResult
from app.answer.verify import CitationVerifier, VerificationReport
from app.logging_config import get_logger

logger = get_logger(__name__)


class Action(str, Enum):
    """What the fallback layer did, or would have done."""

    ACCEPTED = "accepted"
    REQUERIED = "requeried"
    WIDENED = "widened"
    DECLINED = "declined"
    #: The first answer was already a decline; nothing to escalate.
    ALREADY_DECLINED = "already_declined"


@dataclass
class Attempt:
    """One pass at answering, and how it scored."""

    action: Action
    strategy: str
    top_k: int
    confidence: float
    claim_count: int
    declined: bool
    note: str = ""


@dataclass
class AnsweredQuestion:
    """The final answer, and the whole path taken to reach it."""

    question: str
    generation: GenerationResult
    confidence: ConfidenceBreakdown
    verification: VerificationReport | None = None
    attempts: list[Attempt] = field(default_factory=list)
    #: True when no threshold was set, so every answer was accepted and the
    #: score recorded without being judged against anything.
    observed_only: bool = False

    @property
    def answer(self):
        return self.generation.answer

    @property
    def declined(self) -> bool:
        return isinstance(self.answer, InsufficientEvidence)

    @property
    def escalated(self) -> bool:
        return len(self.attempts) > 1

    @property
    def final_action(self) -> Action:
        return self.attempts[-1].action if self.attempts else Action.ACCEPTED

    def summary(self) -> str:
        path = " -> ".join(a.action.value for a in self.attempts)
        mode = " [observing only]" if self.observed_only else ""
        return (
            f"{path}{mode} | confidence {self.confidence.score:.2f} | "
            f"{'declined' if self.declined else f'{self.generation.claim_count} claims'}"
        )


class FallbackPolicy:
    """Decides whether an answer is good enough, and what to do if not."""

    def __init__(
        self,
        retriever,
        generator: AnswerGenerator | None = None,
        verifier: CitationVerifier | None = None,
        threshold: float | None = None,
        widen_to: int = 15,
        requery_strategy: str = "expanded",
    ) -> None:
        """
        Args:
            retriever: Ragwell's retriever, used as-is.
            generator: Answer generator. Built with defaults when omitted.
            verifier: Citation verifier. ``None`` means confidence is scored
                without the verification signal rather than with it assumed good.
            threshold: Confidence below which to escalate. ``None`` — the
                default — means the value has not been measured yet, and the
                policy observes without acting. Set it from
                ``CalibrationCurve.threshold_for`` once there is a curve.
            widen_to: How many chunks the widened retrieval asks for.
            requery_strategy: Retrieval strategy for the re-query attempt.
        """
        self.retriever = retriever
        self.generator = generator or AnswerGenerator()
        self.verifier = verifier
        self.threshold = threshold
        self.widen_to = widen_to
        self.requery_strategy = requery_strategy

        if threshold is None:
            logger.info(
                "no confidence threshold set; every answer will be accepted and "
                "its score recorded. Set one from the calibration curve once "
                "there are enough scored outcomes to plot it."
            )

    def answer(
        self,
        question: str,
        strategy: str = "hybrid",
        top_k: int = 5,
    ) -> AnsweredQuestion:
        """Answer a question, escalating while confidence is below threshold.

        Args:
            question: The question to answer.
            strategy: Retrieval strategy for the first attempt.
            top_k: Chunks to retrieve on the first attempt.

        Returns:
            The final answer with every attempt recorded, including the ones
            that were only considered.
        """
        observing = self.threshold is None
        result, verification, confidence = self._attempt(question, strategy, top_k)

        record = AnsweredQuestion(
            question=question,
            generation=result,
            confidence=confidence,
            verification=verification,
            observed_only=observing,
        )

        if result.declined:
            record.attempts.append(
                self._log_attempt(Action.ALREADY_DECLINED, strategy, top_k,
                                  confidence, result, "the first pass declined")
            )
            return record

        if not result.ok:
            record.attempts.append(
                self._log_attempt(Action.DECLINED, strategy, top_k, confidence,
                                  result, f"generation failed: {result.error}")
            )
            return record

        if observing:
            # With no threshold there is nothing to compare the score against,
            # so the honest record is the score itself rather than a guess at
            # what would have happened. Nothing is re-run: escalating on a
            # threshold nobody has measured spends requests to satisfy a guess.
            record.attempts.append(
                self._log_attempt(
                    Action.ACCEPTED, strategy, top_k, confidence, result,
                    "accepted unconditionally; no threshold has been measured yet",
                )
            )
            return record

        if self._good_enough(confidence):
            record.attempts.append(
                self._log_attempt(Action.ACCEPTED, strategy, top_k, confidence,
                                  result, "confidence above threshold")
            )
            return record

        record.attempts.append(
            self._log_attempt(Action.ACCEPTED, strategy, top_k, confidence, result,
                              "below threshold; escalating")
        )

        for action, kwargs in (
            (Action.REQUERIED, {"strategy": self.requery_strategy, "top_k": top_k}),
            (Action.WIDENED, {"strategy": strategy, "top_k": self.widen_to}),
        ):
            retry, retry_verification, retry_confidence = self._attempt(
                question, **kwargs
            )
            record.attempts.append(
                self._log_attempt(action, kwargs["strategy"], kwargs["top_k"],
                                  retry_confidence, retry, "")
            )
            if retry.ok and not retry.declined and retry_confidence.score > confidence.score:
                # Kept only when it actually improved. A retry that scores lower
                # is a worse answer, and taking it because it came last would
                # make the system less reliable the harder it tried.
                result, verification, confidence = (
                    retry, retry_verification, retry_confidence
                )
                record.generation = retry
                record.verification = retry_verification
                record.confidence = retry_confidence
            if self._good_enough(confidence):
                return record

        # Nothing reached the bar. Declining is the honest end of the path.
        record.generation = self._as_decline(question, result, confidence)
        record.attempts.append(
            self._log_attempt(Action.DECLINED, strategy, top_k, confidence,
                              record.generation,
                              "no attempt reached the confidence threshold")
        )
        return record

    def _attempt(
        self, question: str, strategy: str, top_k: int
    ) -> tuple[GenerationResult, VerificationReport | None, ConfidenceBreakdown]:
        """Retrieve, generate, verify, and score one pass."""
        retrieval = self.retriever.retrieve(question, strategy=strategy, top_k=top_k)
        result = self.generator.generate(question, retrieval.results)

        verification = None
        if self.verifier and result.ok and isinstance(result.answer, GroundedAnswer):
            verification = self.verifier.verify(result.answer, result.retrieved)

        confidence = score_answer(
            question,
            result.answer or InsufficientEvidence(
                searched_for=question, missing=result.error or "generation failed"
            ),
            result.retrieved,
            verification,
        )
        return result, verification, confidence

    def _good_enough(self, confidence: ConfidenceBreakdown) -> bool:
        """Whether the score clears the bar. Always true when none is set."""
        if self.threshold is None:
            return True
        return confidence.score >= self.threshold

    def _as_decline(
        self,
        question: str,
        result: GenerationResult,
        confidence: ConfidenceBreakdown,
    ) -> GenerationResult:
        """Replace a low-confidence answer with an explicit decline."""
        declined = GenerationResult(
            ok=True,
            question=question,
            retrieved=result.retrieved,
            provider=result.provider,
            model=result.model,
            answer=InsufficientEvidence(
                searched_for=question,
                missing=(
                    f"an answer was produced but scored {confidence.score:.2f}, "
                    f"below the {self.threshold:.2f} threshold; "
                    + (confidence.notes[0] if confidence.notes else "evidence was weak")
                ),
            ),
        )
        return declined

    def _log_attempt(
        self,
        action: Action,
        strategy: str,
        top_k: int,
        confidence: ConfidenceBreakdown,
        result: GenerationResult,
        note: str,
    ) -> Attempt:
        logger.info(
            "%s (strategy=%s top_k=%s) confidence=%.2f %s",
            action.value, strategy, top_k, confidence.score, note,
        )
        return Attempt(
            action=action,
            strategy=strategy,
            top_k=top_k,
            confidence=confidence.score,
            claim_count=result.claim_count,
            declined=result.declined,
            note=note,
        )
