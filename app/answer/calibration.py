"""Turns a confidence score into a number that means something.

A confidence score is a claim about the future: *answers I score at 0.8 are
right about 80% of the time.* Nobody checks it, and unchecked it is decoration —
a number that sorts answers into an order without saying what the order is worth.

This bins scored answers by predicted confidence and reports the accuracy
actually measured in each bin. Two things come out of that:

* **Whether the score is calibrated at all.** If the 0.8 bin is right 40% of the
  time, the score is not a probability and must not be presented as one.
* **Where to put the fallback threshold.** Not guessed. Read off the curve,
  with the cost of the choice stated: how many answerable questions get declined
  to avoid how many wrong answers. That trade is the reader's decision, so the
  curve is more useful to them than the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Bin edges. Wider at the bottom because scores cluster high — most answers a
#: RAG system produces look plausible to it — and a uniform split would leave
#: the low bins empty and the top bin carrying everything.
DEFAULT_BINS: tuple[float, ...] = (0.0, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0001)


@dataclass
class ScoredOutcome:
    """One answer, its predicted confidence, and whether it was right."""

    confidence: float
    correct: bool
    question_id: str = ""
    declined: bool = False


@dataclass
class Bin:
    """One confidence band and what actually happened in it."""

    low: float
    high: float
    outcomes: list[ScoredOutcome] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.outcomes)

    @property
    def accuracy(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for o in self.outcomes if o.correct) / self.count

    @property
    def mean_confidence(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.confidence for o in self.outcomes) / self.count

    @property
    def gap(self) -> float:
        """Predicted minus measured. Positive means overconfident."""
        return self.mean_confidence - self.accuracy

    @property
    def is_measurable(self) -> bool:
        """Whether the bin holds enough answers to mean anything.

        Ten is a convention rather than a law, but a bin below it produces an
        accuracy that swings on a single answer, and printing it beside a
        well-populated bin invites a comparison the data cannot support.
        """
        return self.count >= 10

    def label(self) -> str:
        return f"{self.low:.1f}-{self.high:.1f}"


@dataclass
class CalibrationCurve:
    """Predicted confidence against measured accuracy."""

    bins: list[Bin]

    @property
    def total(self) -> int:
        return sum(b.count for b in self.bins)

    @property
    def expected_calibration_error(self) -> float:
        """Mean gap between predicted and measured, weighted by bin size.

        One number for whether the score can be read as a probability. Reported
        alongside the curve rather than instead of it: a low error can hide a
        score that is overconfident at the top and underconfident at the bottom.
        """
        if not self.total:
            return 0.0
        return sum(b.count * abs(b.gap) for b in self.bins) / self.total

    @property
    def is_calibrated(self) -> bool:
        """Whether the score may be presented as a probability."""
        return self.expected_calibration_error < 0.1

    def threshold_for(self, target_accuracy: float) -> tuple[float, str]:
        """The lowest confidence whose bin and everything above it hits a target.

        Args:
            target_accuracy: The accuracy the caller is willing to accept.

        Returns:
            The threshold, and a sentence stating what it costs. The cost is
            returned with it deliberately — a threshold quoted without the
            answers it discards is half the decision.
        """
        measurable = [b for b in self.bins if b.is_measurable]
        if not measurable:
            return 1.0, "no bin holds enough answers to set a threshold from"

        for i, current in enumerate(measurable):
            above = measurable[i:]
            kept = sum(b.count for b in above)
            correct = sum(b.accuracy * b.count for b in above)
            if kept and correct / kept >= target_accuracy:
                declined = self.total - kept
                wrong_avoided = sum(
                    b.count * (1 - b.accuracy) for b in measurable[:i]
                )
                return current.low, (
                    f"answering only above {current.low:.1f} reaches "
                    f"{correct / kept:.0%} accuracy; it declines {declined} of "
                    f"{self.total} answers to avoid roughly "
                    f"{wrong_avoided:.0f} wrong ones"
                )

        return 1.0, (
            f"no threshold reaches {target_accuracy:.0%}; the highest bin "
            f"measures {measurable[-1].accuracy:.0%}"
        )

    def report(self) -> str:
        lines = [
            f"{'confidence':<12}{'n':>6}{'predicted':>12}{'measured':>11}{'gap':>8}",
        ]
        for b in self.bins:
            if not b.count:
                continue
            caveat = "" if b.is_measurable else "   [n too small]"
            lines.append(
                f"{b.label():<12}{b.count:>6}{b.mean_confidence:>12.0%}"
                f"{b.accuracy:>11.0%}{b.gap:>+8.0%}{caveat}"
            )
        lines.append("")
        lines.append(
            f"expected calibration error {self.expected_calibration_error:.1%}"
        )
        lines.append(
            "The score can be read as a probability."
            if self.is_calibrated
            else "The score orders answers but is not a probability; present it as a rank."
        )
        return "\n".join(lines)


def build_curve(
    outcomes: list[ScoredOutcome], edges: tuple[float, ...] = DEFAULT_BINS
) -> CalibrationCurve:
    """Bin scored outcomes by predicted confidence.

    Declines are excluded. A correct decline is not a correct answer and
    counting it as one would let a system that declines everything report
    perfect calibration.
    """
    answered = [o for o in outcomes if not o.declined]
    bins = [Bin(low=edges[i], high=edges[i + 1]) for i in range(len(edges) - 1)]

    for outcome in answered:
        for b in bins:
            if b.low <= outcome.confidence < b.high:
                b.outcomes.append(outcome)
                break

    return CalibrationCurve(bins=bins)
