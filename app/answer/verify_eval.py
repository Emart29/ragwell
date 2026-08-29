"""Measures the entailment judge against hand labels.

The judge is a model judging a model. Its agreement rate has to be established
before any number it produces is quoted, and it is built here in the same module
as the judge rather than added afterwards — a validation set written after the
fact tends to be written around whatever the judge already does.

The cases below are hand-labelled and deliberately include the ones a judge gets
wrong: a claim that is true of the world but absent from the passage, a claim
that paraphrases correctly, a claim that inverts a number, and a claim the
passage contradicts outright. A set of obvious cases measures nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.answer.verify import CitationVerifier, Verdict
from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class JudgeCase:
    """One passage, one claim, and the verdict a careful reader would give."""

    id: str
    chunk: str
    claim: str
    label: Verdict
    #: Why this case is in the set. A case whose purpose cannot be stated is
    #: usually one that measures nothing.
    tests: str


POLICY = (
    "Customers may request a refund within 30 days of purchase. Refunds are "
    "issued to the original payment method and take 5 to 7 business days to "
    "appear. Digital goods are exempt once downloaded."
)
HISTORY = (
    "The merger completed on 14 March 2019. Regulatory approval had been "
    "granted the previous November, after a review lasting eight months."
)

CASES: list[JudgeCase] = [
    JudgeCase(
        id="verbatim",
        chunk=POLICY,
        claim="Refunds take 5 to 7 business days to appear.",
        label=Verdict.SUPPORTED,
        tests="the easy case; a judge failing this is broken",
    ),
    JudgeCase(
        id="paraphrase",
        chunk=POLICY,
        claim="A refund reaches the customer in under a fortnight.",
        label=Verdict.SUPPORTED,
        tests="correct rewording — a judge that only matches strings fails this",
    ),
    JudgeCase(
        id="inverted_number",
        chunk=POLICY,
        claim="Refunds take 5 to 7 weeks to appear.",
        label=Verdict.CONTRADICTED,
        tests="a plausible-looking claim the passage disagrees with",
    ),
    JudgeCase(
        id="inverted_policy",
        chunk=POLICY,
        claim="Digital goods can be refunded after they are downloaded.",
        label=Verdict.CONTRADICTED,
        tests="an inversion of a specific exemption",
    ),
    JudgeCase(
        id="absent_but_plausible",
        chunk=POLICY,
        claim="Refunds are processed by the finance team.",
        label=Verdict.UNRELATED,
        tests="true of many companies, absent here — the failure that matters",
    ),
    JudgeCase(
        id="wrong_passage",
        chunk=HISTORY,
        claim="Refunds take 5 to 7 business days.",
        label=Verdict.UNRELATED,
        tests="a real claim citing the wrong chunk",
    ),
    JudgeCase(
        id="implied",
        chunk=HISTORY,
        claim="Regulatory approval came before the merger completed.",
        label=Verdict.SUPPORTED,
        tests="directly implied rather than stated; needs reading, not matching",
    ),
    JudgeCase(
        id="overreach",
        chunk=HISTORY,
        claim="The merger was approved after an eight month review by three regulators.",
        label=Verdict.UNRELATED,
        tests="mostly supported with one invented detail — the subtle case",
    ),
]


@dataclass
class Agreement:
    """How well the judge matched the hand labels."""

    total: int
    agreed: int
    #: Judged supported when the label says otherwise. The dangerous direction:
    #: a decorative citation waved through.
    false_supports: list[str]
    #: Judged unsupported when the label says supported. Costly but safe.
    false_rejects: list[str]
    unchecked: list[str]

    @property
    def rate(self) -> float:
        return self.agreed / self.total if self.total else 0.0

    @property
    def trustworthy(self) -> bool:
        """Whether the judge's numbers are worth quoting.

        No false supports is the binding condition. A judge that waves through
        a bad citation is worse than one that is merely strict, because the
        whole point is catching citations that do not hold.
        """
        return self.rate >= 0.75 and not self.false_supports

    def report(self) -> str:
        lines = [
            f"agreement      {self.rate:.0%} over {self.total} labelled cases",
            f"false supports {len(self.false_supports)}"
            + (f"  {', '.join(self.false_supports)}" if self.false_supports else ""),
            f"false rejects  {len(self.false_rejects)}"
            + (f"  {', '.join(self.false_rejects)}" if self.false_rejects else ""),
        ]
        if self.unchecked:
            lines.append(f"unchecked      {', '.join(self.unchecked)}")
        lines.append(
            "The judge is worth quoting."
            if self.trustworthy
            else "Report the judge as unreliable rather than quoting its rates."
        )
        return "\n".join(lines)


def measure_judge(
    provider: str = "groq", model: str | None = None, cases: list[JudgeCase] | None = None
) -> Agreement:
    """Score the entailment judge against the hand labels.

    Args:
        provider: Provider hosting the judge.
        model: Judge model, defaulting to that provider's configured one.
        cases: Labelled cases, defaulting to the full set.

    Returns:
        The agreement, split by direction of error. The two directions are not
        equivalent and are never summed into one accuracy figure.
    """
    cases = cases or CASES
    verifier = CitationVerifier(judge_provider=provider, judge_model=model)

    agreed = 0
    false_supports: list[str] = []
    false_rejects: list[str] = []
    unchecked: list[str] = []

    for case in cases:
        verdict, _ = verifier._entails(
            case.claim, [_as_chunk(case.chunk)]
        )
        if verdict is Verdict.UNCHECKED:
            unchecked.append(case.id)
            continue
        if verdict is case.label:
            agreed += 1
        elif verdict is Verdict.SUPPORTED:
            false_supports.append(case.id)
        elif case.label is Verdict.SUPPORTED:
            false_rejects.append(case.id)
        else:
            # Both say unsupported, disagreeing on which kind. Counted as
            # agreement on the question that matters — whether the citation
            # holds — since both verdicts reject it.
            agreed += 1

    return Agreement(
        total=len(cases),
        agreed=agreed,
        false_supports=false_supports,
        false_rejects=false_rejects,
        unchecked=unchecked,
    )


def _as_chunk(text: str):
    """Wrap raw text as a SearchResult, which is what the verifier expects."""
    from app.retrieval.types import SearchResult

    return SearchResult(
        chunk_id="case",
        text=text,
        score=1.0,
        document_id="case",
        filename="case",
    )
