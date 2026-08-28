"""Does the cited chunk actually support the claim?

Every RAG framework emits citations. Almost none check them. This module is the
check, built as a ladder of three tests in increasing cost:

1. **Quote containment** — free, deterministic. Is the claim's quote really in
   the chunk it cites? A citation failing this is decorative, and it costs
   nothing to know.
2. **Lexical overlap** — cheap, deterministic. Do the claim's content words
   appear in the chunk? Catches the case where the quote is genuine but the
   claim generalises well past it.
3. **Entailment** — a model call. Does the chunk support the claim, contradict
   it, or neither?

The ladder exists because the first two rungs are free and objective. If most
citation failures are caught by a string comparison, that is a finding in its
own right: the industry's citation problem would be checkable without an LLM at
all, and the expensive judge is only needed for the residue.

The entailment judge is a model judging a model. Its agreement with hand labels
is measured in ``verify_eval.py`` and must be published beside any number it
produces, because an unvalidated judge is just a second unvalidated model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from app.answer.contract import Claim, GroundedAnswer, normalise_for_match
from app.llm.provider import GenerationError, get_named_provider
from app.logging_config import get_logger
from app.retrieval.types import SearchResult

logger = get_logger(__name__)

#: Words carrying no evidential weight. A claim and a chunk sharing only these
#: share nothing, so they are excluded before overlap is measured.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from
by with without about into over under is are was were be been being has have
had do does did not no nor so as it its their his her they he she we you i
which who whom whose when where while there here also can could may might must
shall should will would very more most much many few some any all each other
""".split())

#: Share of a claim's content words that must appear in the chunk. Set low on
#: purpose: this rung is a smoke alarm for claims that have wandered far from
#: their source, not a paraphrase detector. Anything stricter would fail honest
#: rewording, which is the model doing its job.
OVERLAP_THRESHOLD = 0.4


class Verdict(str, Enum):
    """What the evidence says about a claim."""

    SUPPORTED = "supported"
    #: The chunk says the opposite. Worse than unsupported: the model had the
    #: evidence in front of it and contradicted it.
    CONTRADICTED = "contradicted"
    #: The chunk neither supports nor contradicts. The citation is decorative.
    UNRELATED = "unrelated"
    #: The judge could not be reached. Not a verdict, and never counted as one.
    UNCHECKED = "unchecked"


@dataclass
class ClaimCheck:
    """Every rung's result for one claim."""

    claim_text: str
    chunk_ids: list[str]
    quote: str
    #: Rung 1. None when no cited chunk was available to check against.
    quote_found: bool | None = None
    #: Rung 2, as a share of the claim's content words present in the chunk.
    lexical_overlap: float = 0.0
    #: Rung 3. Absent unless the entailment judge ran.
    entailment: Verdict = Verdict.UNCHECKED
    entailment_reason: str = ""

    @property
    def cheaply_verified(self) -> bool:
        """Whether the free rungs alone establish the citation.

        The number the article turns on: how much of citation checking needs no
        model at all.
        """
        return bool(self.quote_found) and self.lexical_overlap >= OVERLAP_THRESHOLD

    @property
    def decorative(self) -> bool:
        """Whether the citation points at something that does not support it."""
        if self.entailment in (Verdict.CONTRADICTED, Verdict.UNRELATED):
            return True
        if self.entailment is Verdict.SUPPORTED:
            return False
        # Without a judge, a missing quote is the strongest evidence available.
        return self.quote_found is False


@dataclass
class VerificationReport:
    """How well an answer's citations hold up."""

    checks: list[ClaimCheck] = field(default_factory=list)
    judge_ran: bool = False
    judge_model: str = ""

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def quote_verified_rate(self) -> float:
        return self._rate(lambda c: bool(c.quote_found))

    @property
    def cheaply_verified_rate(self) -> float:
        return self._rate(lambda c: c.cheaply_verified)

    @property
    def decorative_rate(self) -> float:
        """Share of claims whose citation does not support them."""
        return self._rate(lambda c: c.decorative)

    @property
    def entailed_rate(self) -> float:
        """Share judged supported, among those the judge actually saw."""
        judged = [c for c in self.checks if c.entailment is not Verdict.UNCHECKED]
        if not judged:
            return 0.0
        return sum(1 for c in judged if c.entailment is Verdict.SUPPORTED) / len(judged)

    @property
    def caught_for_free(self) -> int:
        """Failures the string checks found without asking a model.

        The argument for the ladder: if this is most of them, citation
        verification is affordable on every answer rather than on a sample.
        """
        return sum(1 for c in self.checks if c.quote_found is False)

    def _rate(self, predicate) -> float:
        if not self.checks:
            return 0.0
        return sum(1 for c in self.checks if predicate(c)) / len(self.checks)

    def summary(self) -> str:
        if not self.checks:
            return "no claims to verify"
        parts = [
            f"{self.total} claims",
            f"quote found {self.quote_verified_rate:.0%}",
            f"decorative {self.decorative_rate:.0%}",
        ]
        if self.judge_ran:
            parts.append(f"entailed {self.entailed_rate:.0%}")
        return ", ".join(parts)


def content_words(text: str) -> set[str]:
    """The words in a text that carry evidential weight."""
    words = re.findall(r"[a-z0-9]+", normalise_for_match(text))
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def lexical_overlap(claim_text: str, chunk_text: str) -> float:
    """Share of the claim's content words that appear in the chunk.

    Deliberately asymmetric. A chunk is usually far longer than a claim, so
    measuring the other direction, or a symmetric similarity, would score a
    short true claim against a long chunk as barely related.
    """
    claim_words = content_words(claim_text)
    if not claim_words:
        return 0.0
    return len(claim_words & content_words(chunk_text)) / len(claim_words)


ENTAILMENT_PROMPT = """Does the source passage support the claim?

Source passage:
{chunk}

Claim:
{claim}

Answer with one word on the first line — supported, contradicted, or unrelated:
- supported: the passage states or directly implies the claim
- contradicted: the passage states something incompatible with the claim
- unrelated: the passage neither supports nor contradicts it

Then one short line giving your reason.

Judge only against the passage above. Do not use outside knowledge, and do not \
reward a claim for being plausible."""


class CitationVerifier:
    """Runs the verification ladder over an answer's claims."""

    def __init__(
        self,
        judge_provider: str | None = None,
        judge_model: str | None = None,
    ) -> None:
        """
        Args:
            judge_provider: Provider for the entailment rung. ``None`` disables
                it, and the report then says the judge did not run rather than
                reporting an unchecked claim as passing.
            judge_model: Model for the judge.
        """
        self._judge = (
            get_named_provider(judge_provider, judge_model) if judge_provider else None
        )

    def verify(
        self,
        answer: GroundedAnswer,
        chunks: Iterable[SearchResult],
        run_judge: bool = True,
    ) -> VerificationReport:
        """Check every claim in an answer against the chunks it cites.

        Args:
            answer: The answer to check.
            chunks: The chunks that were retrieved, in any order.
            run_judge: Whether to run the entailment rung.

        Returns:
            One check per claim, with each rung reported separately so the
            cheap ones can be counted on their own.
        """
        by_id = {c.chunk_id: c for c in chunks}
        report = VerificationReport(
            judge_ran=bool(self._judge) and run_judge,
            judge_model=self._judge.model if self._judge else "",
        )

        for claim in answer.claims:
            report.checks.append(
                self._check_claim(claim, by_id, run_judge=run_judge)
            )
        return report

    def _check_claim(
        self,
        claim: Claim,
        by_id: dict[str, SearchResult],
        run_judge: bool,
    ) -> ClaimCheck:
        check = ClaimCheck(
            claim_text=claim.text,
            chunk_ids=list(claim.chunk_ids),
            quote=claim.quote,
        )

        cited = [by_id[cid] for cid in claim.chunk_ids if cid in by_id]
        if not cited:
            # Nothing to check against. Left as None rather than False: a
            # missing chunk is not evidence that the quote was invented.
            return check

        check.quote_found = any(claim.quote_appears_in(c.text) for c in cited)
        check.lexical_overlap = max(
            lexical_overlap(claim.text, c.text) for c in cited
        )

        if self._judge and run_judge:
            check.entailment, check.entailment_reason = self._entails(
                claim.text, cited
            )
        return check

    def _entails(
        self, claim_text: str, cited: list[SearchResult]
    ) -> tuple[Verdict, str]:
        """Ask the judge whether any cited chunk supports the claim."""
        best: tuple[Verdict, str] = (Verdict.UNCHECKED, "")
        for chunk in cited:
            prompt = ENTAILMENT_PROMPT.format(chunk=chunk.text, claim=claim_text)
            try:
                raw = self._judge.generate(prompt, max_tokens=2000)
            except GenerationError as exc:
                logger.error("entailment judge failed: %s", exc)
                return Verdict.UNCHECKED, str(exc)

            verdict, reason = parse_entailment(raw)
            if verdict is Verdict.SUPPORTED:
                # One supporting chunk is enough; the claim cited it for a
                # reason and the others may cover different ground.
                return verdict, reason
            if best[0] is Verdict.UNCHECKED:
                best = (verdict, reason)
        return best


def parse_entailment(raw: str) -> tuple[Verdict, str]:
    """Read a verdict out of the judge's reply.

    Tolerant of the model wrapping its answer in prose, and explicit about
    failing to find one: an unreadable reply becomes ``UNCHECKED`` rather than
    defaulting to a verdict, because a judge that silently reads as "supported"
    when it malfunctions is worse than no judge.
    """
    if not raw or not raw.strip():
        return Verdict.UNCHECKED, "empty reply"

    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    haystack = normalise_for_match(lines[0]) if lines else ""
    reason = lines[1] if len(lines) > 1 else ""

    for verdict in (Verdict.CONTRADICTED, Verdict.UNRELATED, Verdict.SUPPORTED):
        if verdict.value in haystack:
            return verdict, reason

    # Not on the first line; scan the whole reply before giving up.
    whole = normalise_for_match(raw)
    for verdict in (Verdict.CONTRADICTED, Verdict.UNRELATED, Verdict.SUPPORTED):
        if verdict.value in whole:
            return verdict, reason or lines[0][:120]

    return Verdict.UNCHECKED, f"no verdict found in: {raw.strip()[:80]}"
