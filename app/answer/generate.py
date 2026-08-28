"""Turns retrieved chunks into a validated, grounded answer.

Three rules live in code rather than in the prompt, because a prompt is a
request and this layer needs guarantees:

* **A claim citing a chunk that was not retrieved is dropped.** The model can
  only have got that id by inventing it or misreading, and either way the
  citation points at nothing the caller can check. Dropped claims are counted,
  never silently discarded — the rate at which a model cites chunks it was not
  shown is a measurement worth having.
* **A claim whose quote is not in its cited chunk is flagged, not removed.**
  Whether that is fabrication or a paraphrase is the question the verification
  layer answers properly; this layer records it and moves on.
* **An answer with no surviving claims becomes a decline.** Returning an empty
  answer would push the problem downstream, where nobody can attribute it.

There is deliberately no repair loop here. How often the raw generation is
well-formed is a number worth measuring before it is papered over, and a repair
loop built in advance of that measurement is a fix for an unmeasured problem.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from app.answer.contract import (
    Answer,
    AnswerEnvelope,
    Claim,
    GroundedAnswer,
    InsufficientEvidence,
)
from app.answer.schema import strict_answer_schema
from app.config import settings
from app.llm.provider import GenerationError, get_named_provider
from app.logging_config import get_logger
from app.retrieval.types import SearchResult

logger = get_logger(__name__)

#: Enough room for several claims plus a reasoning model's preamble. A budget
#: sized for the answer alone returns nothing at all from a model that thinks
#: before it speaks, which reads as a refusal rather than a truncation.
DEFAULT_MAX_TOKENS = 3000

PROMPT = """You are answering a question using only the numbered source chunks below.

{chunks}

Question: {question}

Rules:
- Every claim must cite the id of the chunk it comes from, exactly as written above.
- Every claim must include a quote copied word for word from that chunk. Do not \
paraphrase, tidy, or shorten the quote.
- Make one assertion per claim. Split a sentence that says two things.
- Use only what the chunks say. Do not add background knowledge.
- If the chunks do not answer the question, return an insufficient_evidence \
answer saying what is missing. That is a correct response, not a failure."""


@dataclass
class GenerationResult:
    """An answer and everything needed to judge how it was produced."""

    ok: bool
    answer: Answer | None = None
    question: str = ""
    #: Chunks the retriever supplied, in the order the model saw them.
    retrieved: list[SearchResult] = field(default_factory=list)
    #: Claims removed because they cited a chunk that was never retrieved.
    dropped_claims: list[dict[str, Any]] = field(default_factory=list)
    #: Claims kept, but whose quote was not found in the chunk they cite.
    unverified_claims: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    raw_output: str = ""
    provider: str = ""
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0

    @property
    def declined(self) -> bool:
        return isinstance(self.answer, InsufficientEvidence)

    @property
    def claim_count(self) -> int:
        return len(self.answer.claims) if isinstance(self.answer, GroundedAnswer) else 0

    @property
    def quote_verified_rate(self) -> float:
        """Share of surviving claims whose quote was found in its chunk.

        The free half of citation verification, computed here because the
        chunks are in hand. The full ladder lives in the verification layer.
        """
        total = self.claim_count
        if not total:
            return 0.0
        return (total - len(self.unverified_claims)) / total

    def summary(self) -> str:
        if not self.ok:
            return f"FAILED: {self.error}"
        if self.declined:
            return f"declined: {self.answer.missing}"
        parts = [f"{self.claim_count} claims"]
        if self.dropped_claims:
            parts.append(f"{len(self.dropped_claims)} dropped (uncited chunk)")
        if self.unverified_claims:
            parts.append(f"{len(self.unverified_claims)} quotes unverified")
        return ", ".join(parts)


def format_chunks(results: list[SearchResult]) -> str:
    """Render chunks for the prompt with their ids attached.

    The id goes inline with the text so citing becomes a copying task rather
    than a recall task. A model asked to remember an identifier it saw in a
    header will eventually write a plausible-looking one instead.
    """
    blocks = []
    for result in results:
        location = result.filename
        if result.page_number is not None:
            location += f", p.{result.page_number}"
        if result.heading_context:
            location += f" — {result.heading_context}"
        blocks.append(f"[{result.chunk_id}] ({location})\n{result.text}")
    return "\n\n".join(blocks)


class AnswerGenerator:
    """Generates grounded answers from retrieved chunks."""

    def __init__(
        self,
        provider_name: str | None = None,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        """
        Args:
            provider_name: Which provider to use. Named explicitly and never
                substituted, so an answer is always attributable.
            model: Model id, defaulting to that provider's configured one.
            max_tokens: Output ceiling.
        """
        self.provider_name = provider_name or settings.DEFAULT_LLM_PROVIDER
        self.provider = get_named_provider(self.provider_name, model)
        self.max_tokens = max_tokens
        self.schema, self.dropped_constraints = strict_answer_schema()

    def generate(
        self, question: str, results: list[SearchResult]
    ) -> GenerationResult:
        """Answer a question from retrieved chunks.

        Args:
            question: The question asked.
            results: Chunks the retriever returned.

        Returns:
            The result, successful or explained. Never ``None``, and never an
            answer whose provenance cannot be reconstructed.
        """
        base = GenerationResult(
            ok=False,
            question=question,
            retrieved=list(results),
            provider=self.provider.name,
            model=self.provider.model,
        )

        if not results:
            # Nothing retrieved is a decline, not an error: the corpus may
            # genuinely not contain the answer, and that is a valid finding.
            base.ok = True
            base.answer = InsufficientEvidence(
                searched_for=question,
                missing="retrieval returned no chunks for this question",
            )
            return base

        prompt = PROMPT.format(
            chunks=format_chunks(results), question=question
        )

        started = time.perf_counter()
        try:
            raw = self._call(prompt)
        except GenerationError as exc:
            base.error = str(exc)
            base.latency_ms = (time.perf_counter() - started) * 1000
            logger.error("generation failed: %s", exc)
            return base
        base.latency_ms = (time.perf_counter() - started) * 1000
        base.raw_output = raw

        try:
            envelope = AnswerEnvelope.model_validate_json(raw)
        except ValidationError as exc:
            # Two different failures arrive as the same exception type, with
            # different causes. Output that is not JSON at all means the
            # enforcement directive did nothing. Output that is JSON but breaks
            # the contract means enforcement applied and the model could not
            # satisfy a rule the schema could not carry - such as the minItems
            # strict mode strips. Collapsing them would hide which happened.
            if any(err["type"] == "json_invalid" for err in exc.errors()):
                base.error = f"answer was not valid JSON: {exc.errors()[0]['msg']}"
            else:
                base.error = (
                    "answer did not satisfy the contract: "
                    f"{exc.error_count()} errors"
                )
            logger.error("%s\n%s", base.error, raw[:400])
            return base

        base.ok = True
        base.answer = self._ground(envelope.answer, results, base)
        return base

    def _call(self, prompt: str) -> str:
        """Send the prompt with the schema attached where the provider takes one."""
        provider = self.provider
        client = getattr(provider, "_client", None)

        if provider.name == "groq" and client is not None:
            try:
                response = client.chat.completions.create(
                    model=provider.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "grounded_answer",
                            "schema": self.schema,
                            "strict": True,
                        },
                    },
                    max_tokens=self.max_tokens,
                    temperature=settings.GENERATION_TEMPERATURE,
                    timeout=settings.GENERATION_TIMEOUT,
                )
            except Exception as exc:  # noqa: BLE001
                raise GenerationError(f"groq/{provider.model}: {exc}") from exc
            text = response.choices[0].message.content
            if not text:
                reason = getattr(response.choices[0], "finish_reason", "unknown")
                raise GenerationError(
                    f"groq/{provider.model}: empty completion "
                    f"(finish_reason={reason})"
                )
            return text

        # Providers without native schema enforcement are asked in the prompt
        # and validated on the way back, which is the weaker guarantee this
        # project exists to measure rather than assume.
        instructed = (
            f"{prompt}\n\nReturn only JSON matching this schema, with no "
            f"other text:\n{json.dumps(self.schema)}"
        )
        return provider.generate(instructed, max_tokens=self.max_tokens)

    def _ground(
        self,
        answer: Answer,
        results: list[SearchResult],
        record: GenerationResult,
    ) -> Answer:
        """Drop claims citing chunks that were never retrieved; flag bad quotes."""
        if isinstance(answer, InsufficientEvidence):
            return answer

        by_id = {r.chunk_id: r for r in results}
        kept: list[Claim] = []

        for claim in answer.claims:
            known = [cid for cid in claim.chunk_ids if cid in by_id]
            if not known:
                record.dropped_claims.append({
                    "text": claim.text,
                    "cited": claim.chunk_ids,
                    "reason": "cites no retrieved chunk",
                })
                continue

            if len(known) != len(claim.chunk_ids):
                # Partly grounded: keep it, narrowed to the real citations, and
                # record that the model produced an id it was never given.
                record.dropped_claims.append({
                    "text": claim.text,
                    "cited": [c for c in claim.chunk_ids if c not in by_id],
                    "reason": "some cited chunks were not retrieved",
                })
                claim = claim.model_copy(update={"chunk_ids": known})

            if not any(claim.quote_appears_in(by_id[cid].text) for cid in known):
                record.unverified_claims.append({
                    "text": claim.text,
                    "cited": claim.chunk_ids,
                    "quote": claim.quote,
                })
            kept.append(claim)

        if not kept:
            return InsufficientEvidence(
                searched_for=record.question,
                missing=(
                    "every claim cited a chunk that was not retrieved, so none "
                    "could be grounded"
                ),
            )
        return GroundedAnswer(claims=kept)
