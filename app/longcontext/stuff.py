"""Answering without retrieval, by putting the whole corpus in the context.

The other half of the experiment. Million-token context windows made "just send
the documents" a real option for many corpora, and whether retrieval still earns
its cost is the open question this project measures.

The comparison is only worth publishing if both arms got a fair attempt, so this
arm is held to the same terms as the RAG one:

* **Same answer contract.** Long context must cite chunk ids and quote verbatim,
  exactly as the retrieval arm does, which means the stuffed corpus is chunked
  and labelled the same way. Letting one arm answer in free prose and scoring it
  against a contract the other had to satisfy would decide the result in advance.
* **Same scoring instrument.** Both are checked by the same verifier and the
  same labels.
* **Tokens counted, not estimated.** Cost is half the finding, and characters
  divided by four is not a measurement.
* **A corpus that does not fit is recorded as not fitting.** Silently truncating
  turns the interesting case — what happens past the window — into a different
  experiment with no label saying so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.answer.contract import (
    Answer,
    AnswerEnvelope,
    GroundedAnswer,
    InsufficientEvidence,
)
from app.answer.generate import PROMPT, GenerationResult, format_chunks
from app.answer.schema import gemini_answer_schema
from app.config import settings
from app.llm.provider import GenerationError, GeminiProvider, retry_with_backoff
from app.logging_config import get_logger
from app.retrieval.types import SearchResult
from app.storage.database import DatabaseManager
from app.storage.models import Chunk

logger = get_logger(__name__)

#: Head-room left below the model's window for the question, the schema, and the
#: answer itself. A corpus sized to the exact limit leaves nothing to reply with.
RESERVED_TOKENS = 8000


@dataclass
class CorpusFit:
    """Whether a corpus fits the context window, measured rather than guessed."""

    chunks: int
    tokens: int
    limit: int
    fits: bool

    @property
    def utilisation(self) -> float:
        return self.tokens / self.limit if self.limit else 0.0

    def describe(self) -> str:
        verdict = "fits" if self.fits else "DOES NOT FIT"
        return (
            f"{self.chunks} chunks, {self.tokens:,} tokens against a "
            f"{self.limit:,} token budget — {verdict} "
            f"({self.utilisation:.0%} of budget)"
        )


def load_corpus_chunks(document_ids: list[str] | None = None) -> list[SearchResult]:
    """Every chunk in the corpus, as the same type the retrieval arm produces.

    Returned as ``SearchResult`` so both arms hand identical objects to the
    prompt builder, the contract, and the verifier. The score is a constant:
    nothing was ranked, and inventing a relevance number here would let the
    confidence layer read meaning into it that does not exist.
    """
    with DatabaseManager() as db:
        query = db.query(Chunk)
        if document_ids:
            query = query.filter(Chunk.document_id.in_(document_ids))
        rows = query.order_by(Chunk.document_id, Chunk.chunk_index).all()

        return [
            SearchResult(
                chunk_id=row.id,
                text=row.text,
                score=0.0,
                document_id=row.document_id,
                filename=getattr(row.document, "filename", row.document_id),
                page_number=row.page_number,
                heading_context=row.heading_context,
            )
            for row in rows
        ]


class LongContextAnswerer:
    """Answers by stuffing the corpus into the context window."""

    def __init__(self, model: str | None = None, max_tokens: int = 3000) -> None:
        """
        Args:
            model: The long-context model. Defaults to the separately
                configured ``LONG_CONTEXT_MODEL`` rather than the general
                Gemini default, so moving one cannot silently change the
                comparison.
            max_tokens: Output ceiling for the answer.
        """
        self.provider = GeminiProvider(model or settings.LONG_CONTEXT_MODEL)
        self.max_tokens = max_tokens
        # Gemini models unions as anyOf and rejects oneOf outright, so the
        # contract is translated for it. Both arms therefore get native
        # schema enforcement, which keeps the comparison about context
        # length rather than about which provider enforced what.
        self.schema = gemini_answer_schema()
        self._window: int | None = None

    def context_limit(self) -> int:
        """The model's input window, read from the provider not assumed."""
        if self._window is None:
            self._window = self._lookup_window()
        return self._window

    def _lookup_window(self) -> int:
        try:
            from google import genai

            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            info = client.models.get(model=f"models/{self.provider.model}")
            limit = getattr(info, "input_token_limit", None)
            if limit:
                return int(limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read the context window: %s", exc)
        # A conservative floor rather than an optimistic guess: overstating the
        # window turns "did not fit" into a truncated request nobody labelled.
        return 128_000

    def measure_fit(self, chunks: list[SearchResult]) -> CorpusFit:
        """Whether the corpus fits, by counting tokens with the model itself."""
        limit = self.context_limit() - RESERVED_TOKENS
        rendered = format_chunks(chunks)
        try:
            tokens = self.provider.count_tokens(rendered)
        except GenerationError as exc:
            logger.warning("token count failed, falling back to an estimate: %s", exc)
            # Flagged in the log rather than silently substituted. An estimate
            # is worth having when the count is unavailable; pretending it is a
            # count is not.
            tokens = len(rendered) // 4
        return CorpusFit(
            chunks=len(chunks),
            tokens=tokens,
            limit=limit,
            fits=tokens <= limit,
        )

    def answer(
        self, question: str, chunks: list[SearchResult]
    ) -> tuple[GenerationResult, CorpusFit]:
        """Answer from the whole corpus, with no retrieval step.

        Args:
            question: The question asked.
            chunks: Every chunk in the corpus.

        Returns:
            The result and the fit measurement. A corpus that does not fit
            returns a failed result saying so rather than a truncated answer.
        """
        fit = self.measure_fit(chunks)
        base = GenerationResult(
            ok=False,
            question=question,
            retrieved=list(chunks),
            provider=self.provider.name,
            model=self.provider.model,
        )

        if not fit.fits:
            # The honest outcome. Truncating here would answer a different
            # question from a different corpus and report it as this one.
            base.error = (
                f"corpus does not fit the context window: {fit.describe()}"
            )
            logger.info("%s", base.error)
            return base, fit

        prompt = PROMPT.format(chunks=format_chunks(chunks), question=question)
        started = time.perf_counter()
        try:
            raw = retry_with_backoff(
                lambda: self._call(prompt),
                settings.GENERATION_RETRIES,
                f"gemini/{self.provider.model}",
            )
        except GenerationError as exc:
            base.error = str(exc)
            base.latency_ms = (time.perf_counter() - started) * 1000
            return base, fit
        base.latency_ms = (time.perf_counter() - started) * 1000
        base.raw_output = raw
        base.prompt_tokens = fit.tokens

        try:
            envelope = AnswerEnvelope.model_validate_json(raw)
        except ValidationError as exc:
            if any(e["type"] == "json_invalid" for e in exc.errors()):
                base.error = f"answer was not valid JSON: {exc.errors()[0]['msg']}"
            else:
                base.error = (
                    f"answer did not satisfy the contract: {exc.error_count()} errors"
                )
            logger.error("%s\n%s", base.error, raw[:400])
            return base, fit

        base.ok = True
        base.answer = self._ground(envelope.answer, chunks, base)
        return base, fit

    def _call(self, prompt: str) -> str:
        """Ask for the answer, with the schema attached."""
        from google.genai import types

        client = getattr(self.provider, "_client", None)
        if client is None:
            raise GenerationError("Gemini provider is not configured")

        try:
            response = client.models.generate_content(
                model=self.provider.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=settings.GENERATION_TEMPERATURE,
                    response_mime_type="application/json",
                    response_schema=self.schema,
                ),
            )
            text = response.text
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"gemini/{self.provider.model}: {exc}") from exc

        if not text:
            reason = "unknown"
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                reason = getattr(candidates[0], "finish_reason", "unknown")
            raise GenerationError(
                f"gemini/{self.provider.model}: empty completion "
                f"(finish_reason={reason})"
            )
        return text

    def _ground(
        self,
        answer: Answer,
        chunks: list[SearchResult],
        record: GenerationResult,
    ) -> Answer:
        """Drop claims citing chunks that are not in the corpus.

        The same rule the retrieval arm applies. It bites differently here —
        every chunk was supplied, so an unknown id is unambiguously invented
        rather than possibly misremembered from a chunk that was filtered out.
        """
        if isinstance(answer, InsufficientEvidence):
            return answer

        by_id = {c.chunk_id: c for c in chunks}
        kept = []
        for claim in answer.claims:
            known = [cid for cid in claim.chunk_ids if cid in by_id]
            if not known:
                record.dropped_claims.append({
                    "text": claim.text,
                    "cited": claim.chunk_ids,
                    "reason": "cites an id that is not in the corpus",
                })
                continue
            if len(known) != len(claim.chunk_ids):
                record.dropped_claims.append({
                    "text": claim.text,
                    "cited": [c for c in claim.chunk_ids if c not in by_id],
                    "reason": "some cited ids are not in the corpus",
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
                missing="no claim cited a chunk present in the corpus",
            )
        return GroundedAnswer(claims=kept)
