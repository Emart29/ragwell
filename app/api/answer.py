"""Grounded answer API.

Extends ragwell's retrieval endpoints rather than replacing them: `/query`
returns chunks and this returns an answer built from chunks, with every claim
carrying the id and the quote that support it.

Answers are stored so a citation can be checked later. Verification is a
separate endpoint on purpose — checking a citation is cheap enough to run on
every answer and expensive enough to be worth not running twice, and a caller
who has already seen the checks should not pay for them again.
"""

from __future__ import annotations

import uuid
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.answer.confidence import score_answer as score_confidence
from app.answer.contract import GroundedAnswer, InsufficientEvidence
from app.answer.generate import AnswerGenerator
from app.answer.verify import CitationVerifier
from app.config import settings
from app.logging_config import get_logger
from app.retrieval.retriever import Retriever

logger = get_logger(__name__)
router = APIRouter()

retriever = Retriever()
_generators: dict[str, AnswerGenerator] = {}
verifier = CitationVerifier(judge_provider=None)

#: Answers kept for later verification. In memory on purpose: this is a
#: benchmark tool, and a store that outlives the process would need a schema,
#: a migration, and a retention policy for text that came from a customer's
#: documents.
_answers: dict[str, "AnswerRecord"] = {}
MAX_STORED_ANSWERS = 500


def _generator(provider: str) -> AnswerGenerator:
    """One generator per provider, reused across requests."""
    if provider not in _generators:
        _generators[provider] = AnswerGenerator(provider_name=provider)
    return _generators[provider]


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    strategy: str = Field("hybrid", description="vector, keyword, hybrid, hyde, expanded")
    top_k: int = Field(8, ge=1, le=50)
    provider: Optional[str] = Field(
        None, description="Generation provider. Defaults to the configured one."
    )
    verify: bool = Field(True, description="Check citations before returning")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "question": "What is the maximum deposit insurance coverage?",
                "strategy": "hybrid",
                "top_k": 8,
                "verify": True,
            }]
        }
    }


class CitedChunk(BaseModel):
    """A chunk a claim rests on, with enough to find it in the source."""

    chunk_id: str
    text: str
    filename: str
    page: Optional[int] = None
    heading: Optional[str] = None


class ClaimOut(BaseModel):
    text: str
    chunk_ids: list[str]
    quote: str
    #: Whether the quote was found in a cited chunk. None when no verification
    #: ran — distinct from False, which means it was checked and was not there.
    quote_verified: Optional[bool] = None
    entailment: Optional[str] = None


class AnswerResponse(BaseModel):
    answer_id: str
    kind: Literal["grounded", "insufficient_evidence"]
    question: str
    #: Empty for a decline. A decline is a valid answer, not an error.
    claims: list[ClaimOut] = []
    missing: Optional[str] = None
    confidence: float
    confidence_notes: list[str] = []
    chunks: list[CitedChunk] = []
    provider: str
    model: str
    prompt_tokens: Optional[int] = None
    latency_ms: float


class VerificationOut(BaseModel):
    answer_id: str
    total_claims: int
    quote_verified_rate: float
    cheaply_verified_rate: float
    decorative_rate: float
    caught_for_free: int
    judge_ran: bool
    claims: list[ClaimOut]


class AnswerRecord(BaseModel):
    """What is kept so an answer can be re-verified without re-answering."""

    response: AnswerResponse
    chunk_text: dict[str, str]


@router.post("/answer", response_model=AnswerResponse)
async def answer_question(request: AnswerRequest) -> AnswerResponse:
    """Retrieve, answer, and cite."""
    provider = request.provider or settings.DEFAULT_LLM_PROVIDER
    try:
        generator = _generator(provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    retrieval = retriever.retrieve(
        request.question, strategy=request.strategy, top_k=request.top_k
    )
    result = generator.generate(request.question, retrieval.results)

    if not result.ok:
        # A generation failure is a 502: the request was fine and the upstream
        # model was not. Returning 200 with an empty answer would make a
        # provider outage indistinguishable from a corpus that says nothing.
        raise HTTPException(status_code=502, detail=result.error)

    verification = None
    if request.verify and isinstance(result.answer, GroundedAnswer):
        verification = verifier.verify(
            result.answer, result.retrieved, run_judge=False
        )

    confidence = score_confidence(
        request.question, result.answer, result.retrieved, verification
    )

    checks = {c.claim_text: c for c in (verification.checks if verification else [])}
    claims = []
    if isinstance(result.answer, GroundedAnswer):
        for claim in result.answer.claims:
            check = checks.get(claim.text)
            claims.append(ClaimOut(
                text=claim.text,
                chunk_ids=claim.chunk_ids,
                quote=claim.quote,
                quote_verified=check.quote_found if check else None,
                entailment=(
                    check.entailment.value
                    if check and check.entailment.value != "unchecked"
                    else None
                ),
            ))

    response = AnswerResponse(
        answer_id=uuid.uuid4().hex[:16],
        kind="grounded" if isinstance(result.answer, GroundedAnswer) else "insufficient_evidence",
        question=request.question,
        claims=claims,
        missing=(
            result.answer.missing
            if isinstance(result.answer, InsufficientEvidence)
            else None
        ),
        confidence=confidence.score,
        confidence_notes=confidence.notes,
        chunks=[
            CitedChunk(
                chunk_id=r.chunk_id,
                text=r.text,
                filename=r.filename,
                page=r.page_number,
                heading=r.heading_context,
            )
            for r in result.retrieved
        ],
        provider=result.provider,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        latency_ms=round(result.latency_ms, 1),
    )

    _remember(response, {r.chunk_id: r.text for r in result.retrieved})
    return response


@router.post("/answer/{answer_id}/verify", response_model=VerificationOut)
async def verify_answer(answer_id: str) -> VerificationOut:
    """Re-check a stored answer's citations, running the entailment judge."""
    record = _answers.get(answer_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no answer {answer_id}")
    if record.response.kind != "grounded":
        raise HTTPException(
            status_code=400,
            detail="a declined answer has no citations to verify",
        )

    from app.retrieval.types import SearchResult

    answer = GroundedAnswer(claims=[
        {"text": c.text, "chunk_ids": c.chunk_ids, "quote": c.quote}
        for c in record.response.claims
    ])
    chunks = [
        SearchResult(
            chunk_id=cid, text=text, score=0.0, document_id="", filename=""
        )
        for cid, text in record.chunk_text.items()
    ]
    report = verifier.verify(answer, chunks, run_judge=True)

    return VerificationOut(
        answer_id=answer_id,
        total_claims=report.total,
        quote_verified_rate=report.quote_verified_rate,
        cheaply_verified_rate=report.cheaply_verified_rate,
        decorative_rate=report.decorative_rate,
        caught_for_free=report.caught_for_free,
        judge_ran=report.judge_ran,
        claims=[
            ClaimOut(
                text=check.claim_text,
                chunk_ids=check.chunk_ids,
                quote=check.quote,
                quote_verified=check.quote_found,
                entailment=(
                    check.entailment.value
                    if check.entailment.value != "unchecked"
                    else None
                ),
            )
            for check in report.checks
        ],
    )


@router.get("/answer/{answer_id}", response_model=AnswerResponse)
async def get_answer(answer_id: str) -> AnswerResponse:
    """Retrieve a stored answer with the chunks it was built from."""
    record = _answers.get(answer_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no answer {answer_id}")
    return record.response


def _remember(response: AnswerResponse, chunk_text: dict[str, str]) -> None:
    """Keep an answer, discarding the oldest once the cap is reached."""
    _answers[response.answer_id] = AnswerRecord(
        response=response, chunk_text=chunk_text
    )
    while len(_answers) > MAX_STORED_ANSWERS:
        _answers.pop(next(iter(_answers)))
