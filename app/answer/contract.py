"""The shape a grounded answer must take.

An answer is a list of claims, not a paragraph. That is the whole design: prose
can imply a source without having one, whereas a claim either carries a chunk id
and a quote or it does not exist. The schema makes an uncited claim
*unrepresentable* rather than merely discouraged — a model that wants to assert
something it cannot source has one legal move, which is to leave it out.

Three decisions worth stating, because each has a tempting alternative:

* **The quote is verbatim and required.** A chunk id alone cannot be checked
  without asking a second model whether the chunk supports the claim. A quote
  can be checked with string containment, for free, deterministically. Designing
  so the cheap check is *possible* is what makes citation verification
  affordable enough to run on every answer rather than on a sample.

* **"I don't know" is a first-class answer, not an error.** A system with no way
  to decline will say something else instead, and that something is the failure
  this project measures. ``InsufficientEvidence`` records what was searched for
  and what was missing, so a decline is diagnosable rather than a shrug.

* **Confidence is not in this module.** It is computed from evidence after the
  fact rather than asserted by the model, so it belongs to the layer that can
  see the retrieval scores and the verification results.
"""

from __future__ import annotations

import re
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def normalise_for_match(text: str) -> str:
    """Collapse whitespace so a quote can be compared against a chunk.

    Models reflow whitespace when quoting: a line break in the source becomes a
    space, runs of spaces collapse, a non-breaking space appears where a plain
    one was. None of that changes whether the quote is really in the chunk, and
    treating it as a mismatch would report honest citations as decorative.
    """
    return re.sub(r"\s+", " ", text.replace(" ", " ")).strip().lower()


class Claim(BaseModel):
    """One assertion, with the evidence it rests on.

    Both evidence fields are required. A claim is the unit that gets verified,
    so a claim without a source is not a weaker claim — it is a different kind
    of object, and this schema does not have one.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=1,
        description=(
            "One assertion, in one sentence. Say one thing: a sentence making "
            "two claims cannot be cited precisely, because its two halves may "
            "come from different places or only one may be supported."
        ),
    )
    chunk_ids: list[str] = Field(
        min_length=1,
        description=(
            "Ids of the retrieved chunks this claim is drawn from. Use the ids "
            "exactly as they were given to you. Never cite a chunk you were not "
            "shown."
        ),
    )
    quote: str = Field(
        min_length=1,
        description=(
            "The exact words from the cited chunk that support this claim, "
            "copied verbatim. Do not paraphrase, summarise, or repair the "
            "wording: this is checked character by character against the "
            "chunk, and a tidied quote reads as a fabricated one."
        ),
    )

    @field_validator("chunk_ids")
    @classmethod
    def _no_blank_or_duplicate_ids(cls, ids: list[str]) -> list[str]:
        cleaned = [i.strip() for i in ids if i and i.strip()]
        if not cleaned:
            raise ValueError("a claim must cite at least one chunk")
        # Order is preserved: the first citation is the primary source, and
        # dropping to a set would lose that.
        seen, unique = set(), []
        for chunk_id in cleaned:
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique.append(chunk_id)
        return unique

    @field_validator("text", "quote")
    @classmethod
    def _not_only_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    def quote_appears_in(self, chunk_text: str) -> bool:
        """Whether this claim's quote really occurs in the given chunk.

        The free half of citation verification. Whitespace is normalised
        because models reflow it when quoting; nothing else is forgiven.
        """
        return normalise_for_match(self.quote) in normalise_for_match(chunk_text)


class GroundedAnswer(BaseModel):
    """An answer made only of sourced claims."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["grounded"] = "grounded"
    claims: list[Claim] = Field(
        min_length=1,
        description=(
            "The claims that answer the question, in reading order. Every one "
            "must cite a chunk. If nothing can be cited, return an "
            "insufficient_evidence answer instead of guessing."
        ),
    )

    @property
    def cited_chunk_ids(self) -> list[str]:
        """Every chunk cited anywhere in the answer, in first-seen order."""
        seen, ordered = set(), []
        for claim in self.claims:
            for chunk_id in claim.chunk_ids:
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    ordered.append(chunk_id)
        return ordered

    def text(self) -> str:
        """The answer as prose, for display or for scoring against a label."""
        return " ".join(claim.text for claim in self.claims)


class InsufficientEvidence(BaseModel):
    """A refusal to answer, with enough detail to act on.

    Deliberately not an error and not an empty answer. "The corpus does not
    contain this" is a correct response to a question about something absent,
    and a system that cannot give it will invent something instead.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["insufficient_evidence"] = "insufficient_evidence"
    searched_for: str = Field(
        min_length=1,
        description="What was looked for, in your own words.",
    )
    missing: str = Field(
        min_length=1,
        description=(
            "What the retrieved material failed to establish. Be specific: "
            '"no chunk gives a date for the merger" is useful, "not enough '
            'information" is not.'
        ),
    )

    @property
    def cited_chunk_ids(self) -> list[str]:
        return []

    def text(self) -> str:
        return f"Insufficient evidence: {self.missing}"


#: What a generation must parse into. A discriminated union so the model
#: chooses between answering and declining explicitly, rather than declining by
#: returning something empty that later code has to interpret.
Answer = Union[GroundedAnswer, InsufficientEvidence]


class AnswerEnvelope(BaseModel):
    """Wrapper carrying the union, for providers that need a single root object.

    Some structured-output modes require the top level to be an object rather
    than a union, so the union is nested under one key with a discriminator.
    """

    model_config = ConfigDict(extra="forbid")

    answer: Answer = Field(discriminator="kind")

    @model_validator(mode="after")
    def _grounded_answers_have_claims(self) -> "AnswerEnvelope":
        if isinstance(self.answer, GroundedAnswer) and not self.answer.claims:
            raise ValueError(
                "a grounded answer with no claims is an insufficient_evidence "
                "answer; return that instead"
            )
        return self


def answer_json_schema() -> dict:
    """The JSON Schema sent to a provider that enforces one natively."""
    return AnswerEnvelope.model_json_schema()
