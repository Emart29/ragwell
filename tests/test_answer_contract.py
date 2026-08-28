"""Tests for the answer contract.

The property under test throughout: an uncited claim must be impossible to
construct. Everything else in this project — citation verification, the
hallucination measurement, the comparison against long context — assumes that a
claim which exists has a source attached, and the schema is the only thing
enforcing it.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.answer.contract import (
    AnswerEnvelope,
    Claim,
    GroundedAnswer,
    InsufficientEvidence,
    answer_json_schema,
    normalise_for_match,
)
from app.answer.schema import for_strict_mode, strict_answer_schema

CHUNK = (
    "The merger completed on 14 March 2019. Regulatory approval had been "
    "granted the previous November, after a review lasting eight months."
)


def claim(**overrides) -> Claim:
    base = dict(
        text="The merger completed on 14 March 2019.",
        chunk_ids=["chunk-1"],
        quote="The merger completed on 14 March 2019.",
    )
    return Claim(**{**base, **overrides})


class TestAnUncitedClaimCannotExist:
    """The load-bearing property of the whole project."""

    def test_a_claim_with_no_chunk_ids_is_rejected(self):
        with pytest.raises(ValidationError):
            claim(chunk_ids=[])

    def test_a_claim_with_blank_chunk_ids_is_rejected(self):
        """A list of empty strings is not a citation."""
        with pytest.raises(ValidationError):
            claim(chunk_ids=["", "   "])

    def test_a_claim_with_no_quote_is_rejected(self):
        with pytest.raises(ValidationError):
            claim(quote="")

    def test_a_claim_with_a_whitespace_quote_is_rejected(self):
        with pytest.raises(ValidationError):
            claim(quote="   \n  ")

    def test_chunk_ids_cannot_be_omitted_entirely(self):
        with pytest.raises(ValidationError):
            Claim(text="something", quote="something")

    def test_extra_fields_are_refused(self):
        """A model inventing a `source` field alongside chunk_ids would look
        cited while citing nothing the system can check."""
        with pytest.raises(ValidationError):
            Claim(
                text="x", chunk_ids=["c1"], quote="x", source="somewhere else"
            )


class TestQuoteChecking:
    def test_a_real_quote_is_found(self):
        assert claim().quote_appears_in(CHUNK)

    def test_an_invented_quote_is_not(self):
        assert not claim(quote="The merger was abandoned.").quote_appears_in(CHUNK)

    def test_reflowed_whitespace_still_matches(self):
        """Models turn a line break into a space when quoting. That is not a
        fabrication and must not be scored as one."""
        reflowed = claim(quote="The merger  completed\non 14   March 2019.")
        assert reflowed.quote_appears_in(CHUNK)

    def test_a_non_breaking_space_still_matches(self):
        assert claim(quote="The merger completed on 14 March 2019.").quote_appears_in(CHUNK)

    def test_case_differences_still_match(self):
        assert claim(quote="the MERGER completed on 14 march 2019.").quote_appears_in(CHUNK)

    def test_a_paraphrase_does_not_match(self):
        """The check is deliberately strict: a tidied quote is indistinguishable
        from an invented one, so both fail."""
        assert not claim(quote="The merger was completed in March 2019.").quote_appears_in(CHUNK)

    def test_a_quote_from_a_different_chunk_does_not_match(self):
        assert not claim().quote_appears_in("An unrelated paragraph entirely.")


class TestNormalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("  hello   world  ", "hello world"),
            ("hello\nworld", "hello world"),
            ("HELLO WORLD", "hello world"),
            ("hello world", "hello world"),
        ],
    )
    def test_it_collapses_what_should_not_matter(self, raw, expected):
        assert normalise_for_match(raw) == expected


class TestChunkIds:
    def test_duplicates_are_collapsed(self):
        assert claim(chunk_ids=["c1", "c1", "c2"]).chunk_ids == ["c1", "c2"]

    def test_order_is_preserved(self):
        """The first citation is the primary source; a set would lose that."""
        assert claim(chunk_ids=["c3", "c1", "c2"]).chunk_ids == ["c3", "c1", "c2"]

    def test_surrounding_whitespace_is_stripped(self):
        assert claim(chunk_ids=[" c1 "]).chunk_ids == ["c1"]


class TestGroundedAnswer:
    def test_an_answer_with_no_claims_is_rejected(self):
        with pytest.raises(ValidationError):
            GroundedAnswer(claims=[])

    def test_cited_chunks_are_collected_in_order(self):
        answer = GroundedAnswer(claims=[
            claim(chunk_ids=["c2", "c1"]),
            claim(chunk_ids=["c1", "c3"]),
        ])
        assert answer.cited_chunk_ids == ["c2", "c1", "c3"]

    def test_it_renders_as_prose(self):
        answer = GroundedAnswer(claims=[
            claim(text="First point."), claim(text="Second point."),
        ])
        assert answer.text() == "First point. Second point."


class TestDeclining:
    def test_declining_needs_a_reason(self):
        with pytest.raises(ValidationError):
            InsufficientEvidence(searched_for="the merger date", missing="")

    def test_a_decline_cites_nothing(self):
        declined = InsufficientEvidence(
            searched_for="the merger date",
            missing="no chunk gives a date for the merger",
        )
        assert declined.cited_chunk_ids == []
        assert "no chunk gives a date" in declined.text()


class TestEnvelope:
    def test_a_grounded_answer_round_trips(self):
        envelope = AnswerEnvelope.model_validate(
            {"answer": {"kind": "grounded", "claims": [claim().model_dump()]}}
        )
        assert isinstance(envelope.answer, GroundedAnswer)

    def test_a_decline_round_trips(self):
        envelope = AnswerEnvelope.model_validate({
            "answer": {
                "kind": "insufficient_evidence",
                "searched_for": "the merger date",
                "missing": "no chunk gives a date",
            }
        })
        assert isinstance(envelope.answer, InsufficientEvidence)

    def test_the_discriminator_is_required(self):
        """Without it the union is guessed at, and a malformed grounded answer
        could be silently read as a decline."""
        with pytest.raises(ValidationError):
            AnswerEnvelope.model_validate({"answer": {"claims": [claim().model_dump()]}})

    def test_the_schema_names_both_branches(self):
        schema = answer_json_schema()
        rendered = str(schema)
        assert "grounded" in rendered
        assert "insufficient_evidence" in rendered

    def test_the_schema_marks_evidence_as_required(self):
        """If a provider enforces this schema, it must enforce the citation."""
        schema = answer_json_schema()
        claim_schema = schema["$defs"]["Claim"]
        assert set(claim_schema["required"]) == {"text", "chunk_ids", "quote"}


class TestStrictModeTranslation:
    """Pydantic writes valid JSON Schema; providers accept a subset.

    Two gaps bite here. A field with a Python default is omitted from
    `required`, which strict mode rejects outright. And a `$ref` inside a
    `oneOf` is not resolved by every provider, so the union's branches must be
    inlined. Both were found by sending the schema and reading the 400.
    """

    def test_every_property_becomes_required(self):
        """`kind` carries a default, so Pydantic leaves it out and strict mode
        refuses the schema for it."""
        schema, _ = strict_answer_schema()
        for branch in schema["properties"]["answer"]["oneOf"]:
            assert set(branch["required"]) == set(branch["properties"])

    def test_refs_are_inlined(self):
        """A provider that does not follow $ref inside oneOf rejects the union
        while pointing at a branch that is only a reference."""
        schema, _ = strict_answer_schema()
        assert "$ref" not in json.dumps(schema)
        assert "$defs" not in schema

    def test_the_discriminator_is_removed(self):
        """Its mapping points at $defs entries that inlining deleted, and a
        dangling mapping is itself rejected."""
        assert "discriminator" not in schema_of_answer_property()

    def test_additional_properties_is_closed_everywhere(self):
        schema, _ = strict_answer_schema()

        def check(node):
            if isinstance(node, dict):
                if node.get("type") == "object" and "properties" in node:
                    assert node.get("additionalProperties") is False
                for value in node.values():
                    check(value)
            elif isinstance(node, list):
                for item in node:
                    check(item)

        check(schema)

    def test_stripped_constraints_are_reported_not_discarded(self):
        """`minItems: 1` on chunk_ids is exactly the rule that enforces "at
        least one citation". The provider will not apply it, so the caller has
        to know it is gone and Pydantic has to enforce it on parse."""
        _, dropped = strict_answer_schema()
        assert any("chunk_ids" in d and "minItems" in d for d in dropped)

    def test_the_contract_still_enforces_what_the_schema_dropped(self):
        """The layered defence: provider enforces shape, Pydantic enforces the
        constraints the provider stripped."""
        with pytest.raises(ValidationError):
            Claim(text="x", chunk_ids=[], quote="x")

    def test_a_self_referential_schema_does_not_loop(self):
        recursive = {
            "$defs": {"Node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}},
            }},
            "type": "object",
            "properties": {"root": {"$ref": "#/$defs/Node"}},
        }
        translated, _ = for_strict_mode(recursive)
        assert translated  # returned rather than recursing forever


def schema_of_answer_property() -> str:
    schema, _ = strict_answer_schema()
    return json.dumps(schema["properties"]["answer"])
