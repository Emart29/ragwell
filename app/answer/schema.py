"""Turns the answer contract into a schema a provider will actually accept.

Pydantic emits valid JSON Schema. Providers accept a subset of it, and the gap
is not cosmetic: a field with a Python default is left out of ``required``,
which strict schema modes reject outright with *"`required` is required to be
supplied and to be an array including every key in properties"*.

``kind`` on each answer variant has a default, so the contract as Pydantic
writes it is rejected by the very mode meant to enforce it. Optionality has to
be carried by the type admitting null rather than by absence from ``required``,
which is a translation step rather than a configuration flag.

Everything here rewrites the schema and never the contract. The Python model
keeps its defaults, so constructing an answer in code stays convenient.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

#: Keys that carry no meaning for a validator and that some providers reject.
#: Stripped rather than left to chance.
#: ``discriminator`` goes with them: once the branches are inlined its mapping
#: points at ``$defs`` entries that no longer exist, and a dangling mapping is
#: rejected by providers that read it.
COSMETIC_KEYS = frozenset({
    "title", "examples", "default", "$schema", "discriminator",
})

#: Value constraints strict modes commonly drop. They are removed here so the
#: schema is accepted, and the caller is told which ones went, because a
#: constraint silently discarded is a constraint nobody is enforcing.
VALUE_CONSTRAINTS = frozenset({
    "minLength", "maxLength", "pattern", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems",
    "multipleOf", "format",
})


def for_strict_mode(schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Rewrite a Pydantic schema into the form strict modes accept.

    Args:
        schema: The schema as Pydantic produced it.

    Returns:
        The rewritten schema, and the value constraints that were stripped from
        it. The caller is expected to enforce those after parsing: a length
        limit the provider will not apply still has to hold, and pretending
        otherwise is how a constraint quietly stops being one.
    """
    dropped: list[str] = []
    inlined = _inline_refs(deepcopy(schema))
    rewritten = _walk(inlined, dropped, path="")
    return rewritten, dropped


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace every ``$ref`` with the definition it points at.

    Groq resolves a ``$ref`` at the top level but not inside a ``oneOf``, and
    rejects the union with *"additionalProperties:false must be set on every
    object"* while pointing at a branch that is only a reference. Inlining
    sidesteps the question of which positions a given provider follows, at the
    cost of a larger schema — these are small.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [resolve(item, seen) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.split("/")[-1]
            if name in seen:
                # A self-referential schema cannot be inlined. None of these
                # contracts are, and silently looping would be worse than
                # leaving the reference for the provider to handle.
                return node
            target = deepcopy(defs.get(name, {}))
            merged = {k: v for k, v in node.items() if k != "$ref"}
            target.update(merged)
            return resolve(target, seen | {name})
        return {key: resolve(value, seen) for key, value in node.items()}

    resolved = resolve({k: v for k, v in schema.items() if k != "$defs"}, frozenset())
    return resolved


def _walk(node: Any, dropped: list[str], path: str) -> Any:
    if isinstance(node, list):
        return [_walk(item, dropped, f"{path}[]") for item in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in COSMETIC_KEYS:
            continue
        if key in VALUE_CONSTRAINTS:
            dropped.append(f"{path or 'root'}.{key}")
            continue
        out[key] = _walk(value, dropped, f"{path}.{key}" if path else key)

    if out.get("type") == "object" and "properties" in out:
        # Every property must appear in `required`. A property that was
        # genuinely optional would need its type widened to admit null, but
        # this contract has none: `kind` is absent from `required` only because
        # it carries a Python-side default.
        out["required"] = sorted(out["properties"].keys())
        out.setdefault("additionalProperties", False)

    return out


def strict_answer_schema() -> tuple[dict[str, Any], list[str]]:
    """The answer contract, ready to send to a provider that enforces schemas."""
    from app.answer.contract import answer_json_schema

    return for_strict_mode(answer_json_schema())
