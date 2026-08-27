"""Tests for the named generation providers.

The rule these protect: a provider is named and stays named. The previous
wrapper tried Groq and quietly used Gemini on any failure, so an answer
attributed to one model may have come from the other — fatal to a comparison
between them, which is this project's headline measurement.
"""

from __future__ import annotations

import pytest

from app.llm.provider import (
    GenerationError,
    GeminiProvider,
    GroqProvider,
    LLMProvider,
    get_named_provider,
)


class TestNaming:
    def test_each_provider_is_reachable_by_name(self):
        assert isinstance(get_named_provider("groq"), GroqProvider)
        assert isinstance(get_named_provider("gemini"), GeminiProvider)

    def test_an_unknown_name_lists_the_known_ones(self):
        with pytest.raises(ValueError) as exc:
            get_named_provider("not_a_provider")
        assert "groq" in str(exc.value)
        assert "gemini" in str(exc.value)

    def test_a_model_override_is_honoured(self):
        assert get_named_provider("groq", "some-other-model").model == "some-other-model"

    def test_providers_are_reused_per_name_and_model(self):
        """Constructing a client per call would re-handshake on every query."""
        assert get_named_provider("groq") is get_named_provider("groq")


class TestNoSilentFallback:
    def test_a_groq_failure_does_not_reach_gemini(self, monkeypatch):
        """The failure that made the old wrapper unusable for a comparison."""
        groq = GroqProvider()
        if not groq.is_available():
            pytest.skip("needs GROQ_API_KEY")

        def explode(*_args, **_kwargs):
            raise RuntimeError("groq is down")

        monkeypatch.setattr(groq._client.chat.completions, "create", explode)
        with pytest.raises(GenerationError) as exc:
            groq.generate("anything")
        assert "groq" in str(exc.value)
        assert "gemini" not in str(exc.value).lower()

    def test_an_unconfigured_provider_raises_rather_than_returning_none(self):
        provider = GroqProvider(model="x")
        provider._client = None
        with pytest.raises(GenerationError):
            provider.generate("anything")

    def test_an_empty_completion_is_an_error_not_an_answer(self, monkeypatch):
        """Returning "" would be indistinguishable from a model with nothing
        to say, and downstream would ground an answer in it."""
        groq = GroqProvider()
        if not groq.is_available():
            pytest.skip("needs GROQ_API_KEY")

        class _Choice:
            message = type("M", (), {"content": ""})()
            finish_reason = "length"

        class _Response:
            choices = [_Choice()]

        monkeypatch.setattr(
            groq._client.chat.completions, "create", lambda **_: _Response()
        )
        with pytest.raises(GenerationError) as exc:
            groq.generate("anything")
        assert "finish_reason=length" in str(exc.value)


class TestCompatibilityWrapper:
    def test_it_still_returns_none_for_the_retrieval_strategies(self, monkeypatch):
        """HyDE and query expansion treat None as "fall back to hybrid", so
        changing that would silently alter which strategy ran."""
        wrapper = LLMProvider()

        def explode(*_args, **_kwargs):
            raise GenerationError("provider is down")

        monkeypatch.setattr(wrapper.provider, "generate", explode)
        assert wrapper.generate("anything") is None

    def test_it_reports_which_provider_it_uses(self):
        assert LLMProvider("gemini").provider_name == "gemini"
