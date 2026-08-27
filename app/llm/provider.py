"""Named LLM providers, with no silent fallback between them.

Two rules, both of which the previous version broke:

* **A provider is named and stays named.** The old wrapper tried Groq, and on
  any failure quietly used Gemini instead. An answer attributed to one model may
  have been produced by another, which is fatal to any comparison between them —
  and this project's headline measurement is exactly such a comparison.
* **A configured model that the provider no longer serves is reported.** Both
  models this file used to pin have since been withdrawn:
  ``llama-3.3-70b-versatile`` and ``gemini-1.5-flash``. Neither failure was
  visible, because the fallback swallowed the first and ``generate`` returned
  ``None`` for the second.

The legacy ``generate()`` helper is kept for the retrieval strategies that
already call it, and still returns ``Optional[str]`` so their fallback
behaviour is unchanged. New code should use ``get_named_provider`` and handle
``GenerationError``, so a failure is impossible to mistake for an empty answer.
"""

from __future__ import annotations

from typing import Optional

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class GenerationError(RuntimeError):
    """A generation request failed.

    Raised rather than returning ``None``: a caller that forgets to check a
    ``None`` produces an answer grounded in nothing and reports no error, which
    is the failure this layer exists to prevent.
    """


class GenerationProvider:
    """One provider, one model, no fallback elsewhere."""

    name = "unnamed"

    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} model={self.model}>"


class GroqProvider(GenerationProvider):
    """Groq, over the official client."""

    name = "groq"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(model or settings.GROQ_MODEL)
        self._client = None
        if not settings.GROQ_API_KEY:
            logger.info("GROQ_API_KEY not set; Groq provider unavailable")
            return
        try:
            from groq import Groq

            self._client = Groq(api_key=settings.GROQ_API_KEY)
            logger.info("Groq provider ready (model=%s)", self.model)
        except ImportError:
            logger.warning("groq package not installed; Groq provider unavailable")
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            logger.error("Groq provider failed to initialise: %s", exc)

    def is_available(self) -> bool:
        return self._client is not None

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        if self._client is None:
            raise GenerationError("Groq provider is not configured")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=settings.GENERATION_TEMPERATURE,
                timeout=settings.GENERATION_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"groq/{self.model}: {exc}") from exc

        text = response.choices[0].message.content
        if not text:
            # An empty completion is a failure with a cause worth naming, not an
            # answer. finish_reason distinguishes a length cut from a refusal.
            reason = getattr(response.choices[0], "finish_reason", "unknown")
            raise GenerationError(
                f"groq/{self.model}: empty completion (finish_reason={reason})"
            )
        return text


class GeminiProvider(GenerationProvider):
    """Gemini, used for the long-context arm.

    Kept separate from Groq rather than sharing a client, because the whole
    point of having it is to compare the two, and a shared fallback path would
    let one arm answer for the other.
    """

    name = "gemini"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(model or settings.GEMINI_MODEL)
        self._model = None
        if not settings.GEMINI_API_KEY:
            logger.info("GEMINI_API_KEY not set; Gemini provider unavailable")
            return
        try:
            import google.generativeai as genai

            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._model = genai.GenerativeModel(self.model)
            logger.info("Gemini provider ready (model=%s)", self.model)
        except ImportError:
            logger.warning(
                "google-generativeai not installed; Gemini provider unavailable"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Gemini provider failed to initialise: %s", exc)

    def is_available(self) -> bool:
        return self._model is not None

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        if self._model is None:
            raise GenerationError("Gemini provider is not configured")
        try:
            response = self._model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": settings.GENERATION_TEMPERATURE,
                },
            )
            text = response.text
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"gemini/{self.model}: {exc}") from exc

        if not text:
            raise GenerationError(f"gemini/{self.model}: empty completion")
        return text


PROVIDERS: dict[str, type[GenerationProvider]] = {
    "groq": GroqProvider,
    "gemini": GeminiProvider,
}

_named: dict[str, GenerationProvider] = {}


def get_named_provider(name: str, model: str | None = None) -> GenerationProvider:
    """Return the provider called ``name``, and only that provider.

    Args:
        name: ``groq`` or ``gemini``.
        model: Model id, defaulting to the configured one for that provider.

    Raises:
        ValueError: If the name is unknown, naming the alternatives.
    """
    key = f"{name}:{model or ''}"
    if key not in _named:
        try:
            factory = PROVIDERS[name]
        except KeyError:
            known = ", ".join(sorted(PROVIDERS))
            raise ValueError(f"unknown provider {name!r}. Known: {known}") from None
        _named[key] = factory(model)
    return _named[key]


class LLMProvider:
    """Compatibility wrapper for the retrieval strategies.

    HyDE and query expansion already treat a missing generation as a reason to
    fall back to hybrid search, so this keeps returning ``None`` for them rather
    than changing behaviour those paths depend on. It differs from the old
    version in saying which provider it used, and in not pretending one
    provider's answer came from another.
    """

    def __init__(self, provider_name: str | None = None) -> None:
        self.provider_name = provider_name or settings.DEFAULT_LLM_PROVIDER
        self.provider = get_named_provider(self.provider_name)

    def generate(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Generate text, or ``None`` when the configured provider cannot."""
        try:
            return self.provider.generate(prompt, max_tokens)
        except GenerationError as exc:
            logger.error("%s", exc)
            return None

    def is_available(self) -> bool:
        return self.provider.is_available()


_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        _provider = LLMProvider()
    return _provider


def generate(prompt: str, max_tokens: int = 500) -> Optional[str]:
    """Generate with the default provider. ``None`` when it cannot."""
    return get_provider().generate(prompt, max_tokens)
