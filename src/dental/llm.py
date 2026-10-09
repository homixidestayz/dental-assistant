"""Provider cascade: Groq, then Gemini, then OpenRouter, then OpenAI.

No provider-specific type escapes this module, so nothing downstream branches on
which model is live.
"""

import os
from typing import Callable

from dotenv import load_dotenv

load_dotenv()


class NoProviderConfigured(RuntimeError):
    """Raised when no provider key is present. Names every key that was checked."""


def _groq() -> "object":
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.7,
    )


def _gemini() -> "object":
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.7,
    )


def _openrouter() -> "object":
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"),
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
    )


def _openai() -> "object":
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.7,
    )


# Cascade order is fixed by the spec. Tuples are (env key, provider name, factory).
PROVIDERS: tuple[tuple[str, str, Callable], ...] = (
    ("GROQ_API_KEY", "groq", _groq),
    ("GOOGLE_API_KEY", "gemini", _gemini),
    ("OPENROUTER_API_KEY", "openrouter", _openrouter),
    ("OPENAI_API_KEY", "openai", _openai),
)


def _configured() -> tuple[str, str, Callable] | None:
    """The first provider whose key is a non-empty string. Reads env only."""
    for env_key, name, factory in PROVIDERS:
        if os.getenv(env_key, "").strip():
            return env_key, name, factory
    return None


def resolved_provider() -> str:
    """Provider name from env alone, so tests need no keys. "none" if unconfigured."""
    found = _configured()
    return found[1] if found else "none"


def get_llm() -> "object":
    """The configured chat model.

    Raises NoProviderConfigured naming every key checked, rather than letting a
    provider raise a missing-key error on the first turn.
    """
    found = _configured()
    if found is None:
        checked = ", ".join(env_key for env_key, _, _ in PROVIDERS)
        raise NoProviderConfigured(
            f"No LLM provider configured. Set one of: {checked}"
        )
    _, name, factory = found
    return factory()
