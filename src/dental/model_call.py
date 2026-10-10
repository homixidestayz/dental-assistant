"""One place that knows how to get structured output from a model.

Production talks to a LangChain chat model via `with_structured_output`. Tests pass a
callable double. The two shapes are incompatible — a real model is not callable and a
double is — so the dispatch lives here rather than in each module.

Checking the wrong attribute name (`with_structured`) silently sent every real model
down the double's path, which only surfaced on the first live call.

Structured output is also the flakiest call a provider makes. Groq's gpt-oss models
occasionally emit a tool call with misnamed keys ("differentals" for "differentials"),
and Groq validates against the schema server-side, so it fails with a provider error
before we ever see a ValidationError. Two defences live here:

- a bounded retry per model, keyed on the provider's own wording for a schema mismatch,
  because resampling usually fixes a misnamed key;
- the provider cascade, applied only when *we* resolved the model (the caller passed
  `llm=None`). If Groq cannot produce valid structured output, the same call is tried
  on Gemini, then OpenRouter, then OpenAI — the fallback order the spec asks for.

A model the caller supplied explicitly is never silently swapped, so tests and any
caller pinning a specific model stay deterministic.
"""

from typing import Any

_STRUCTURED_OUTPUT = "with_structured_output"

# Provider wordings meaning "the model's tool call did not match the schema".
_STRUCTURED_FAILURE_MARKERS = (
    "tool_use_failed",
    "tool call validation failed",
    "did not match schema",
    "failed_generation",
)

_MAX_ATTEMPTS = 3


def _is_structured_output_failure(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _STRUCTURED_FAILURE_MARKERS)


def invoke_structured(llm: Any, schema: type, system: str, human: str) -> Any:
    """Return the model's output for `schema`, as an instance of it.

    When `llm` is None the configured provider cascade is used, so a provider that
    cannot return valid structured output is retried on the next one. When `llm` is
    given, only that model is used.

    Raises TypeError if an explicitly supplied `llm` supports neither interface, so a
    misconfigured model fails with a clear error instead of "'X' object is not callable".
    """
    if schema is None:
        raise TypeError("invoke_structured requires a schema")

    if llm is None:
        models = _configured_models()
    else:
        models = [llm]

    last_error: BaseException | None = None
    for index, model in enumerate(models):
        try:
            return _invoke_one(model, schema, system, human)
        except Exception as exc:  # noqa: BLE001 — fall through to the next provider
            last_error = exc
            if index == len(models) - 1:
                raise
    raise last_error  # pragma: no cover - the loop either returns or raises


def _configured_models() -> list:
    """Every configured provider's model, in cascade order.

    Falls back to `get_llm()` for its message when nothing is configured, so the
    caller still sees which env keys were checked.
    """
    from dental.llm import available_llms, get_llm

    models = available_llms()
    if not models:
        get_llm()  # raises NoProviderConfigured naming every key
    return models


def _invoke_one(llm: Any, schema: type, system: str, human: str) -> Any:
    """One model, with the bounded retry for provider-side schema mismatches."""
    if callable(getattr(llm, _STRUCTURED_OUTPUT, None)):
        return _invoke_model(llm, schema, system, human)
    if callable(llm):
        response = llm(system, human)
        return _coerce(getattr(response, "structured_response", None), schema)
    raise TypeError(
        f"{type(llm).__name__} is neither a LangChain chat model (no "
        f"{_STRUCTURED_OUTPUT}) nor a callable test double"
    )


def _invoke_model(model: Any, schema: type, system: str, human: str) -> Any:
    last_error: BaseException | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            result = model.with_structured_output(schema).invoke(
                [("system", system), ("human", human)]
            )
            return _coerce(result, schema)
        except Exception as exc:  # noqa: BLE001 — re-raised unless a schema mismatch
            last_error = exc
            if attempt >= _MAX_ATTEMPTS or not _is_structured_output_failure(exc):
                raise
    raise last_error  # pragma: no cover - the loop either returns or raises


def _coerce(result: Any, schema: type) -> Any:
    """An instance of `schema` whether the model gave us one or a plain dict."""
    if isinstance(result, schema):
        return result
    if isinstance(result, dict):
        return schema(**result)
    raise ValueError(
        f"expected {schema.__name__} or a dict, got {type(result).__name__}"
    )
