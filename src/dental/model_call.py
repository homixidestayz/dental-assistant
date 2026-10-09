"""One place that knows how to get structured output from a model.

Production talks to a LangChain chat model via `with_structured_output`. Tests pass a
callable double. The two shapes are incompatible — a real model is not callable and a
double is — so the dispatch lives here rather than in each module.

Checking the wrong attribute name (`with_structured`) silently sent every real model
down the double's path, which only surfaced on the first live call.
"""

from typing import Any

_STRUCTURED_OUTPUT = "with_structured_output"


def invoke_structured(llm: Any, schema: type, system: str, human: str) -> Any:
    """Return the model's output for `schema`, as an instance of it.

    Raises TypeError if `llm` supports neither interface, so a misconfigured model
    fails with a clear error instead of "'X' object is not callable".
    """
    binder = getattr(llm, _STRUCTURED_OUTPUT, None)
    if callable(binder):
        result = binder(schema).invoke([("system", system), ("human", human)])
        return _coerce(result, schema)

    if callable(llm):
        response = llm(system, human)
        return _coerce(getattr(response, "structured_response", None), schema)

    raise TypeError(
        f"{type(llm).__name__} is neither a LangChain chat model (no "
        f"{_STRUCTURED_OUTPUT}) nor a callable test double"
    )


def _coerce(result: Any, schema: type) -> Any:
    """An instance of `schema` whether the model gave us one or a plain dict."""
    if isinstance(result, schema):
        return result
    if isinstance(result, dict):
        return schema(**result)
    raise ValueError(
        f"expected {schema.__name__} or a dict, got {type(result).__name__}"
    )
