"""Test doubles. Every test runs against these; nothing here touches the network."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class FakeResponse:
    content: str = ""
    structured_response: Any = None


@dataclass
class FakeLLM:
    """Scripted stand-in for a chat model.

    Callable as ``llm(system_prompt, human_prompt)`` so it matches the injection
    seam every production function takes. ``responses`` is a queue consumed one
    entry per call, for tests that script turn-by-turn behaviour; ``debrief_response``
    is held back for the grading call.
    """

    structured_response: Any = None
    responses: list = field(default_factory=list)
    debrief_response: FakeResponse | None = None
    calls: list = field(default_factory=list)

    def __call__(self, system_prompt: str = "", human_prompt: str = "") -> FakeResponse:
        self.calls.append((system_prompt, human_prompt))
        if self.debrief_response is not None and not self.responses:
            return self.debrief_response
        if self.responses:
            return self.responses.pop(0)
        return FakeResponse(content="", structured_response=self.structured_response)

    def with_structured(self, schema: Any) -> "FakeLLM":
        return self

    @property
    def prompts(self) -> str:
        """Every prompt this model was shown, joined — for assertions on rules."""
        return " ".join(f"{a} {b}" for a, b in self.calls)
