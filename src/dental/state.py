"""Encounter state: the clinical phase enum and the graph's state shape."""

from enum import Enum
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class Phase(str, Enum):
    COMPLAINT = "complaint"
    HISTORY = "history"
    EXAMINATION = "examination"
    INVESTIGATIONS = "investigations"
    DIAGNOSIS = "diagnosis"
    MANAGEMENT = "management"
    CLOSING = "closing"
    DEBRIEF = "debrief"


PHASE_ORDER: list[Phase] = [
    Phase.COMPLAINT,
    Phase.HISTORY,
    Phase.EXAMINATION,
    Phase.INVESTIGATIONS,
    Phase.DIAGNOSIS,
    Phase.MANAGEMENT,
    Phase.CLOSING,
]

# A model that decides its phase goals are never met must not hang the encounter.
# A real history of complaint is 4 to 6 turns, so this only fires on a stuck run.
TURN_CEILING: int = 8

CEILING_REASON: str = "turn ceiling reached"

# Supported encounter languages. The whole encounter runs in this language.
LANGUAGES: tuple[str, ...] = ("en", "fr", "ar")

# Display names for the UI.
LANGUAGE_LABELS: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "ar": "Arabic",
}


class DentalState(TypedDict, total=False):
    """Everything the encounter carries from turn to turn.

    `llm` rides in state rather than a module global so tests substitute a fake
    without monkeypatching, and so the Streamlit app can supply one model per
    session. It is an injected dependency, not encounter data.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    phase: str
    patient: dict
    rubric: list
    phase_complete: bool
    phase_reason: str
    phase_reasons: dict
    turns_in_phase: int
    total_turns: int
    debrief: Any
    llm: Any
    language: str


def next_phase(phase: Phase) -> Phase:
    """The phase after `phase`. Raises on DEBRIEF, which has no successor."""
    if phase is Phase.DEBRIEF:
        raise ValueError("debrief is terminal and has no next phase")
    index = PHASE_ORDER.index(phase)
    if index + 1 >= len(PHASE_ORDER):
        return Phase.DEBRIEF
    return PHASE_ORDER[index + 1]


def initial_state() -> DentalState:
    return DentalState(
        messages=[],
        phase=Phase.COMPLAINT.value,
        patient={},
        rubric=[],
        phase_complete=False,
        phase_reason="",
        phase_reasons={},
        turns_in_phase=0,
        total_turns=0,
        debrief=None,
        language="en",
    )
