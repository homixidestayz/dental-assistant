"""Graph node for the debrief: hand the encounter to the grader."""

from dental.grader import grade
from dental.state import Phase


def node_debrief(state: dict, llm=None) -> dict:
    """Grade the encounter and return the debrief in state.

    Reads the live `llm` from state the same way the phase nodes do, so one fake
    covers a whole traversal.
    """
    model = llm if llm is not None else state.get("llm")
    debrief = grade(
        case=state.get("patient", {}),
        rubric=state.get("rubric", []),
        transcript=state.get("messages", []),
        phase_reasons=state.get("phase_reasons", {}),
        llm=model,
    )
    return {
        **state,
        "phase": Phase.DEBRIEF.value,
        "debrief": debrief.model_dump(),
    }
