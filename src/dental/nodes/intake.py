"""Graph entry for a fresh encounter: generate the patient the graph will play.

The Streamlit app generates a case before it invokes, so this node is a no-op
there. LangGraph Studio (and any other direct consumer of `graph.py`) starts from
an empty state and has no case bank, so the encounter makes its own patient.

The model comes from state, exactly as the phase nodes do, so one fake covers a
whole traversal in tests and a real provider is used in production.
"""

from dental.case_gen import DIFFICULTIES, generate_case
from dental.state import Phase


def node_intake(state: dict, llm=None) -> dict:
    """Generate a patient and rubric, unless the state already carries a patient."""
    if state.get("patient"):
        return state

    model = llm if llm is not None else state.get("llm")
    difficulty = state.get("difficulty") or "mid"
    if difficulty not in DIFFICULTIES:
        difficulty = "mid"

    bundle = generate_case(
        difficulty=difficulty,
        language=state.get("language", "en"),
        llm=model,
    )
    return {
        **state,
        "patient": bundle.patient.model_dump(),
        "rubric": [r.model_dump() for r in bundle.rubric],
        "phase": Phase.COMPLAINT.value,
    }
