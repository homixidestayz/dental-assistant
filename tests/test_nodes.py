from dental.nodes import (
    closing,
    complaint,
    diagnosis,
    examination,
    history,
    investigations,
    management,
)
from dental.state import Phase
from conftest import FakeLLM


def _never_complete():
    return FakeLLM(
        structured_response={
            "patient_reply": "ok",
            "phase_complete": False,
            "phase_reason": "",
        }
    )


def test_each_node_delegates_to_its_own_phase():
    cases = [
        (complaint, Phase.COMPLAINT),
        (history, Phase.HISTORY),
        (examination, Phase.EXAMINATION),
        (investigations, Phase.INVESTIGATIONS),
        (diagnosis, Phase.DIAGNOSIS),
        (management, Phase.MANAGEMENT),
        (closing, Phase.CLOSING),
    ]
    for module, phase in cases:
        llm = _never_complete()
        state = {
            "phase": phase.value,
            "turns_in_phase": 0,
            "total_turns": 0,
            "patient": {},
            "rubric": [],
            "phase_reasons": {},
            "messages": [],
        }
        out = getattr(module, f"node_{phase.value}")(state, llm=llm)
        assert out["phase"] == phase.value  # not complete, so it self-loops
        assert out["turns_in_phase"] == 1


def test_nodes_module_exports_all_seven():
    import dental.nodes as nodes

    for phase in (
        "complaint",
        "history",
        "examination",
        "investigations",
        "diagnosis",
        "management",
        "closing",
    ):
        assert hasattr(nodes, phase)


def test_node_advances_when_its_phase_completes():
    """Each node must advance on its own phase's flag, not the previous phase's."""
    llm = FakeLLM(
        structured_response={
            "patient_reply": "Thank you.",
            "phase_complete": True,
            "phase_reason": "done",
        }
    )
    state = {
        "phase": Phase.MANAGEMENT.value,
        "turns_in_phase": 0,
        "total_turns": 0,
        "patient": {},
        "rubric": [],
        "phase_reasons": {},
        "messages": [],
    }
    out = management.node_management(state, llm=llm)
    assert out["phase"] == Phase.CLOSING.value
