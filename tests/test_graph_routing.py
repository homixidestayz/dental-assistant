import pytest

from dental.graph import build_graph, get_graph
from dental.state import TURN_CEILING, Phase
from conftest import FakeLLM, FakeResponse


def _never_complete():
    return FakeLLM(
        structured_response={
            "patient_reply": "...",
            "phase_complete": False,
            "phase_reason": "",
        }
    )


def _always_complete():
    return FakeLLM(
        structured_response={
            "patient_reply": "yes",
            "phase_complete": True,
            "phase_reason": "",
        }
    )


def _state(phase=Phase.COMPLAINT, llm=None):
    state = {
        "messages": [],
        "phase": phase.value,
        "patient": {},
        "rubric": [],
        "turns_in_phase": 0,
        "total_turns": 0,
        "debrief": None,
        "phase_reasons": {},
    }
    if llm is not None:
        state["llm"] = llm
    return state


def test_graph_exposes_a_node_per_phase_and_debrief():
    builder = build_graph()
    nodes = set(builder.nodes)
    for phase in Phase:
        assert phase.value in nodes


def test_wrong_answers_do_not_loop_forever():
    """A model that never finishes must still reach the debrief."""
    graph = get_graph()
    out = graph.invoke(
        _state(llm=_never_complete()),
        config={"configurable": {"thread_id": "t1"}, "recursion_limit": 500},
    )
    assert out["phase"] == Phase.DEBRIEF.value


def test_happy_path_visits_every_phase_in_order():
    graph = get_graph()
    out = graph.invoke(
        _state(llm=_always_complete()),
        config={"configurable": {"thread_id": "t2"}, "recursion_limit": 200},
    )
    assert out["phase"] == Phase.DEBRIEF.value
    assert out["total_turns"] == len(PHASE_ORDER_EXPECTED)
    assert set(out["phase_reasons"]) == set(PHASE_ORDER_EXPECTED)


PHASE_ORDER_EXPECTED = [
    Phase.COMPLAINT.value,
    Phase.HISTORY.value,
    Phase.EXAMINATION.value,
    Phase.INVESTIGATIONS.value,
    Phase.DIAGNOSIS.value,
    Phase.MANAGEMENT.value,
    Phase.CLOSING.value,
]


def test_ceiling_bounds_a_stuck_phase_at_eight_turns():
    """A model that completes nothing costs exactly TURN_CEILING per phase."""
    graph = get_graph()
    out = graph.invoke(
        _state(Phase.HISTORY, llm=_never_complete()),
        config={"configurable": {"thread_id": "t3"}, "recursion_limit": 500},
    )
    # START routes to complaint, so all seven phases are played and each is capped.
    assert out["total_turns"] == len(PHASE_ORDER_EXPECTED) * TURN_CEILING
    assert out["phase_reasons"][Phase.HISTORY.value] == "turn ceiling reached"


def test_a_single_stuck_phase_is_bounded_not_the_whole_encounter():
    """One phase overruns; the rest advance normally. The learner's time is capped
    per phase, not per encounter."""
    from dental.state import Phase as P

    class Scripted(FakeLLM):
        def __call__(self, system_prompt="", human_prompt=""):
            self.calls.append((system_prompt, human_prompt))
            turn = int(human_prompt.split("Turn ")[1].split(".")[0])
            # Never complete during the first two phases' worth of turns; the
            # ceiling carries each one out. After that, complete immediately.
            return FakeResponse(
                structured_response={
                    "patient_reply": "ok",
                    "phase_complete": turn > 2 * TURN_CEILING,
                    "phase_reason": "",
                }
            )

    graph = get_graph()
    out = graph.invoke(
        _state(llm=Scripted()),
        config={"configurable": {"thread_id": "t6"}, "recursion_limit": 500},
    )
    # complaint and history each capped at 8, then the remaining 5 phases at 1 turn
    assert out["total_turns"] == 2 * TURN_CEILING + 5
    assert out["phase_reasons"][P.COMPLAINT.value] == "turn ceiling reached"
    assert out["phase_reasons"][P.HISTORY.value] == "turn ceiling reached"
    assert out["phase_reasons"][P.CLOSING.value] == ""


def test_injected_llm_is_used_by_the_nodes():
    """State-carried llm must actually reach the model, not a module global."""
    llm = _always_complete()
    graph = get_graph()
    graph.invoke(
        {**_state(), "llm": llm},
        config={"configurable": {"thread_id": "t4"}, "recursion_limit": 200},
    )
    assert len(llm.calls) == len(PHASE_ORDER_EXPECTED)


def test_graph_export_exists_for_langgraph_json():
    from dental import graph as graph_module

    assert hasattr(graph_module, "graph")
    assert hasattr(graph_module, "ALIAS")


def test_get_graph_accepts_a_custom_checkpointer():
    from langgraph.checkpoint.memory import MemorySaver

    compiled = get_graph(checkpointer=MemorySaver())
    assert compiled.checkpointer is not None
