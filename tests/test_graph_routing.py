"""Phase machine routing, driven one learner turn per invocation.

Each test drives the graph the way the Streamlit app does: append a learner message,
invoke once, carry state forward. One invocation is one encounter turn.
"""

from langchain_core.messages import HumanMessage

from dental.graph import build_graph, get_graph
from dental.state import TURN_CEILING, Phase
from conftest import FakeLLM, FakeResponse

PHASE_ORDER_EXPECTED = [
    Phase.COMPLAINT.value,
    Phase.HISTORY.value,
    Phase.EXAMINATION.value,
    Phase.INVESTIGATIONS.value,
    Phase.DIAGNOSIS.value,
    Phase.MANAGEMENT.value,
    Phase.CLOSING.value,
]

STATE_KEYS = (
    "messages",
    "phase",
    "patient",
    "rubric",
    "turns_in_phase",
    "total_turns",
    "phase_reasons",
    "debrief",
    "llm",
)


def _debrief_response(score=5, max_score=7):
    return FakeResponse(
        structured_response={
            "phase_scores": [
                {
                    "phase": "history",
                    "score": score,
                    "max_score": max_score,
                    "covered": ["chief complaint"],
                    "missed": ["drug allergies"],
                    "reasoning": "partial history",
                }
            ],
            "overall": 71,
            "band": "competent",
            "focus_areas": ["ask about drug allergies"],
            "strengths": ["clear opening"],
        }
    )


class Scripted(FakeLLM):
    """complete=True finishes each phase in one turn; False never finishes.

    stuck_turns holds the number of leading turns the model refuses to complete,
    which is how a single phase is pushed to its ceiling.
    """

    def __init__(self, complete=True, stuck_turns=0):
        super().__init__(debrief_response=_debrief_response())
        self.complete = complete
        self.stuck_turns = stuck_turns
        self.turns = 0

    def __call__(self, system_prompt="", human_prompt=""):
        if "examiner" in system_prompt.lower():
            self.calls.append((system_prompt, human_prompt))
            return self.debrief_response
        self.calls.append((system_prompt, human_prompt))
        self.turns += 1
        return FakeResponse(
            structured_response={
                "patient_reply": "ok",
                "phase_complete": self.complete and self.turns > self.stuck_turns,
                "phase_reason": "",
            }
        )


def _never_complete():
    return Scripted(complete=False)


def _always_complete():
    return Scripted(complete=True)


def _state(phase=Phase.COMPLAINT, llm=None):
    state = {
        "messages": [],
        "phase": phase.value,
        "patient": {"age": 54, "complaint": "My tooth is killing me."},
        "rubric": [{"phase": "history", "criteria": ["allergies"], "max_score": 7}],
        "turns_in_phase": 0,
        "total_turns": 0,
        "debrief": None,
        "phase_reasons": {},
    }
    if llm is not None:
        state["llm"] = llm
    return state


def _turn(graph, state, thread_id, line="how can I help"):
    """One learner turn. Returns the carried-forward state."""
    state = dict(state)
    state["messages"] = list(state.get("messages", [])) + [HumanMessage(content=line)]
    out = graph.invoke(
        state, config={"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    )
    return {key: out[key] for key in STATE_KEYS}


def _drive(state, llm, turns, thread_id="drive"):
    graph = get_graph()
    for i in range(turns):
        state = _turn(graph, state, thread_id, f"learner line {i}")
    return state


def test_graph_exposes_a_node_per_phase_and_debrief():
    builder = build_graph()
    nodes = set(builder.nodes)
    for phase in Phase:
        assert phase.value in nodes


def test_happy_path_visits_every_phase_in_order():
    llm = _always_complete()
    state = _drive(_state(llm=llm), llm, len(PHASE_ORDER_EXPECTED))
    assert set(state["phase_reasons"]) == set(PHASE_ORDER_EXPECTED)
    # the closing phase advanced into the debrief
    assert state["phase"] == Phase.DEBRIEF.value


def test_wrong_answers_do_not_loop_forever():
    """A model that never finishes still leaves each phase at the ceiling."""
    llm = _never_complete()
    state = _drive(_state(llm=llm), llm, 7 * TURN_CEILING)
    assert state["phase"] == Phase.DEBRIEF.value
    assert state["phase_reasons"][Phase.HISTORY.value] == "turn ceiling reached"


def test_ceiling_bounds_a_stuck_phase_at_eight_turns():
    """Complaint never completes, so it costs exactly TURN_CEILING turns."""
    llm = _never_complete()
    state = _drive(_state(llm=llm), llm, TURN_CEILING + 1, "ceiling")
    assert state["total_turns"] == TURN_CEILING + 1
    assert state["phase"] == Phase.HISTORY.value
    assert state["phase_reasons"][Phase.COMPLAINT.value] == "turn ceiling reached"


def test_a_single_stuck_phase_is_bounded_not_the_whole_encounter():
    """One phase overruns to its ceiling; the rest advance in a single turn each."""
    llm = Scripted(complete=True, stuck_turns=TURN_CEILING)
    # complaint takes its full ceiling, then the six remaining phases take one turn each
    total = TURN_CEILING + (len(PHASE_ORDER_EXPECTED) - 1)
    state = _drive(_state(llm=llm), llm, total, "single-stuck")

    assert state["total_turns"] == total
    assert state["phase_reasons"][Phase.COMPLAINT.value] == "turn ceiling reached"
    assert state["phase_reasons"][Phase.CLOSING.value] == ""
    assert state["phase"] == Phase.DEBRIEF.value


def test_injected_llm_is_used_by_the_nodes():
    """State-carried llm must actually reach the model, not a module global."""
    llm = _always_complete()
    graph = get_graph()
    state = _state(llm=llm)
    before = len(llm.calls)
    _turn(graph, state, "t4")
    assert len(llm.calls) - before == 1


def test_graph_export_exists_for_langgraph_json():
    from dental import graph as graph_module

    assert hasattr(graph_module, "graph")
    assert hasattr(graph_module, "ALIAS")


def test_get_graph_accepts_a_custom_checkpointer():
    from langgraph.checkpoint.memory import MemorySaver

    compiled = get_graph(checkpointer=MemorySaver())
    assert compiled.checkpointer is not None


def test_a_fresh_state_starts_at_the_complaint():
    from dental.graph import entry_phase

    assert entry_phase({}) == Phase.COMPLAINT.value
    assert entry_phase({"phase": Phase.MANAGEMENT.value}) == Phase.MANAGEMENT.value
    assert entry_phase({"phase": "nonsense"}) == Phase.COMPLAINT.value
