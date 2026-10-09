"""The one-turn-per-invoke contract the Streamlit app depends on.

The app invokes the graph once per learner message and expects exactly one patient
reply back. Without this the graph replays every phase on each keystroke, because
START routes to the complaint node.
"""

import pytest
from langchain_core.messages import HumanMessage

from dental.graph import get_graph
from dental.state import Phase
from conftest import FakeLLM, FakeResponse


class Scripted(FakeLLM):
    """Completes every phase immediately; grades on request."""

    def __init__(self, complete=True):
        super().__init__()
        self.complete = complete
        self.phase_tags = []

    def __call__(self, system_prompt="", human_prompt=""):
        self.calls.append((system_prompt, human_prompt))
        if "examiner" in system_prompt.lower():
            return FakeResponse(
                structured_response={
                    "phase_scores": [
                        {
                            "phase": "history",
                            "score": 5,
                            "max_score": 7,
                            "covered": [],
                            "missed": [],
                            "reasoning": "",
                        }
                    ],
                    "overall": 71,
                    "band": "competent",
                    "focus_areas": [],
                    "strengths": [],
                }
            )
        if "working on:" in system_prompt:
            self.phase_tags.append(
                system_prompt.split("working on:")[1].split("\n")[0].strip()[:30]
            )
        return FakeResponse(
            structured_response={
                "patient_reply": "ok",
                "phase_complete": self.complete,
                "phase_reason": "",
            }
        )


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


def _fresh(llm):
    return {
        "messages": [],
        "phase": Phase.COMPLAINT.value,
        "patient": {"age": 50, "complaint": "my tooth hurts"},
        "rubric": [{"phase": "history", "criteria": ["x"], "max_score": 7}],
        "turns_in_phase": 0,
        "total_turns": 0,
        "phase_reasons": {},
        "debrief": None,
        "llm": llm,
    }


def _step(graph, state, learner_line, thread_id, add_message=True):
    """One invocation: optionally append a learner message, invoke, carry state forward."""
    state = dict(state)
    if add_message:
        state["messages"] = list(state["messages"]) + [HumanMessage(content=learner_line)]
    out = graph.invoke(
        state, config={"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    )
    return {key: out[key] for key in STATE_KEYS}


def test_each_invoke_plays_exactly_one_turn():
    """The app sends one learner message and expects one patient reply."""
    llm = Scripted()
    graph = get_graph()
    state = _fresh(llm)

    for i in range(4):
        before = len(llm.calls)
        state = _step(graph, state, f"learner line {i}", "one-turn")
        assert len(llm.calls) - before == 1, (
            f"turn {i} made {len(llm.calls) - before} model calls, expected 1"
        )


def test_turns_do_not_restart_at_the_complaint_phase():
    """Phases advance across invokes instead of replaying from START each time."""
    llm = Scripted()
    graph = get_graph()
    state = _fresh(llm)

    phases = []
    for i in range(4):
        state = _step(graph, state, f"learner line {i}", "advance")
        phases.append(state["phase"])

    assert phases == [
        Phase.HISTORY.value,
        Phase.EXAMINATION.value,
        Phase.INVESTIGATIONS.value,
        Phase.DIAGNOSIS.value,
    ], f"phases restarted instead of advancing: {phases}"


def test_debrief_is_its_own_invocation():
    """Closing advances to DEBRIEF; grading is the next turn.

    The debrief node is only entered once state already says DEBRIEF, so the
    encounter's final turn does not grade. The caller drives one more invocation.
    """
    llm = Scripted()
    graph = get_graph()
    state = _fresh(llm)

    for i in range(7):
        state = _step(graph, state, f"learner line {i}", "debrief-once")

    # the seventh turn closed the encounter but did not grade
    assert state["phase"] == Phase.DEBRIEF.value
    assert not [s for s, _ in llm.calls if "examiner" in s.lower()]

    # the caller follows up with no new learner message, exactly as the app does
    state = _step(graph, state, "", "debrief-once", add_message=False)
    assert state["debrief"] is not None
    assert state["debrief"]["band"] == "competent"


def test_grading_happens_once_not_on_every_later_turn():
    """A learner poking at a finished encounter must not re-grade silently."""
    llm = Scripted()
    graph = get_graph()
    state = _fresh(llm)

    for i in range(8):
        state = _step(graph, state, f"learner line {i}", "grade-once")

    first = state["debrief"]["overall"]
    for _ in range(3):
        state = _step(graph, state, "anything else?", "grade-once", add_message=False)
        assert state["debrief"]["overall"] == first

    grader_calls = [s for s, _ in llm.calls if "examiner" in s.lower()]
    assert len(grader_calls) == 1, f"graded {len(grader_calls)} times, expected once"


def test_a_stuck_phase_does_not_replay_completed_phases():
    """With a model that never completes, turns still land in the right phase."""
    llm = Scripted(complete=False)
    graph = get_graph()
    state = _fresh(llm)

    # 10 turns at the 8-turn ceiling per phase.
    for i in range(10):
        state = _step(graph, state, f"learner line {i}", "stuck")

    assert state["phase"] in {p.value for p in Phase}
    # complaint's 8 turns are not repeated after it advances
    assert state["total_turns"] == 10
    assert len(llm.calls) == 10


def test_turns_in_phase_accumulates_across_invokes():
    """The app does not own the phase counter; the graph must."""
    llm = Scripted(complete=False)
    graph = get_graph()
    state = _fresh(llm)

    counts = []
    for i in range(4):
        state = _step(graph, state, f"learner line {i}", "counter")
        counts.append(state["turns_in_phase"])

    assert counts == [1, 2, 3, 4]
