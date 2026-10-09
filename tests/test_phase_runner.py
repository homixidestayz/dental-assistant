import pytest
from langchain_core.messages import AIMessage, HumanMessage

from dental.phase_runner import (
    MalformedModelOutput,
    build_patient_prompt,
    run_phase,
    should_advance,
)
from dental.state import TURN_CEILING, Phase, initial_state
from conftest import FakeLLM, FakeResponse


def _state(turns=0, total=0, phase=Phase.HISTORY):
    state = initial_state()
    state.update(
        phase=phase.value,
        turns_in_phase=turns,
        total_turns=total,
        patient={"age": 54, "sex": "female", "complaint": "My tooth is killing me."},
        rubric=[],
        messages=[HumanMessage(content="How can I help you today?")],
    )
    return state


def _turn(reply="ok", complete=False, reason=""):
    return {
        "patient_reply": reply,
        "phase_complete": complete,
        "phase_reason": reason,
    }


def test_advances_when_model_says_phase_complete():
    llm = FakeLLM(structured_response=_turn("It started Monday.", True, "history covered"))
    out = run_phase(_state(), Phase.HISTORY, llm=llm)
    assert out["phase"] == Phase.EXAMINATION.value
    assert out["phase_complete"] is True
    assert out["turns_in_phase"] == 0
    assert out["total_turns"] == 1


def test_loops_when_phase_not_complete():
    llm = FakeLLM(structured_response=_turn("Penicillin.", False, "more history needed"))
    out = run_phase(_state(), Phase.HISTORY, llm=llm)
    assert out["phase"] == Phase.HISTORY.value
    assert out["turns_in_phase"] == 1


def test_patient_reply_appended_as_ai_message():
    llm = FakeLLM(structured_response=_turn("I take metformin.", False))
    out = run_phase(_state(), Phase.HISTORY, llm=llm)
    assert isinstance(out["messages"][-1], AIMessage)
    assert out["messages"][-1].content == "I take metformin."


def test_turn_ceiling_force_advances():
    """A model that never finishes must not hang the encounter."""
    llm = FakeLLM(structured_response=_turn("I dunno.", False))
    out = run_phase(_state(turns=TURN_CEILING - 1), Phase.HISTORY, llm=llm)
    assert out["phase"] == Phase.EXAMINATION.value
    assert out["phase_reason"] == "turn ceiling reached"


def test_ceiling_reason_recorded_for_the_grader():
    """The live reason is overwritten each turn, so it must also be archived."""
    llm = FakeLLM(structured_response=_turn("I dunno.", False))
    out = run_phase(_state(turns=TURN_CEILING - 1), Phase.HISTORY, llm=llm)
    assert out["phase_reasons"][Phase.HISTORY.value] == "turn ceiling reached"


def test_normal_phase_reason_archived_too():
    llm = FakeLLM(structured_response=_turn("right", True, "history covered"))
    out = run_phase(_state(), Phase.HISTORY, llm=llm)
    assert out["phase_reasons"][Phase.HISTORY.value] == "history covered"


def test_malformed_output_retries_then_raises():
    """Never fall back to free text, never silently hang on a missing flag."""
    bad = FakeLLM(responses=[FakeResponse(structured_response={"patient_reply": "hi"})])
    with pytest.raises(MalformedModelOutput):
        run_phase(_state(), Phase.HISTORY, llm=bad)
    assert len(bad.calls) == 2  # one retry


def test_should_advance_routes_where_run_phase_left_the_phase():
    """Router, not a second decision: run_phase already advanced the phase.

    Asserting on post-run state is what the graph actually does. The earlier
    version of this test passed state as it looked *before* the turn, which made it
    green while the graph skipped every other phase.
    """
    looping = _state(turns=1)
    looping.update(phase=Phase.HISTORY.value, phase_complete=False)
    assert should_advance(looping) == Phase.HISTORY.value  # another turn here

    done = _state(turns=1)
    done.update(phase=Phase.EXAMINATION.value, phase_complete=True)
    assert should_advance(done) == Phase.EXAMINATION.value  # run_phase advanced it


def test_should_advance_debrief_routes_to_itself():
    """DEBRIEF is terminal; asking for its successor is the bug this guards."""
    assert should_advance({"phase": Phase.DEBRIEF.value}) == Phase.DEBRIEF.value


def test_should_advance_forces_next_at_the_ceiling():
    stuck = _state(turns=TURN_CEILING)
    stuck.update(phase=Phase.HISTORY.value, phase_complete=False)
    assert should_advance(stuck) == Phase.EXAMINATION.value


def test_patient_prompt_forbids_volunteering_and_breaking_character():
    """These constraints are the mechanism behind 'silent until the end'."""
    prompt = build_patient_prompt(
        {"age": 54, "sex": "female", "complaint": "My tooth hurts."}, Phase.HISTORY
    ).lower()
    assert "never volunteer" in prompt
    assert "never suggest" in prompt
    assert "never break character" in prompt


def test_patient_prompt_carries_hidden_findings_and_phase_goal():
    prompt = build_patient_prompt(
        {"age": 54, "sex": "female", "complaint": "My tooth hurts.",
         "hidden_findings": ["spontaneous bleeding of the gingiva"]},
        Phase.HISTORY,
    )
    assert "spontaneous bleeding of the gingiva" in prompt
    assert "allergies" in prompt.lower()
