"""The patient must see the conversation so far, or it repeats itself.

Found by playing it: the model kept re-asking and re-answering the same questions.
`run_phase` builds a prompt from the patient and the turn counter but never includes
what was already said, so each turn was a fresh conversation with no memory.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from dental.phase_runner import run_phase
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


def test_prior_learner_questions_are_in_the_prompt():
    llm = _never_complete()
    state = {
        "patient": {"age": 54, "complaint": "my tooth hurts"},
        "phase": Phase.HISTORY.value,
        "turns_in_phase": 1,
        "total_turns": 1,
        "phase_reasons": {},
        "messages": [
            HumanMessage(content="How long has this been going on?"),
            AIMessage(content="About a week."),
            HumanMessage(content="Any medications?"),
        ],
    }
    run_phase(state, Phase.HISTORY, llm=llm)
    prompt = llm.prompts
    assert "How long has this been going on?" in prompt
    assert "Any medications?" in prompt


def test_prior_patient_answers_are_in_the_prompt():
    llm = _never_complete()
    state = {
        "patient": {"age": 54, "complaint": "my tooth hurts"},
        "phase": Phase.HISTORY.value,
        "turns_in_phase": 1,
        "total_turns": 1,
        "phase_reasons": {},
        "messages": [
            HumanMessage(content="How long has this been going on?"),
            AIMessage(content="About a week, and it throbs at night."),
        ],
    }
    run_phase(state, Phase.HISTORY, llm=llm)
    assert "About a week, and it throbs at night." in llm.prompts


def test_the_prompt_tells_the_patient_not_to_repeat_itself():
    """The constraint has to be in the prompt, or the model has no way to know."""
    llm = _never_complete()
    state = {
        "patient": {"age": 54, "complaint": "my tooth hurts"},
        "phase": Phase.HISTORY.value,
        "turns_in_phase": 2,
        "total_turns": 2,
        "phase_reasons": {},
        "messages": [HumanMessage(content="Any medications?")],
    }
    run_phase(state, Phase.HISTORY, llm=llm)
    lowered = llm.prompts.lower()
    assert "already" in lowered
    assert "conversation so far" in lowered


def test_conversation_grows_across_turns():
    """Each turn sees everything prior, not just the last exchange."""
    llm = _never_complete()
    state = {
        "patient": {},
        "phase": Phase.HISTORY.value,
        "turns_in_phase": 0,
        "total_turns": 0,
        "phase_reasons": {},
        "messages": [],
    }
    seen = []
    for i in range(3):
        state = run_phase(state, Phase.HISTORY, llm=llm)
        seen.append(llm.prompts)
        state["messages"] = state["messages"] + [HumanMessage(content=f"question {i}")]

    # each turn sees everything said before it, and the earlier turns are still there
    assert [len(s) for s in seen] == sorted(len(s) for s in seen)
    assert "question 0" in seen[2]
    assert "question 1" in seen[2]
    # the third prompt predates question 2, since it was built before it was asked
    assert "question 2" not in seen[2]
