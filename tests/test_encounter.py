"""The two failure modes that only appear across a whole encounter.

Neither is visible inside a single node: a learner who asks nothing looks efficient,
and a patient who volunteers a finding looks like a thorough candidate.
"""

from dental.graph import get_graph
from dental.state import Phase
from conftest import FakeLLM, FakeResponse


def _debrief(score, max_score, covered, missed):
    return FakeResponse(
        structured_response={
            "phase_scores": [
                {
                    "phase": "history",
                    "score": score,
                    "max_score": max_score,
                    "covered": covered,
                    "missed": missed,
                    "reasoning": "minimal history taken",
                }
            ],
            "overall": score,
            "band": "borderline",
            "focus_areas": ["take a full history"],
            "strengths": [],
        }
    )


class _Encounter(FakeLLM):
    """Completes every phase in one turn, then hands back a scripted debrief."""

    def __init__(self, debrief):
        super().__init__(
            structured_response={
                "patient_reply": "My tooth hurts.",
                "phase_complete": True,
                "phase_reason": "",
            },
            debrief_response=debrief,
        )

    def __call__(self, system_prompt="", human_prompt=""):
        if "examiner" in system_prompt.lower():
            return self.debrief_response
        return super().__call__(system_prompt, human_prompt)


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


def _run(llm):
    """Drive a whole encounter one learner turn per invoke, as the app does."""
    graph = get_graph()
    state = {
        "messages": [],
        "phase": Phase.COMPLAINT.value,
        "patient": {"age": 54, "complaint": "My tooth hurts."},
        "rubric": [{"phase": "history", "criteria": ["allergies"], "max_score": 8}],
        "turns_in_phase": 0,
        "total_turns": 0,
        "debrief": None,
        "phase_reasons": {},
        "llm": llm,
    }
    config = {"configurable": {"thread_id": "enc1"}, "recursion_limit": 100}

    # seven phases, each completing in one turn, then one invocation to grade
    for _ in range(8):
        out = graph.invoke(state, config=config)
        state = {key: out[key] for key in STATE_KEYS}
    return state


def test_learner_who_asks_nothing_is_not_rewarded():
    """An encounter with one learner line must not read as an efficient one."""
    llm = _Encounter(
        _debrief(0, 8, covered=[], missed=["history of complaint", "allergies", "drug history"])
    )
    out = _run(llm)
    assert out["debrief"]["overall"] == 0
    assert out["debrief"]["band"] == "needs another session"
    assert out["debrief"]["focus_areas"] == ["take a full history"]
    assert out["debrief"]["phase_scores"][0]["missed"]


def test_learner_who_asks_nothing_still_gets_an_actionable_debrief():
    """No crash and no empty panel — the learner gets something to act on."""
    out = _run(_Encounter(_debrief(0, 8, [], ["allergies"])))
    assert out["phase"] == Phase.DEBRIEF.value
    debrief = out["debrief"]
    assert debrief["phase_scores"][0]["reasoning"] == "minimal history taken"
    assert debrief["phase_scores"][0]["missed"] == ["allergies"]
    assert debrief["focus_areas"]


def test_volunteered_finding_gets_no_credit():
    """The patient handing over a finding is not the learner earning it."""
    from dental.grader import grade

    llm = FakeLLM(
        structured_response={
            "phase_scores": [],
            "overall": 0,
            "band": "needs another session",
            "focus_areas": [],
            "strengths": [],
        }
    )
    grade(
        case={"age": 54, "hidden_findings": ["spontaneous bleeding of the gingiva"]},
        rubric=[{"phase": "history", "criteria": ["asked about bleeding gums"]}],
        transcript=[{"role": "assistant", "content": "My gums bleed all the time actually."}],
        llm=llm,
    )
    lowered = llm.prompts.lower()
    assert "unasked" in lowered
    assert "award no credit" in lowered


def test_full_encounter_reaches_a_debrief_with_per_phase_scores():
    out = _run(_Encounter(_debrief(6, 8, ["chief complaint"], ["drug allergies"])))
    assert out["phase"] == Phase.DEBRIEF.value
    assert out["debrief"]["phase_scores"][0]["max_score"] == 8
    assert out["debrief"]["overall"] == 75  # 6 of 8, not the 6 the model reported
