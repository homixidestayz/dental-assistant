import pytest

from dental.grader import band_for, grade
from dental.schemas import Debrief, PhaseScore
from conftest import FakeLLM


def _debrief(score=5, max_score=7, covered=None, missed=None, overall=95, band="excellent"):
    return {
        "phase_scores": [
            {
                "phase": "history",
                "score": score,
                "max_score": max_score,
                "covered": covered if covered is not None else ["chief complaint"],
                "missed": missed if missed is not None else ["drug allergies"],
                "reasoning": "took a partial history",
            }
        ],
        "overall": overall,
        "band": band,
        "focus_areas": ["ask about drug allergies"],
        "strengths": ["clear opening"],
    }


def test_band_thresholds():
    assert band_for(95) == "excellent"
    assert band_for(80) == "excellent"
    assert band_for(79) == "competent"
    assert band_for(65) == "competent"
    assert band_for(64) == "borderline"
    assert band_for(50) == "borderline"
    assert band_for(49) == "needs another session"


def test_grade_requires_rubric():
    with pytest.raises(ValueError):
        grade(case={"age": 54}, rubric=[], transcript=[], llm=FakeLLM())


def test_band_is_computed_in_python_not_taken_from_the_model():
    """The scripted model claims excellent; 5 of 7 points is competent."""
    llm = FakeLLM(structured_response=_debrief(score=5, max_score=7))
    result = grade(case={"age": 54}, rubric=[{"phase": "history"}], transcript=[], llm=llm)
    assert result.overall == 71  # round(100 * 5 / 7)
    assert result.band == "competent"


def test_overall_is_the_percentage_of_points_earned():
    llm = FakeLLM(
        structured_response={
            "phase_scores": [
                {"phase": "history", "score": 3, "max_score": 4, "covered": [], "missed": [], "reasoning": ""},
                {"phase": "examination", "score": 4, "max_score": 6, "covered": [], "missed": [], "reasoning": ""},
            ],
            "overall": 0,
            "band": "nonsense",
            "focus_areas": [],
            "strengths": [],
        }
    )
    result = grade(case={"age": 1}, rubric=[{"phase": "history"}], transcript=[], llm=llm)
    assert result.overall == 70  # 7 of 10 points
    assert result.band == "competent"


def test_phase_reasons_reach_the_model():
    """A phase that ran long must be graded as run-long, not as complete."""
    llm = FakeLLM(structured_response=_debrief())
    grade(
        case={"age": 54},
        rubric=[{"phase": "history"}],
        transcript=[{"role": "user", "content": "hi"}],
        phase_reasons={"history": "turn ceiling reached"},
        llm=llm,
    )
    assert "turn ceiling reached" in llm.prompts


def test_grader_is_told_not_to_credit_unasked_findings():
    llm = FakeLLM(structured_response=_debrief())
    grade(
        case={"age": 54, "hidden_findings": ["spontaneous bleeding of the gingiva"]},
        rubric=[{"phase": "history", "criteria": ["asked about bleeding gums"]}],
        transcript=[{"role": "assistant", "content": "My gums bleed all the time actually."}],
        llm=llm,
    )
    lowered = llm.prompts.lower()
    assert "unasked" in lowered
    assert "award no credit" in lowered


def test_grader_is_told_to_score_completeness_separately_from_accuracy():
    llm = FakeLLM(structured_response=_debrief())
    grade(case={"age": 54}, rubric=[{"phase": "history"}], transcript=[], llm=llm)
    lowered = llm.prompts.lower()
    assert "completeness" in lowered
    assert "accuracy" in lowered


def test_grade_is_reusable_without_replaying():
    """Re-grade is the reason the grader is its own module."""
    llm = FakeLLM(structured_response=_debrief(score=5, max_score=7))
    first = grade(case={"age": 1}, rubric=[{"phase": "history"}], transcript=[], llm=llm)
    second = grade(case={"age": 1}, rubric=[{"phase": "history"}], transcript=[], llm=llm)
    assert first.band == second.band == "competent"
    assert len(llm.calls) == 2  # two independent calls, no state carried


def test_transcript_is_passed_to_the_model():
    llm = FakeLLM(structured_response=_debrief())
    grade(
        case={"age": 54},
        rubric=[{"phase": "history"}],
        transcript=[{"role": "user", "content": "any pain?"}, {"role": "assistant", "content": "yes a lot"}],
        llm=llm,
    )
    assert "any pain?" in llm.prompts


def test_debrief_shape():
    llm = FakeLLM(structured_response=_debrief())
    result = grade(case={"age": 54}, rubric=[{"phase": "history"}], transcript=[], llm=llm)
    assert isinstance(result, Debrief)
    assert isinstance(result.phase_scores[0], PhaseScore)
    assert result.focus_areas == ["ask about drug allergies"]


def test_grade_handles_empty_transcript_without_crashing():
    llm = FakeLLM(structured_response=_debrief(score=0, max_score=8))
    result = grade(case={"age": 54}, rubric=[{"phase": "history"}], transcript=[], llm=llm)
    assert result.overall == 0
    assert result.band == "needs another session"


def test_debrief_node_puts_the_debrief_in_state():
    from dental.nodes.debrief import node_debrief

    llm = FakeLLM(structured_response=_debrief())
    state = {
        "patient": {"age": 54, "complaint": "my tooth hurts"},
        "rubric": [{"phase": "history", "criteria": ["allergies"], "max_score": 8}],
        "messages": [{"role": "user", "content": "hi"}],
        "phase_reasons": {},
        "llm": llm,
    }
    out = node_debrief(state, llm=llm)
    assert out["debrief"]["band"] == "competent"
    assert out["phase"] == "debrief"
