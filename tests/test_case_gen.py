import pytest

from dental.case_gen import DIFFICULTIES, generate_case
from dental.schemas import CaseBundle, PatientCase, PhaseRubric


def _bundle() -> CaseBundle:
    return CaseBundle(
        patient=PatientCase(
            age=54,
            sex="female",
            complaint="My tooth is killing me.",
            medical_history=["type 2 diabetes"],
            drug_history=["metformin"],
            allergies=["penicillin"],
            hidden_findings=["swelling in left buccal sulcus"],
            intended_diagnosis="acute apical abscess 36",
            differentials=["symptomatic irreversible pulpitis", "cracked tooth syndrome"],
            investigation_results=["periapical radiograph shows radiolucency at 36 apex"],
        ),
        rubric=[PhaseRubric(phase="history", criteria=["asked about drug allergies"], max_score=4)],
    )


def _scripted(bundle: CaseBundle | None = None) -> "FakeLLM":
    from conftest import FakeLLM

    return FakeLLM(structured_response=bundle or _bundle())


def test_generate_case_returns_case_and_rubric_together():
    bundle = generate_case(difficulty="senior", llm=_scripted())
    assert bundle.patient.intended_diagnosis == "acute apical abscess 36"
    assert bundle.rubric[0].phase == "history"
    assert bundle.rubric[0].criteria == ["asked about drug allergies"]


def test_generate_case_rejects_unknown_difficulty():
    with pytest.raises(ValueError):
        generate_case(difficulty="expert", llm=_scripted())


def test_difficulties_are_exactly_three_values():
    assert DIFFICULTIES == ("junior", "mid", "senior")


def test_default_difficulty_is_mid():
    llm = _scripted()
    generate_case(llm=llm)
    assert "mid" in llm.calls[0][1]


def test_rubric_is_generated_from_the_case_not_generic_dentistry():
    """The rubric must be built from this patient, not from generic dentistry."""
    llm = _scripted()
    generate_case(llm=llm)
    system_prompt = llm.calls[0][0].lower()
    assert "rubric" in system_prompt
    assert "patient" in system_prompt
    assert "criteria" in system_prompt
