"""The intake node: an empty state generates its own patient.

LangGraph Studio and other direct graph consumers start from an empty input, so
the encounter has to make its own case rather than rely on the app to supply one.
"""

from dental.nodes.intake import node_intake
from conftest import FakeLLM


def _case_response():
    return {
        "patient": {
            "age": 42,
            "sex": "female",
            "complaint": "my tooth hurts",
            "medical_history": [],
            "drug_history": [],
            "allergies": [],
            "hidden_findings": ["penicillin allergy"],
            "intended_diagnosis": "irreversible pulpitis",
            "differentials": ["reversible pulpitis"],
            "investigation_results": ["percussion tender"],
        },
        "rubric": [
            {"phase": "history", "criteria": ["asked about allergies"], "max_score": 7}
        ],
    }


def test_intake_generates_a_patient_and_rubric_for_an_empty_state():
    llm = FakeLLM(structured_response=_case_response())
    out = node_intake({}, llm=llm)
    assert out["patient"]["age"] == 42
    assert out["patient"]["hidden_findings"] == ["penicillin allergy"]
    assert out["rubric"][0]["phase"] == "history"
    assert out["phase"] == "complaint"


def test_intake_is_a_noop_when_a_patient_already_exists():
    llm = FakeLLM(structured_response=_case_response())
    existing = {"patient": {"age": 30, "complaint": "x"}, "rubric": []}
    out = node_intake(existing, llm=llm)
    assert out == existing
    assert llm.calls == []


def test_intake_falls_back_to_mid_on_an_unknown_difficulty():
    llm = FakeLLM(structured_response=_case_response())
    out = node_intake({"difficulty": "impossible"}, llm=llm)
    assert out["patient"]["age"] == 42
