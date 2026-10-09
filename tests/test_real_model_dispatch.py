"""Real chat models are not callable; the fakes are.

Every module that talks to a model has to pick between `with_structured_output` (a
real LangChain model) and calling the object directly (the test double). Getting that
dispatch wrong is invisible to a fake-only suite, because the fake satisfies whichever
side it was written for.

This test uses a stand-in shaped like a real chat model: has `with_structured_output`,
is not callable.
"""

import pytest


class RealishResponse:
    def __init__(self, value):
        self.value = value


class RealishModel:
    """Shaped like a LangChain chat model: not callable, no `structured_response`."""

    def __init__(self, value):
        self.value = value
        self.structured_calls = []
        self.invoked = []

    def with_structured_output(self, schema):
        outer = self

        class Bound:
            def invoke(self, messages):
                outer.structured_calls.append(schema)
                outer.invoked.append(messages)
                return outer.value

        return Bound()


def test_case_gen_uses_with_structured_output_on_a_real_model():
    from dental.case_gen import generate_case
    from dental.schemas import CaseBundle

    bundle = CaseBundle(
        patient={
            "age": 40,
            "sex": "male",
            "complaint": "my jaw hurts",
            "medical_history": [],
            "drug_history": [],
            "allergies": [],
            "hidden_findings": [],
            "intended_diagnosis": "TMD",
            "differentials": [],
            "investigation_results": [],
        },
        rubric=[{"phase": "history", "criteria": ["x"], "max_score": 4}],
    )
    model = RealishModel(bundle)
    result = generate_case(difficulty="mid", llm=model)

    assert result.patient.intended_diagnosis == "TMD"
    assert model.structured_calls == [CaseBundle]


def test_phase_runner_uses_with_structured_output_on_a_real_model():
    from dental.phase_runner import run_phase
    from dental.schemas import PhaseTurn
    from dental.state import Phase

    model = RealishModel(
        PhaseTurn(patient_reply="it hurts", phase_complete=True, phase_reason="done")
    )
    state = {"patient": {}, "phase": Phase.HISTORY.value, "turns_in_phase": 0,
             "total_turns": 0, "messages": [], "phase_reasons": {}}
    out = run_phase(state, Phase.HISTORY, llm=model)

    assert out["phase"] == Phase.EXAMINATION.value
    assert model.structured_calls == [PhaseTurn]


def test_grader_uses_with_structured_output_on_a_real_model():
    from dental.grader import grade
    from dental.schemas import Debrief

    model = RealishModel(
        Debrief(
            phase_scores=[
                {"phase": "history", "score": 3, "max_score": 4,
                 "covered": [], "missed": [], "reasoning": ""}
            ],
            overall=0,
            band="nonsense",
            focus_areas=[],
            strengths=[],
        )
    )
    result = grade(case={"age": 40}, rubric=[{"phase": "history"}],
                   transcript=[], llm=model)

    assert result.overall == 75  # computed in python, not the model's 0
    assert model.structured_calls == [Debrief]


def test_a_non_callable_non_model_is_rejected_clearly():
    """Neither branch applies: fail loudly instead of a confusing TypeError."""
    from dental.case_gen import generate_case

    class Useless:
        pass

    with pytest.raises((TypeError, ValueError)):
        generate_case(difficulty="mid", llm=Useless())
