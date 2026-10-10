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


class FlakyModel:
    """Fails the first N structured calls with a given error, then succeeds.

    Shaped like a real chat model: not callable, has `with_structured_output`.
    """

    def __init__(self, value, failures=1, message="tool_use_failed: did not match schema"):
        self.value = value
        self.failures = failures
        self.message = message
        self.attempts = 0
        self.structured_calls = []

    def with_structured_output(self, schema):
        outer = self

        class Bound:
            def invoke(self, messages):
                outer.structured_calls.append(schema)
                outer.attempts += 1
                if outer.attempts <= outer.failures:
                    raise RuntimeError(outer.message)
                return outer.value

        return Bound()


def _turn():
    from dental.schemas import PhaseTurn

    return PhaseTurn(patient_reply="ok", phase_complete=True, phase_reason="done")


def test_structured_call_retries_a_provider_tool_call_failure():
    from dental.model_call import invoke_structured
    from dental.schemas import PhaseTurn

    model = FlakyModel(_turn(), failures=1)
    out = invoke_structured(model, PhaseTurn, "sys", "human")
    assert out.patient_reply == "ok"
    assert model.attempts == 2


def test_structured_call_gives_up_after_bounded_attempts():
    from dental.model_call import invoke_structured
    from dental.schemas import PhaseTurn

    model = FlakyModel(_turn(), failures=99, message="tool_use_failed")
    with pytest.raises(RuntimeError):
        invoke_structured(model, PhaseTurn, "sys", "human")
    assert model.attempts == 3


def test_structured_call_does_not_retry_an_unrelated_error():
    """A bad API key must fail immediately, not be hammered three times."""
    from dental.model_call import invoke_structured
    from dental.schemas import PhaseTurn

    model = FlakyModel(_turn(), failures=99, message="401 invalid api key")
    with pytest.raises(RuntimeError):
        invoke_structured(model, PhaseTurn, "sys", "human")
    assert model.attempts == 1


def test_structured_call_falls_back_to_the_next_provider(monkeypatch):
    """With no model pinned, a broken provider hands off to the next one."""
    import dental.llm as llm_mod
    from dental.model_call import invoke_structured
    from dental.schemas import PhaseTurn

    broken = FlakyModel(_turn(), failures=99, message="tool_use_failed")
    healthy = FlakyModel(_turn(), failures=0)
    monkeypatch.setattr(llm_mod, "available_llms", lambda: [broken, healthy])

    out = invoke_structured(None, PhaseTurn, "sys", "human")
    assert out.patient_reply == "ok"
    assert broken.attempts == 3  # exhausted its own retries before handing off
    assert healthy.attempts == 1


def test_structured_call_raises_when_every_provider_fails(monkeypatch):
    import dental.llm as llm_mod
    from dental.model_call import invoke_structured
    from dental.schemas import PhaseTurn

    first = FlakyModel(_turn(), failures=99, message="tool_use_failed")
    second = FlakyModel(_turn(), failures=99, message="tool_use_failed")
    monkeypatch.setattr(llm_mod, "available_llms", lambda: [first, second])

    with pytest.raises(RuntimeError):
        invoke_structured(None, PhaseTurn, "sys", "human")
    assert first.attempts == 3
    assert second.attempts == 3


def test_an_explicit_model_is_not_swapped_for_a_provider(monkeypatch):
    """A pinned model must never silently become a different provider."""
    import dental.llm as llm_mod
    from dental.model_call import invoke_structured
    from dental.schemas import PhaseTurn

    pinned = FlakyModel(_turn(), failures=99, message="tool_use_failed")

    def _unexpected():
        raise AssertionError("cascade must not be consulted for an explicit model")

    monkeypatch.setattr(llm_mod, "available_llms", _unexpected)
    with pytest.raises(RuntimeError):
        invoke_structured(pinned, PhaseTurn, "sys", "human")
    assert pinned.attempts == 3
