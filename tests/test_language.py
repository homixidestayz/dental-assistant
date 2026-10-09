"""Language support: the encounter runs in the chosen language."""

from dental.state import Phase, initial_state, LANGUAGES, LANGUAGE_LABELS
from dental.phase_runner import run_phase, build_patient_prompt
from dental.grader import grade


def test_initial_state_includes_language():
    state = initial_state()
    assert "language" in state
    assert state["language"] == "en"


def test_language_enum():
    assert LANGUAGES == ("en", "fr", "ar")


def test_language_labels():
    assert LANGUAGE_LABELS == {"en": "English", "fr": "French", "ar": "Arabic"}


def test_language_persists_across_turns():
    state = initial_state()
    state["language"] = "fr"
    assert state["language"] == "fr"


def test_build_patient_prompt_includes_language_instruction():
    """The prompt should tell the model what language to reply in."""
    patient = {"age": 54, "complaint": "my tooth hurts"}
    prompt = build_patient_prompt(patient, Phase.HISTORY, language="fr")
    assert "french" in prompt.lower()


def test_run_phase_accepts_language_and_passes_it_to_prompt():
    from conftest import FakeLLM

    class CapturingLLM(FakeLLM):
        def __call__(self, system_prompt="", human_prompt=""):
            self.calls.append((system_prompt, human_prompt))
            return super().__call__(system_prompt, human_prompt)

    llm = CapturingLLM(
        structured_response={
            "patient_reply": "ok",
            "phase_complete": True,
            "phase_reason": "done",
        }
    )
    state = {
        "patient": {"age": 54, "complaint": "my tooth hurts"},
        "phase": Phase.HISTORY.value,
        "turns_in_phase": 0,
        "total_turns": 0,
        "phase_reasons": {},
        "messages": [],
        "language": "ar",
    }
    run_phase(state, Phase.HISTORY, llm=llm)
    # The system prompt should mention Arabic
    system_prompt = llm.calls[0][0]
    assert "arabic" in system_prompt.lower()


def test_grade_accepts_language_and_writes_debrief_in_that_language():
    """The grader should be told to write the debrief in the chosen language."""
    from conftest import FakeLLM

    class CapturingLLM(FakeLLM):
        def __call__(self, system_prompt="", human_prompt=""):
            if "examiner" in system_prompt.lower():
                self.calls.append((system_prompt, human_prompt))
                from conftest import FakeResponse
                return FakeResponse(
                    structured_response={
                        "phase_scores": [
                            {
                                "phase": "history",
                                "score": 5,
                                "max_score": 7,
                                "covered": ["chief complaint"],
                                "missed": ["allergies"],
                                "reasoning": "partial history",
                            }
                        ],
                        "overall": 71,
                        "band": "competent",
                        "focus_areas": ["ask about allergies"],
                        "strengths": ["clear opening"],
                    }
                )
            return super().__call__(system_prompt, human_prompt)

    llm = CapturingLLM()
    grade(
        case={"age": 54, "complaint": "toothache"},
        rubric=[{"phase": "history", "criteria": ["allergies"], "max_score": 7}],
        transcript=[{"role": "user", "content": "hi"}],
        language="fr",
        llm=llm,
    )
    # The grader human prompt should mention French
    grader_human_prompts = [c[1] for c in llm.calls if "examiner" in c[0].lower()]
    assert any("french" in h.lower() for h in grader_human_prompts)