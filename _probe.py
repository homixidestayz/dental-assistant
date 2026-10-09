from dental.grader import grade
from tests.conftest import FakeLLM, FakeResponse

class CapturingLLM(FakeLLM):
    def __call__(self, system_prompt="", human_prompt=""):
        if "examiner" in system_prompt.lower():
            self.calls.append((system_prompt, human_prompt))
            return FakeResponse(
                structured_response={
                    "phase_scores": [{"phase": "history", "score": 5, "max_score": 7,
                                      "covered": ["chief complaint"], "missed": ["allergies"],
                                      "reasoning": "partial history"}],
                    "overall": 71, "band": "competent",
                    "focus_areas": ["ask about allergies"], "strengths": ["clear opening"]})
        return super().__call__(system_prompt, human_prompt)

llm = CapturingLLM()
grade(
    case={"age": 54, "complaint": "toothache"},
    rubric=[{"phase": "history", "criteria": ["allergies"], "max_score": 7}],
    transcript=[{"role": "user", "content": "hi"}],
    language="fr",
    llm=llm,
)

print("Calls captured:", len(llm.calls))
for i, (s, h) in enumerate(llm.calls):
    print(f"Call {i}: system contains 'french' = {'french' in s.lower()}")
    if "examiner" in s.lower():
        print("EXAMINER CALL:")
        print(s[:500])
        break
