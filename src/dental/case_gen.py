"""Case and rubric generation: one structured call per encounter."""

from typing import Any

from dental.model_call import invoke_structured
from dental.schemas import CaseBundle

DIFFICULTIES: tuple[str, ...] = ("junior", "mid", "senior")

_SYSTEM = """You generate patient cases for a dental OSCE examination simulator.

Scope: the full breadth of dentistry — caries, pulp and periapical pathology,
periodontal disease, trauma, oral surgery, prosthodontics, orthodontics, paediatric
dentistry, oral medicine. Choose whatever case best fits the requested difficulty.

For the `patient` field:
- `complaint` is the patient's own words, not clinical language. A patient says
  "my tooth is killing me", not "severe spontaneous pain".
- `hidden_findings` holds the clinical facts the patient knows but will NOT reveal
  unless the candidate asks for them directly. This is what makes history-taking
  worth doing.

For the `rubric` field:
- One entry per clinical phase: complaint, history, examination, investigations,
  diagnosis, management, closing.
- `criteria` must name THIS patient's specifics — this patient's drug history, this
  patient's likely investigation, this patient's hidden findings. Never generic
  dentistry. A criterion like "took a full history" is useless; "asked about
  penicillin allergy, which is relevant to this patient's presentation" is checkable.
- `max_score` is the points available for that phase.

A judge will score a trainee against this rubric later. It is the answer key, so it
must be complete enough to score against and specific enough to be fair.
"""

_HUMAN = """Generate one patient case at {difficulty} difficulty.

Difficulty shapes case complexity and how much is hidden:
- junior: a common presentation, most facts volunteered easily, one clear diagnosis.
- mid: some facts require direct questioning, plausible differentials to separate.
- senior: atypical or mixed presentation, significant hidden history that changes
  management, several differentials that must be actively excluded.

Write the entire case in {lang_name}. The complaint must be in the patient's own words
in {lang_name}, not clinical language. All history fields, hidden findings, diagnosis,
differentials, and investigation results must be in {lang_name}.
"""


def _invoke_structured(llm: Any, schema: type, system: str, human: str) -> Any:
    return invoke_structured(llm, schema, system, human)


def generate_case(difficulty: str = "mid", language: str = "en", llm: Any = None) -> CaseBundle:
    """Generate a patient case and its grading rubric together.

    `difficulty` is one of DIFFICULTIES and feeds generation only — not the rubric,
    not the patient's behaviour during the encounter.
    """
    if difficulty not in DIFFICULTIES:
        raise ValueError(
            f"difficulty must be one of {DIFFICULTIES}, got {difficulty!r}"
        )
    from dental.state import LANGUAGE_LABELS
    lang_name = LANGUAGE_LABELS.get(language, "English")
    return _invoke_structured(
        llm,
        CaseBundle,
        _SYSTEM,
        _HUMAN.format(difficulty=difficulty, lang_name=lang_name),
    )
