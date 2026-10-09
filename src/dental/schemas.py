"""Pydantic shapes for structured model output.

extra="forbid" on every model: a malformed generation should surface as a validation
error rather than silently carrying junk into the rubric the grader scores against.
"""

from pydantic import BaseModel, ConfigDict, Field


class PatientCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int
    sex: str
    complaint: str = Field(description="The chief complaint in the patient's own words.")
    medical_history: list[str]
    drug_history: list[str]
    allergies: list[str]
    hidden_findings: list[str] = Field(
        description="Clinical facts the patient knows but has not revealed unless asked."
    )
    intended_diagnosis: str
    differentials: list[str]
    investigation_results: list[str] = Field(
        description="What each investigation the candidate orders will show."
    )


class PhaseRubric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: str
    criteria: list[str] = Field(
        description="What a strong candidate covers in this phase, specific to this patient."
    )
    max_score: int


class CaseBundle(BaseModel):
    """A case and the rubric grading it, generated together in one call.

    Generating both at once is deliberate: the rubric is the grader's answer key,
    fixed before it sees how the learner behaved. A grader meeting the case while
    grading is generous one run and harsh the next.
    """

    model_config = ConfigDict(extra="forbid")

    patient: PatientCase
    rubric: list[PhaseRubric]


class PhaseTurn(BaseModel):
    """One encounter turn: what the patient says, and whether the phase is done."""

    model_config = ConfigDict(extra="forbid")

    patient_reply: str
    phase_complete: bool
    phase_reason: str = Field(description="Short reason for the pacing decision.")


class PhaseScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: str
    score: int
    max_score: int
    covered: list[str]
    missed: list[str]
    reasoning: str


class Debrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase_scores: list[PhaseScore]
    overall: int
    band: str
    focus_areas: list[str] = Field(description="Top three, phrased as actions not topics.")
    strengths: list[str]
