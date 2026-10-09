"""Grading: the debrief, as its own compiled subgraph.

Separate from the encounter so a debrief can be re-run against the same transcript
without replaying it. Grading is the slowest call in the app and the one users will
want to retry.

The rubric is the answer key, generated with the case before the learner spoke. A
grader meeting the case for the first time while grading is generous one run and
harsh the next.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from dental.llm import get_llm
from dental.model_call import invoke_structured
from dental.schemas import Debrief
from dental.state import LANGUAGE_LABELS

# Band thresholds live here, in code, so the model cannot invent them per run.
EXCELLENT = 80
COMPETENT = 65
BORDERLINE = 50

_SYSTEM = """You are the examiner for a dental OSCE station. Score a candidate's
encounter against the rubric that was written for this specific patient before they
began.

Rules:

- Score completeness separately from accuracy. A candidate who never asked about
  drug allergies has lost those points, and saying so is more useful to them than
  any comment about how they handled what they did ask.
- Award no credit for a finding the patient volunteered unasked. If the patient
  mentioned something the candidate did not ask for, it was the patient who
  uncovered it, not the candidate.
- A phase that ran to the turn ceiling was cut off. Score it as incomplete, and say
  so in the reasoning.
- Be specific. "Took a poor history" helps nobody. "Never asked whether the patient
  was taking warfarin, which changes the extraction plan" does.

For each phase in the rubric, return a score out of that phase's max_score, what was
covered, what was missed, and your assessment of their clinical reasoning.

Return exactly three focus areas, phrased as actions the candidate can take rather
than topics to revise. Name their strengths concretely, not as encouragement.
"""


def band_for(score: int) -> str:
    """The band for a 0-100 score. Explicit comparisons, not a model decision."""
    if score >= EXCELLENT:
        return "excellent"
    if score >= COMPETENT:
        return "competent"
    if score >= BORDERLINE:
        return "borderline"
    return "needs another session"


def _transcript_text(transcript: list) -> str:
    lines = []
    for entry in transcript or []:
        if isinstance(entry, dict):
            role = entry.get("role", "unknown")
            content = entry.get("content", "")
        else:
            role = getattr(entry, "type", "unknown")
            content = getattr(entry, "content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _call_model(llm: Any, schema: type, system: str, human: str) -> Any:
    return invoke_structured(llm, schema, system, human)


def _overall_and_band(raw: Debrief) -> Debrief:
    """Recompute the score and band in Python, whatever the model reported.

    A model asked for a percentage will happily average its per-phase scores its own
    way. Totalling the points is the only definition that cannot drift between runs.
    """
    earned = sum(ps.score for ps in raw.phase_scores)
    possible = sum(ps.max_score for ps in raw.phase_scores)
    overall = round(100 * earned / possible) if possible else 0
    return Debrief(
        phase_scores=raw.phase_scores,
        overall=overall,
        band=band_for(overall),
        focus_areas=raw.focus_areas,
        strengths=raw.strengths,
    )


def grade(
    case: dict,
    rubric: list,
    transcript: list,
    phase_reasons: dict | None = None,
    llm: Any = None,
    language: str = "en",
) -> Debrief:
    """Score an encounter. Callable directly — no graph run required.

    `rubric` is required: grading without the answer key the case was generated with
    would mean the grader reconstructs the correct answer under time pressure, which
    is exactly the inconsistency this design exists to avoid.
    """
    if not rubric:
        raise ValueError("grade requires the rubric generated with the case")

    if llm is None:
        llm = get_llm()

    reasons = phase_reasons or {}
    stalled = {phase: reason for phase, reason in reasons.items() if reason}
    stalled_note = (
        "\n".join(f"- {phase}: {reason}" for phase, reason in stalled.items())
        if stalled
        else "(none)"
    )

    lang_name = LANGUAGE_LABELS.get(language, "English")

    human = f"""PATIENT
{case}

RUBRIC
{rubric}

TRANSCRIPT
{_transcript_text(transcript)}

PHASE NOTES
Phases that did not end on their own terms:
{stalled_note}

Write the debrief in {lang_name}. All fields (band, focus_areas, strengths, reasoning)
must be in {lang_name}.
"""
    raw = _call_model(llm, Debrief, _SYSTEM, human)
    if not isinstance(raw, Debrief):
        raw = Debrief(**raw) if isinstance(raw, dict) else raw
    return _overall_and_band(raw)


# The same grading, as a compiled subgraph for callers that want it in a graph.
_grader_builder = StateGraph(dict)
_grader_builder.add_node("grade", lambda state: {"debrief": grade(**state)})
_grader_builder.add_edge(START, "grade")
_grader_builder.add_edge("grade", END)

graph: CompiledStateGraph = _grader_builder.compile()
