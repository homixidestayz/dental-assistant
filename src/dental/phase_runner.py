"""One encounter turn: the patient replies and decides whether the phase is done.

Every phase node delegates here, so the pacing rule, the message bookkeeping, and
the turn ceiling exist once rather than seven times.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from dental.llm import get_llm
from dental.model_call import invoke_structured
from dental.schemas import PhaseTurn
from dental.state import CEILING_REASON, TURN_CEILING, Phase, next_phase


class MalformedModelOutput(RuntimeError):
    """The model did not return a usable phase turn, even after one retry.

    Raised rather than defaulted, because a missing phase_complete flag left as
    False would loop the encounter until the ceiling on every single run.
    """


PHASE_GOALS: dict[Phase, str] = {
    Phase.COMPLAINT: (
        "the patient opening with why they are here, in their own words. "
        "Let them describe it; do not lead them."
    ),
    Phase.HISTORY: (
        "history of the complaint, then medical history, drug history, allergies, "
        "dental history, and social and habit history."
    ),
    Phase.EXAMINATION: (
        "extraoral and intraoral examination, and soft tissue assessment. The "
        "candidate describes what they want to look at; you report only what is "
        "there."
    ),
    Phase.INVESTIGATIONS: (
        "the investigations the candidate orders — radiographs, pulp sensibility "
        "tests, palpation. Report results only for tests they actually order."
    ),
    Phase.DIAGNOSIS: (
        "the candidate telling you what they think is going on and what else it "
        "might be. A patient does not know, so respond as one who is worried."
    ),
    Phase.MANAGEMENT: (
        "the treatment plan, consent, risks, and what it will cost. Patients ask "
        "the questions a patient would ask."
    ),
    Phase.CLOSING: (
        "answering the patient's remaining questions, safety-netting, and "
        "follow-up. Then the encounter is over."
    ),
}

_CONSTRAINTS = """Rules you must never break:
- Never volunteer a clinical finding the candidate did not ask for. If they did not
  ask about something, it does not come up.
- Never suggest an investigation, a diagnosis, or a treatment. You are the patient,
  not the clinician.
- Never break character to comment on the candidate's performance. No coaching, no
  hints, no encouragement, no grading.
- Never repeat yourself. Everything you have already said is in the conversation so
  far. Do not restate your complaint, do not re-answer a question they have already
  asked, and do not offer information you have already given. Move the conversation
  forward instead.
"""


def build_patient_prompt(patient: dict, phase: Phase) -> str:
    """The patient persona, the phase goal, and the standing constraints."""
    age = patient.get("age", "an adult")
    sex = patient.get("sex", "")
    complaint = patient.get("complaint", "a dental problem")
    hidden = patient.get("hidden_findings", []) or []

    lines = [
        f"You are a {age}-year-old {sex} patient at a dental appointment. "
        f"You came in because: {complaint}",
        "",
        "You know the following about yourself, but you will only say any of it if "
        "the candidate asks you directly, in your own words, not as clinical "
        "findings:",
    ]
    if hidden:
        lines.extend(f"- {fact}" for fact in hidden)
    else:
        lines.append("- (nothing hidden)")
    lines += [
        "",
        f"Right now the candidate is working on: {PHASE_GOALS[phase]}",
        "",
        _CONSTRAINTS,
        "Reply as this patient would: a few sentences at most, in the language a "
        "patient uses, not clinical terminology.",
        "",
        "After your reply, decide whether this phase is complete — that is, "
        "whether the candidate has covered everything this stage needs. Set "
        "phase_complete true only if they genuinely have. Give a short reason.",
    ]
    return "\n".join(lines)


def _transcript_text(messages: list) -> str:
    """The conversation so far, labelled by speaker.

    Without this the model sees only the patient details and answers in a vacuum,
    re-asking and re-answering the same questions every turn.
    """
    lines = []
    for message in messages or []:
        content = getattr(message, "content", "")
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            lines.append(f"CANDIDATE: {content}")
        else:
            lines.append(f"PATIENT: {content}")
    return "\n".join(lines)


def _turn_prompt(patient: dict, phase: Phase, state: dict) -> tuple[str, str]:
    transcript = _transcript_text(state.get("messages", []))
    return (
        build_patient_prompt(patient, phase),
        f"Turn {state.get('total_turns', 0) + 1}. "
        f"Turns taken in this phase: {state.get('turns_in_phase', 0)}."
        + (f"\n\nCONVERSATION SO FAR\n{transcript}" if transcript else ""),
    )


def _call_model(llm: Any, schema: type, system: str, human: str) -> Any:
    return invoke_structured(llm, schema, system, human)


def run_phase(state: dict, phase: Phase, llm: Any = None) -> dict:
    """Play one turn of `phase`. Returns updated state with the phase possibly advanced."""
    if llm is None:
        llm = state.get("llm") or get_llm()

    system, human = _turn_prompt(state.get("patient", {}), phase, state)

    try:
        turn = _call_model(llm, PhaseTurn, system, human)
    except (ValidationError, ValueError):
        # One retry with the output contract restated, then give up loudly.
        strict = system + (
            "\n\nYour previous reply could not be parsed. Return patient_reply "
            "(string), phase_complete (boolean), and phase_reason (string)."
        )
        try:
            turn = _call_model(llm, PhaseTurn, strict, human)
        except (ValidationError, ValueError) as exc:
            raise MalformedModelOutput(
                f"Model failed to return a valid phase turn for phase {phase.value!r} "
                f"after one retry: {exc}"
            ) from exc

    if not isinstance(turn, PhaseTurn):
        turn = PhaseTurn(**turn) if isinstance(turn, dict) else turn

    turns_in_phase = state.get("turns_in_phase", 0) + 1
    total_turns = state.get("total_turns", 0) + 1
    ceiling_hit = turns_in_phase >= TURN_CEILING

    reason = CEILING_REASON if ceiling_hit else turn.phase_reason
    advance = turn.phase_complete or ceiling_hit
    new_phase = next_phase(phase) if advance else phase

    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=turn.patient_reply))

    phase_reasons = dict(state.get("phase_reasons", {}))
    phase_reasons[phase.value] = reason

    return {
        **state,
        "messages": messages,
        "phase": new_phase.value,
        "phase_complete": bool(turn.phase_complete),
        "phase_reason": reason,
        "phase_reasons": phase_reasons,
        "turns_in_phase": 0 if advance else turns_in_phase,
        "total_turns": total_turns,
    }


def should_advance(state: dict) -> str:
    """Route to the phase the graph should visit next.

    `run_phase` has already advanced `state["phase"]` by the time this runs, so this
    is a router, not a second decision: the destination is whatever phase the node
    left in state. Advancing here as well would skip every other phase.

    DEBRIEF routes to itself, and the graph's edge from it goes to END.
    """
    current = Phase(state["phase"])
    if current is Phase.DEBRIEF:
        return Phase.DEBRIEF.value
    # Defensive only: run_phase advances at the ceiling itself. This keeps routing
    # correct if a caller ever populates the counters directly.
    if (
        state.get("turns_in_phase", 0) >= TURN_CEILING
        and not state.get("phase_complete")
    ):
        return next_phase(current).value
    return current.value
