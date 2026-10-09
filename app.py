"""Dental OSCE simulator.

Run it with:  streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from dental.case_gen import DIFFICULTIES, generate_case  # noqa: E402
from dental.grader import band_for, grade  # noqa: E402
from dental.llm import NoProviderConfigured, get_llm  # noqa: E402
from dental.graph import get_graph  # noqa: E402
from dental.phase_runner import PHASE_GOALS  # noqa: E402
from dental.state import Phase, LANGUAGE_LABELS, UI_TEXT  # noqa: E402

st.set_page_config(page_title="Dental OSCE", layout="wide")

PHASE_LABELS = {phase: phase.value.replace("_", " ").title() for phase in Phase}


def _t(key: str) -> str:
    """Get translated text for current language."""
    lang = st.session_state.get("language", "en")
    return UI_TEXT.get(lang, UI_TEXT["en"]).get(key, UI_TEXT["en"].get(key, key))


def _reset(bundle, difficulty: str) -> None:
    st.session_state.case = bundle.patient.model_dump()
    st.session_state.rubric = [r.model_dump() for r in bundle.rubric]
    st.session_state.messages = []
    st.session_state.phase = Phase.COMPLAINT.value
    st.session_state["phase_reasons"] = {}
    st.session_state["turns_in_phase"] = 0
    st.session_state["total_turns"] = 0
    st.session_state["debrief"] = None
    st.session_state["revealed"] = set()
    st.session_state.difficulty = difficulty
    # Language persists across encounters unless explicitly changed
    if "language" not in st.session_state:
        st.session_state.language = "en"
    st.session_state.thread_id = f"enc-{len(st.session_state.get('all_threads', []))}"
    st.session_state.all_threads = st.session_state.get("all_threads", []) + [
        st.session_state.thread_id
    ]


def _render_sidebar() -> None:
    st.sidebar.title(_t("station"))
    difficulty = st.sidebar.selectbox(
        _t("difficulty"), DIFFICULTIES, index=DIFFICULTIES.index("mid")
    )

    language = st.sidebar.selectbox(
        _t("language"),
        list(LANGUAGE_LABELS.keys()),
        index=list(LANGUAGE_LABELS.keys()).index(st.session_state.get("language", "en")),
        format_func=lambda k: LANGUAGE_LABELS[k],
    )
    if language != st.session_state.get("language", "en"):
        st.session_state.language = language

    if st.sidebar.button(_t("start_encounter"), type="primary"):
        try:
            with st.spinner(_t("generating_patient")):
                bundle = generate_case(difficulty=difficulty)
            _reset(bundle, difficulty)
            st.rerun()
        except NoProviderConfigured as exc:
            st.sidebar.error(str(exc))
        except Exception as exc:
            st.sidebar.error(f"Could not generate a case: {exc}")

    st.sidebar.divider()
    st.sidebar.subheader(_t("patient_card"))
    case = st.session_state.get("case")
    if not case:
        st.sidebar.caption(_t("no_patient"))
        return

    st.sidebar.markdown(f"**{case.get('age', '?')}-year-old {case.get('sex', '')}**")
    st.sidebar.caption(f"“{case.get('complaint', '')}”")

    for label in ("medical_history", "drug_history", "allergies"):
        values = case.get(label) or []
        if values:
            st.sidebar.markdown(f"**{_t(label)}**")
            for value in values:
                st.sidebar.markdown(f"- {value}")

    hidden = case.get("hidden_findings") or []
    revealed = st.session_state.get("revealed", set())
    unrevealed = [
        fact for i, fact in enumerate(hidden) if fact not in revealed and i not in revealed
    ]
    st.sidebar.divider()
    if unrevealed:
        st.sidebar.caption(_t("uncovered_count").format(count=len(unrevealed)))
    else:
        st.sidebar.success(_t("all_uncovered"))


def _note_revealed_findings() -> None:
    """Record which hidden findings have surfaced in the transcript.

    The card shows what the learner actually uncovered, not what the generator knew —
    otherwise the debrief is theatre.
    """
    case = st.session_state.get("case", {})
    revealed = set(st.session_state.get("revealed", set()))
    transcript = " ".join(
        str(getattr(m, "content", m)) for m in st.session_state.get("messages", [])
    ).lower()
    for index, fact in enumerate(case.get("hidden_findings") or []):
        # Crude keyword overlap, deliberately: the card is a record, not a grader.
        # The grader makes the judgement calls, not this.
        words = [w for w in fact.lower().split() if len(w) > 4]
        if words and sum(w in transcript for w in words) >= max(1, len(words) // 2):
            revealed.add(index)
            revealed.add(fact)
    st.session_state["revealed"] = revealed


def _advance_once() -> None:
    """One graph invocation: the phase in state plays a single turn.

    The graph is a one-turn-per-invocation machine. `phase` and the turn counters
    live in session state and are carried into the call, so each learner message
    advances exactly one phase and nothing replays.
    """
    graph = get_graph()
    state = {
        "messages": st.session_state["messages"],
        "phase": st.session_state["phase"],
        "patient": st.session_state["case"],
        "rubric": st.session_state["rubric"],
        "turns_in_phase": st.session_state["turns_in_phase"],
        "total_turns": st.session_state["total_turns"],
        "phase_reasons": st.session_state["phase_reasons"],
        "debrief": st.session_state["debrief"],
        "language": st.session_state.get("language", "en"),
        "llm": get_llm(),
    }
    config = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "recursion_limit": 100,
    }
    out = graph.invoke(state, config=config)
    st.session_state["messages"] = out["messages"]
    st.session_state["phase"] = out["phase"]
    st.session_state["phase_reasons"] = out.get("phase_reasons", {})
    st.session_state["turns_in_phase"] = out.get("turns_in_phase", 0)
    st.session_state["total_turns"] = out.get("total_turns", 0)
    st.session_state["debrief"] = out.get("debrief")


def _render_chat() -> None:
    st.subheader(_t("encounter"))
    phase = Phase(st.session_state.get("phase", Phase.COMPLAINT.value))
    if phase is not Phase.DEBRIEF:
        st.caption(f"{_t('phase')}: **{PHASE_LABELS[phase]}** — {PHASE_GOALS[phase]}")

    for message in st.session_state.get("messages", []):
        role = getattr(message, "type", "user")
        with st.chat_message("user" if role == "human" else "assistant"):
            st.write(getattr(message, "content", message))

    if phase is Phase.DEBRIEF:
        return

    if not st.session_state.get("case"):
        st.info(_t("start_info"))
        return

    # The encounter ends when a phase advances to DEBRIEF. Grading is its own
    # invocation, kicked off here rather than by the learner typing again.
    if phase is Phase.DEBRIEF and not st.session_state.get("debrief"):
        try:
            _advance_once()
        except Exception as exc:
            st.error(f"Grading failed: {exc}")
            st.caption("Your transcript is kept. Use Re-grade below to retry.")
        else:
            st.rerun()
        return

    prompt = st.chat_input(_t("speak_to_patient"))
    if not prompt:
        return

    from langchain_core.messages import HumanMessage

    st.session_state["messages"].append(HumanMessage(content=prompt))
    st.session_state["total_turns"] += 1
    _note_revealed_findings()

    try:
        _advance_once()
    except NoProviderConfigured as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(_t("encounter_stopped").format(error=exc))
        st.caption(_t("messages_kept"))

    st.rerun()


def _render_debrief() -> None:
    debrief = st.session_state.get("debrief")
    if not debrief:
        return

    st.subheader(_t("debrief"))
    overall = debrief.get("overall", 0)
    st.metric(_t("overall"), f"{overall}%", band_for(overall))

    st.markdown(f"**{_t('per_phase')}**")
    for score in debrief.get("phase_scores", []):
        label = PHASE_LABELS.get(Phase(score["phase"]), score["phase"])
        st.progress(
            score["score"] / score["max_score"] if score["max_score"] else 0.0,
            text=f"{label}: {score['score']}/{score['max_score']}",
        )
        if score.get("reasoning"):
            st.caption(score["reasoning"])

    if debrief.get("focus_areas"):
        st.markdown(f"**{_t('work_on')}**")
        for area in debrief["focus_areas"]:
            st.markdown(f"- {area}")

    if debrief.get("strengths"):
        st.markdown(f"**{_t('strengths')}**")
        for strength in debrief["strengths"]:
            st.markdown(f"- {strength}")

    if st.button(_t("regrade")):
        try:
            with st.spinner(_t("grading_again")):
                fresh = grade(
                    case=st.session_state["case"],
                    rubric=st.session_state["rubric"],
                    transcript=st.session_state["messages"],
                    phase_reasons=st.session_state["phase_reasons"],
                    language=st.session_state.get("language", "en"),
                )
            st.session_state["debrief"] = fresh.model_dump()
            st.rerun()
        except Exception as exc:
            st.error(_t("regrade_failed").format(error=exc))


def main() -> None:
    st.title("Dental OSCE")
    st.caption(
        "A practice patient who will not volunteer anything you do not ask for, "
        "and will not correct you until the debrief."
    )

    for key, default in (
        ("case", None),
        ("rubric", []),
        ("messages", []),
        ("phase", Phase.COMPLAINT.value),
        ("phase_reasons", {}),
        ("turns_in_phase", 0),
        ("total_turns", 0),
        ("debrief", None),
        ("revealed", set()),
        ("all_threads", []),
        ("thread_id", "enc-0"),
    ):
        if key not in st.session_state:
            st.session_state[key] = default

    left, right = st.columns([2, 1])
    with left:
        _render_chat()
        _render_debrief()
    with right:
        _render_sidebar()

    st.divider()
    st.caption(_t("not_for_clinical"))


if __name__ == "__main__":
    main()
