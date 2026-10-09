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
from dental.state import Phase, LANGUAGE_LABELS  # noqa: E402

st.set_page_config(page_title="Dental OSCE", layout="wide")

PHASE_LABELS = {phase: phase.value.replace("_", " ").title() for phase in Phase}


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
    st.sidebar.title("Station")
    difficulty = st.sidebar.selectbox(
        "Difficulty", DIFFICULTIES, index=DIFFICULTIES.index("mid")
    )

    language = st.sidebar.selectbox(
        "Language",
        list(LANGUAGE_LABELS.keys()),
        index=list(LANGUAGE_LABELS.keys()).index(st.session_state.get("language", "en")),
        format_func=lambda k: LANGUAGE_LABELS[k],
    )
    if language != st.session_state.get("language", "en"):
        st.session_state.language = language

    if st.sidebar.button("Start new encounter", type="primary"):
        try:
            with st.spinner("Generating a patient…"):
                bundle = generate_case(difficulty=difficulty)
            _reset(bundle, difficulty)
            st.rerun()
        except NoProviderConfigured as exc:
            st.sidebar.error(str(exc))
        except Exception as exc:  # provider/network/schema failure
            st.sidebar.error(f"Could not generate a case: {exc}")

    st.sidebar.divider()
    st.sidebar.subheader("Patient card")
    case = st.session_state.get("case")
    if not case:
        st.sidebar.caption("No patient yet. Start an encounter.")
        return

    st.sidebar.markdown(f"**{case.get('age', '?')}-year-old {case.get('sex', '')}**")
    st.sidebar.caption(f"“{case.get('complaint', '')}”")

    for label in ("medical_history", "drug_history", "allergies"):
        values = case.get(label) or []
        if values:
            st.sidebar.markdown(f"**{label.replace('_', ' ').title()}**")
            for value in values:
                st.sidebar.markdown(f"- {value}")

    hidden = case.get("hidden_findings") or []
    revealed = st.session_state.get("revealed", set())
    unrevealed = [
        fact for i, fact in enumerate(hidden) if fact not in revealed and i not in revealed
    ]
    st.sidebar.divider()
    if unrevealed:
        st.sidebar.caption(f"{len(unrevealed)} finding(s) not yet uncovered")
    else:
        st.sidebar.success("All findings uncovered")


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
    st.subheader("Encounter")
    phase = Phase(st.session_state.get("phase", Phase.COMPLAINT.value))
    if phase is not Phase.DEBRIEF:
        st.caption(f"Phase: **{PHASE_LABELS[phase]}** — {PHASE_GOALS[phase]}")

    for message in st.session_state.get("messages", []):
        role = getattr(message, "type", "user")
        with st.chat_message("user" if role == "human" else "assistant"):
            st.write(getattr(message, "content", message))

    if phase is Phase.DEBRIEF:
        return

    if not st.session_state.get("case"):
        st.info("Start an encounter from the sidebar.")
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

    prompt = st.chat_input("Speak to the patient…")
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
        st.error(f"The encounter stopped: {exc}")
        st.caption("Your messages are kept. Try again to continue.")

    st.rerun()


def _render_debrief() -> None:
    debrief = st.session_state.get("debrief")
    if not debrief:
        return

    st.subheader("Debrief")
    overall = debrief.get("overall", 0)
    st.metric("Overall", f"{overall}%", band_for(overall))

    st.markdown("**Per phase**")
    for score in debrief.get("phase_scores", []):
        label = PHASE_LABELS.get(Phase(score["phase"]), score["phase"])
        st.progress(
            score["score"] / score["max_score"] if score["max_score"] else 0.0,
            text=f"{label}: {score['score']}/{score['max_score']}",
        )
        if score.get("reasoning"):
            st.caption(score["reasoning"])

    if debrief.get("focus_areas"):
        st.markdown("**Work on these**")
        for area in debrief["focus_areas"]:
            st.markdown(f"- {area}")

    if debrief.get("strengths"):
        st.markdown("**What you did well**")
        for strength in debrief["strengths"]:
            st.markdown(f"- {strength}")

    if st.button("Re-grade"):
        try:
            with st.spinner("Grading again…"):
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
            st.error(f"Re-grade failed: {exc}")


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
    st.caption("Teaching tool. Not for clinical use or patient advice.")


if __name__ == "__main__":
    main()
