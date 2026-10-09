"""Dental OSCE simulator.

Run it with:  streamlit run app.py
"""

import asyncio
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from dental.case_gen import DIFFICULTIES, generate_case  # noqa: E402
from dental.grader import band_for, grade  # noqa: E402
from dental.llm import NoProviderConfigured, get_llm  # noqa: E402
from dental.graph import get_graph, stream_graph  # noqa: E402
from dental.phase_runner import PHASE_GOALS  # noqa: E402
from dental.state import Phase, LANGUAGE_LABELS, UI_TEXT  # noqa: E402

st.set_page_config(page_title="Dental OSCE", layout="wide", page_icon="🦷")

# Dental clinic theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --dental-navy: #0d1b2a;
        --dental-teal: #00b4d8;
        --dental-teal-dark: #0096c7;
        --dental-teal-light: #caf0f8;
        --dental-white: #ffffff;
        --dental-gray-50: #f8fafc;
        --dental-gray-100: #f1f5f9;
        --dental-gray-200: #e2e8f0;
        --dental-gray-300: #cbd5e1;
        --dental-gray-600: #475569;
        --dental-gray-900: #0f172a;
        --dental-red: #ef4444;
        --dental-green: #22c55e;
        --dental-amber: #f59e0b;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--dental-gray-50) 0%, var(--dental-teal-light) 100%);
        font-family: 'Inter', sans-serif;
    }
    
    section[data-testid="stSidebar"] {
        background: var(--dental-white);
        border-right: 1px solid var(--dental-gray-200);
        box-shadow: 4px 0 20px rgba(13, 27, 42, 0.08);
    }
    
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }
    
    .dental-header {
        background: linear-gradient(135deg, var(--dental-navy) 0%, var(--dental-teal-dark) 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(13, 27, 42, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .dental-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 200%;
        background: radial-gradient(circle, rgba(0, 180, 216, 0.15) 0%, transparent 70%);
        animation: pulse 8s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .dental-header h1 {
        color: var(--dental-white);
        font-weight: 700;
        font-size: 2.25rem;
        margin: 0;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    
    .dental-header .subtitle {
        color: var(--dental-teal-light);
        font-weight: 400;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .phase-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: var(--dental-white);
        border-radius: 50px;
        box-shadow: 0 2px 12px rgba(13, 27, 42, 0.08);
        border: 1px solid var(--dental-gray-200);
        margin-bottom: 1rem;
    }
    
    .phase-badge .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--dental-teal);
        animation: blink 2s ease-in-out infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    
    .phase-badge .text {
        font-weight: 600;
        color: var(--dental-navy);
        font-size: 0.95rem;
    }
    
    [data-testid="stChatMessage"] {
        background: var(--dental-white);
        border-radius: 16px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 12px rgba(13, 27, 42, 0.06);
        border: 1px solid var(--dental-gray-100);
        transition: all 0.2s ease;
    }
    
    [data-testid="stChatMessage"]:hover {
        box-shadow: 0 4px 20px rgba(13, 27, 42, 0.1);
        border-color: var(--dental-gray-200);
    }
    
    [data-testid="stChatMessage"][aria-label="user"] {
        border-left: 4px solid var(--dental-teal);
    }
    
    [data-testid="stChatMessage"][aria-label="assistant"] {
        border-left: 4px solid var(--dental-amber);
    }
    
    .stChatInput textarea {
        border-radius: 12px !important;
        border: 2px solid var(--dental-gray-200) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        padding: 0.875rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stChatInput textarea:focus {
        border-color: var(--dental-teal) !important;
        box-shadow: 0 0 0 4px var(--dental-teal-light) !important;
    }
    
    [data-testid="stMetric"] {
        background: var(--dental-white);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid var(--dental-gray-200);
        box-shadow: 0 2px 12px rgba(13, 27, 42, 0.05);
        transition: all 0.2s ease;
    }
    
    [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 20px rgba(13, 27, 42, 0.1);
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--dental-gray-600) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--dental-navy) !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--dental-teal) 0%, var(--dental-teal-dark) 100%) !important;
        border-radius: 8px !important;
        height: 10px !important;
    }
    
    .stProgress > div > div {
        background: var(--dental-gray-100) !important;
        border-radius: 8px !important;
        height: 10px !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--dental-teal) 0%, var(--dental-teal-dark) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 16px rgba(0, 180, 216, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 180, 216, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    .stButton > button[kind="secondary"] {
        background: var(--dental-gray-100);
        color: var(--dental-navy);
        box-shadow: none;
        border: 1px solid var(--dental-gray-200);
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: var(--dental-gray-200);
        box-shadow: none;
    }
    
    .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 2px solid var(--dental-gray-200) !important;
        transition: all 0.2s ease !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--dental-teal) !important;
        box-shadow: 0 0 0 4px var(--dental-teal-light) !important;
    }
    
    .stSpinner > div {
        border-top-color: var(--dental-teal) !important;
    }
    
    hr {
        border-color: var(--dental-gray-200) !important;
        margin: 1.5rem 0 !important;
    }
    
    .streamlit-expanderHeader {
        background: var(--dental-white);
        border-radius: 12px;
        border: 1px solid var(--dental-gray-200);
        font-weight: 600;
        color: var(--dental-navy);
    }
    
    .streamlit-expanderContent {
        background: var(--dental-gray-50);
        border-radius: 0 0 12px 12px;
        border: 1px solid var(--dental-gray-200);
        border-top: none;
    }
    
    .stSuccess, .stError, .stInfo, .stWarning {
        border-radius: 12px !important;
        border: none !important;
        padding: 1rem 1.25rem !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%) !important;
        border-left: 4px solid var(--dental-green) !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%) !important;
        border-left: 4px solid var(--dental-red) !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(0, 180, 216, 0.1) 0%, rgba(0, 180, 216, 0.05) 100%) !important;
        border-left: 4px solid var(--dental-teal) !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%) !important;
        border-left: 4px solid var(--dental-amber) !important;
    }
    
    .patient-card-item {
        background: var(--dental-gray-50);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border: 1px solid var(--dental-gray-200);
    }
    
    .patient-card-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--dental-gray-600);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    
    .patient-card-value {
        font-size: 0.9rem;
        color: var(--dental-navy);
        font-weight: 500;
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--dental-gray-600);
        font-style: italic;
        font-size: 0.9rem;
    }
    
    .typing-indicator .dots {
        display: flex;
        gap: 3px;
    }
    
    .typing-indicator .dots span {
        width: 6px;
        height: 6px;
        background: var(--dental-teal);
        border-radius: 50%;
        animation: typing 1.4s ease-in-out infinite both;
    }
    
    .typing-indicator .dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-indicator .dots span:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
        40% { transform: scale(1.2); opacity: 1; }
    }
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--dental-gray-100);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--dental-gray-300);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--dental-gray-400);
    }
    
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
        background: var(--dental-teal-light) !important;
        border: 2px solid var(--dental-teal) !important;
    }
    
    [data-testid="stChatMessage"][aria-label="assistant"] [data-testid="stChatMessageAvatar"] {
        background: #fff3e0 !important;
        border: 2px solid var(--dental-amber) !important;
    }
</style>
""", unsafe_allow_html=True)

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
                bundle = generate_case(difficulty=difficulty, language=st.session_state.get("language", "en"))
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
            st.sidebar.markdown(
                f'<div class="patient-card-item">'
                f'<div class="patient-card-label">{_t(label)}</div>'
                f'<div class="patient-card-value">{" • ".join(values)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

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

    # Render existing messages
    for message in st.session_state.get("messages", []):
        role = getattr(message, "type", "user")
        avatar = "👤" if role == "human" else "🦷"
        with st.chat_message("user" if role == "human" else "assistant", avatar=avatar):
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

    from langchain_core.messages import HumanMessage, AIMessage

    # Add user message immediately
    user_msg = HumanMessage(content=prompt)
    st.session_state["messages"].append(user_msg)
    st.session_state["total_turns"] += 1
    _note_revealed_findings()

    # Show user message immediately
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    # Stream the patient's response
    try:
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

        # Show typing indicator and stream response
        with st.chat_message("assistant", avatar="🦷"):
            placeholder = st.empty()
            placeholder.markdown("🦷  *typing...*")

            # Stream the response and collect the final state
            with st.chat_message("assistant", avatar="🦷"):
                placeholder = st.empty()
                placeholder.markdown("🦷  *typing...*")

                final_state = None
                for chunk in graph.stream(state, config=config, stream_mode="updates"):
                    for node_name, node_output in chunk.items():
                        if node_name == "messages" and node_output:
                            for msg in node_output:
                                if hasattr(msg, "type") and msg.type == "ai":
                                    placeholder.markdown(msg.content + "▌")
                        # Capture the final state from the last update
                        if node_output:
                            final_state = {**state, **node_output}

                # Use the final state from streaming (no second invoke)
                if final_state is None:
                    # Fallback: single invoke if streaming yielded nothing
                    out = graph.invoke(state, config=config)
                    final_state = out

                # Final message without cursor
                if final_state and final_state.get("messages"):
                    for msg in final_state["messages"]:
                        if hasattr(msg, "type") and msg.type == "ai":
                            placeholder.markdown(msg.content)
                            break

                # Update session state from the streamed final state
                st.session_state["messages"] = final_state["messages"]
                st.session_state["phase"] = final_state["phase"]
                st.session_state["phase_reasons"] = final_state.get("phase_reasons", {})
                st.session_state["turns_in_phase"] = final_state.get("turns_in_phase", 0)
                st.session_state["total_turns"] = final_state.get("total_turns", 0)
                st.session_state["debrief"] = final_state.get("debrief")

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
    st.markdown(
        '<div class="dental-header">'
        '<h1>🦷 Dental OSCE</h1>'
        '<p class="subtitle">A practice patient who will not volunteer anything you do not ask for, '
        'and will not correct you until the debrief.</p>'
        '</div>',
        unsafe_allow_html=True,
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