"""DentOSCE simulator.

Run it with:  streamlit run app.py
"""

import asyncio
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit Community Cloud installs from requirements.txt only and does not
# install this package, so make the local `src/` tree importable here. Harmless
# when the package is already installed.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Community Cloud delivers secrets through st.secrets; the provider cascade reads
# os.environ. Mirror the provider keys across when present, and do nothing locally.
try:
    _secret_keys = (
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "GROQ_MODEL",
        "GEMINI_MODEL",
        "OPENROUTER_MODEL",
        "OPENAI_MODEL",
    )
    for _secret_key in _secret_keys:
        if _secret_key in st.secrets and not os.getenv(_secret_key):
            os.environ[_secret_key] = str(st.secrets[_secret_key])
except Exception:
    pass

from dental.case_gen import DIFFICULTIES, generate_case  # noqa: E402
from dental.grader import band_for, grade  # noqa: E402
from dental.llm import NoProviderConfigured  # noqa: E402
from dental.graph import get_graph, stream_graph  # noqa: E402
from dental.phase_runner import PHASE_GOALS  # noqa: E402
from dental.state import Phase, LANGUAGE_LABELS, UI_TEXT  # noqa: E402

st.set_page_config(page_title="DentOSCE", layout="wide", page_icon="🦷")

# Light, high-contrast clinic theme. Deliberately minimal: no forced text
# colours or dark backgrounds that could make content unreadable.
st.markdown(
    """
<style>
    .block-container, [data-testid="stMainBlockContainer"] {
        padding-top: 2rem;
        max-width: 1080px;
        margin: 0 auto;
    }

    /* Long words or pasted strings must never create a sideways scrollbar. */
    .stApp, .stMarkdown, [data-testid="stChatMessage"], [data-testid="stMetricValue"] {
        overflow-wrap: anywhere;
    }
    [data-testid="stMetric"] { min-width: 0; }

    /* The phone copies of the setup/patient panel: hidden on desktop. */
    .st-key-mobile_panel_setup, .st-key-mobile_panel_card { display: none; }

    .clinic-header {
        background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 100%);
        padding: 1.25rem 1.75rem;
        border-radius: 12px;
        margin-bottom: 1.25rem;
    }
    .clinic-header h1 {
        color: #ffffff;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
    }
    .clinic-header p {
        color: #bae6fd;
        font-size: 0.95rem;
        margin: 0.35rem 0 0 0;
    }

    .patient-card-item {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        margin: 0.3rem 0;
        border: 1px solid #e2e8f0;
    }
    .patient-card-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .patient-card-value {
        font-size: 0.85rem;
        color: #0f172a;
    }

    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.5rem 0.75rem;
    }
    [data-testid="stChatMessageAvatarUser"] { background: #e0f2fe; }
    [data-testid="stChatMessageAvatarAssistant"] { background: #fef3c7; }

    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }

    div[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    @media (max-width: 640px) {
        .block-container, [data-testid="stMainBlockContainer"] {
            padding: 0.75rem 0.75rem 4rem;
        }
        .clinic-header { padding: 0.9rem 1rem; border-radius: 10px; }
        .clinic-header h1 { font-size: 1.3rem; }
        .clinic-header p { font-size: 0.82rem; }

        /* No sidebar, and no toggle to open one, on a phone. */
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"],
        [data-testid="stExpandSidebarButton"] {
            display: none !important;
        }

        .st-key-mobile_panel_setup, .st-key-mobile_panel_card { display: block; }
        [data-testid="stChatMessage"] { padding: 0.35rem 0.5rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)

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


def _on_language_change(changed_key: str) -> None:
    """Mirror a language change to the other selector and the whole UI.

    The selector exists twice (desktop sidebar and phone panel) with different
    widget keys, so a change in one has to be copied to the other or the two
    would drift and fight over `st.session_state.language`.
    """
    chosen = st.session_state.get(changed_key, st.session_state.get("language", "en"))
    st.session_state["language"] = chosen
    other = "language_mobile" if changed_key == "language_desktop" else "language_desktop"
    st.session_state[other] = chosen


def _encounter_setup(container, key_prefix: str) -> None:
    """Difficulty, language, and the start button.

    Rendered into the sidebar on desktop and into the main column on phones, so
    it takes a container and a key prefix to keep the two copies' widget IDs
    distinct (both copies share the same labels).
    """
    difficulty = container.selectbox(
        _t("difficulty"),
        DIFFICULTIES,
        key=f"difficulty_{key_prefix}",
    )
    container.selectbox(
        _t("language"),
        list(LANGUAGE_LABELS.keys()),
        format_func=lambda k: LANGUAGE_LABELS[k],
        key=f"language_{key_prefix}",
        on_change=_on_language_change,
        args=(f"language_{key_prefix}",),
    )

    if container.button(
        _t("start_encounter"), type="primary", key=f"start_{key_prefix}"
    ):
        try:
            with st.spinner(_t("generating_patient")):
                bundle = generate_case(
                    difficulty=difficulty,
                    language=st.session_state.get("language", "en"),
                )
            _reset(bundle, difficulty)
            st.rerun()
        except NoProviderConfigured as exc:
            container.error(str(exc))
        except Exception as exc:
            container.error(f"Could not generate a case: {exc}")


def _patient_card(container) -> None:
    """The patient's known history and how much the learner has uncovered.

    Container-agnostic so the desktop sidebar and the phone panel can share it.
    Pure text, so rendering it twice creates no duplicate widget IDs.
    """
    case = st.session_state.get("case")
    if not case:
        container.caption(_t("no_patient"))
        return

    container.markdown(f"**{case.get('age', '?')}-year-old {case.get('sex', '')}**")
    container.caption(f"“{case.get('complaint', '')}”")

    for label in ("medical_history", "drug_history", "allergies"):
        values = case.get(label) or []
        if values:
            container.markdown(
                f'<div class="patient-card-item">'
                f'<div class="patient-card-label">{_t(label)}</div>'
                f'<div class="patient-card-value">{" • ".join(values)}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    hidden = case.get("hidden_findings") or []
    revealed = st.session_state.get("revealed", set())
    unrevealed = [
        fact for i, fact in enumerate(hidden) if fact not in revealed and i not in revealed
    ]
    container.divider()
    if unrevealed:
        container.caption(_t("uncovered_count").format(count=len(unrevealed)))
    else:
        container.success(_t("all_uncovered"))


def _render_sidebar() -> None:
    """The desktop sidebar: station title, setup controls, and the patient card."""
    st.sidebar.title("🦷 " + _t("station"))
    _encounter_setup(st.sidebar, "desktop")
    st.sidebar.divider()
    st.sidebar.subheader(_t("patient_card"))
    _patient_card(st.sidebar)


def _render_mobile_panel() -> None:
    """Phones only: the same setup and patient card inline above the chat.

    Under 640px the sidebar and its toggle are hidden by CSS, so these keyed
    containers (hidden on desktop, shown on phones) keep every control reachable.
    """
    with st.container(key="mobile_panel_setup"):
        with st.expander(
            _t("start_encounter"), expanded=not st.session_state.get("case")
        ):
            _encounter_setup(st, "mobile")
    with st.container(key="mobile_panel_card"):
        with st.expander(
            _t("patient_card"), expanded=bool(st.session_state.get("case"))
        ):
            _patient_card(st)


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
        "llm": None,
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
        # The encounter ended on the previous turn. Grading is its own graph
        # invocation (the debrief node), run here exactly once, then rerun so the
        # debrief panel can render.
        if not st.session_state.get("debrief"):
            with st.spinner(_t("grading")):
                try:
                    _advance_once()
                except NoProviderConfigured as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(_t("grade_failed").format(error=exc))
                    st.caption(_t("messages_kept"))
                else:
                    st.rerun()
        return

    if not st.session_state.get("case"):
        st.info(_t("start_info"))
        return

    prompt = st.chat_input(_t("speak_to_patient"))
    if not prompt:
        return

    from langchain_core.messages import HumanMessage

    # Add the learner's message immediately so it renders on this pass.
    user_msg = HumanMessage(content=prompt)
    st.session_state["messages"].append(user_msg)
    st.session_state["total_turns"] += 1
    _note_revealed_findings()

    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    # Stream the patient's response.
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
            "llm": None,
        }
        config = {
            "configurable": {"thread_id": st.session_state["thread_id"]},
            "recursion_limit": 100,
        }

        with st.chat_message("assistant", avatar="🦷"):
            placeholder = st.empty()
            placeholder.markdown("🦷 *typing...*")

            final_state = None
            for chunk in graph.stream(state, config=config, stream_mode="updates"):
                for _node_name, node_output in chunk.items():
                    # Capture the latest state from each node update.
                    if node_output:
                        final_state = {**state, **node_output}

            # Use the final state from streaming (no second invoke).
            if final_state is None:
                final_state = graph.invoke(state, config=config)

            # Show the patient's newest reply without the cursor.
            ai_messages = [
                m
                for m in (final_state or {}).get("messages", [])
                if getattr(m, "type", None) == "ai"
            ]
            if ai_messages:
                placeholder.markdown(ai_messages[-1].content)

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
        '<div class="clinic-header">'
        "<h1>🦷 DentOSCE</h1>"
        "<p>A practice patient who will not volunteer anything you do not ask for, "
        "and will not correct you until the debrief.</p>"
        "</div>",
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

    # Seed the duplicated language/difficulty selectors before they render, so a
    # keyed widget never needs a default value it would then have to override.
    for selector in ("language_desktop", "language_mobile"):
        if selector not in st.session_state:
            st.session_state[selector] = st.session_state.get("language", "en")
    for selector in ("difficulty_desktop", "difficulty_mobile"):
        if selector not in st.session_state:
            st.session_state[selector] = "mid"

    _render_sidebar()
    _render_mobile_panel()
    _render_chat()
    _render_debrief()

    st.divider()
    st.caption(_t("not_for_clinical"))


if __name__ == "__main__":
    main()
