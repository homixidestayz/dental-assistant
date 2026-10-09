"""Encounter state: the clinical phase enum and the graph's state shape."""

from enum import Enum
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class Phase(str, Enum):
    COMPLAINT = "complaint"
    HISTORY = "history"
    EXAMINATION = "examination"
    INVESTIGATIONS = "investigations"
    DIAGNOSIS = "diagnosis"
    MANAGEMENT = "management"
    CLOSING = "closing"
    DEBRIEF = "debrief"


PHASE_ORDER: list[Phase] = [
    Phase.COMPLAINT,
    Phase.HISTORY,
    Phase.EXAMINATION,
    Phase.INVESTIGATIONS,
    Phase.DIAGNOSIS,
    Phase.MANAGEMENT,
    Phase.CLOSING,
]

# A model that decides its phase goals are never met must not hang the encounter.
# A real history of complaint is 4 to 6 turns, so this only fires on a stuck run.
TURN_CEILING: int = 8

CEILING_REASON: str = "turn ceiling reached"

# Supported encounter languages. The whole encounter runs in this language.
LANGUAGES: tuple[str, ...] = ("en", "fr", "ar")

# Display names for the UI.
LANGUAGE_LABELS: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "ar": "Arabic",
}

# UI text translations for the Streamlit interface.
UI_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "station": "Station",
        "difficulty": "Difficulty",
        "language": "Language",
        "start_encounter": "Start new encounter",
        "generating_patient": "Generating a patient…",
        "patient_card": "Patient card",
        "no_patient": "No patient yet. Start an encounter.",
        "medical_history": "Medical History",
        "drug_history": "Drug History",
        "allergies": "Allergies",
        "uncovered_count": "{count} finding(s) not yet uncovered",
        "all_uncovered": "All findings uncovered",
        "encounter": "Encounter",
        "phase": "Phase",
        "speak_to_patient": "Speak to the patient…",
        "start_info": "Start an encounter from the sidebar.",
        "debrief": "Debrief",
        "overall": "Overall",
        "per_phase": "Per phase",
        "work_on": "Work on these",
        "strengths": "What you did well",
        "regrade": "Re-grade",
        "grading_again": "Grading again…",
        "regrade_failed": "Re-grade failed: {error}",
        "encounter_stopped": "The encounter stopped: {error}",
        "messages_kept": "Your messages are kept. Try again to continue.",
        "not_for_clinical": "Teaching tool. Not for clinical use or patient advice.",
    },
    "fr": {
        "station": "Station",
        "difficulty": "Difficulté",
        "language": "Langue",
        "start_encounter": "Nouvel entretien",
        "generating_patient": "Génération d'un patient…",
        "patient_card": "Fiche patient",
        "no_patient": "Aucun patient pour l'instant. Commencez un entretien.",
        "medical_history": "Antécédents médicaux",
        "drug_history": "Traitements en cours",
        "allergies": "Allergies",
        "uncovered_count": "{count} élément(s) non encore découvert(s)",
        "all_uncovered": "Tous les éléments découverts",
        "encounter": "Entretien",
        "phase": "Phase",
        "speak_to_patient": "Parlez au patient…",
        "start_info": "Commencez un entretien depuis la barre latérale.",
        "debrief": "Débriefing",
        "overall": "Global",
        "per_phase": "Par phase",
        "work_on": "À travailler",
        "strengths": "Points forts",
        "regrade": "Re-noter",
        "grading_again": "Re-notation en cours…",
        "regrade_failed": "Re-notation échouée : {error}",
        "encounter_stopped": "L'entretien s'est arrêté : {error}",
        "messages_kept": "Vos messages sont conservés. Réessayez pour continuer.",
        "not_for_clinical": "Outil pédagogique. Pas pour un usage clinique ou conseil patient.",
    },
    "ar": {
        "station": "المحطة",
        "difficulty": "الصعوبة",
        "language": "اللغة",
        "start_encounter": "بدء مقابلة جديدة",
        "generating_patient": "جاري إنشاء مريض…",
        "patient_card": "بطاقة المريض",
        "no_patient": "لا يوجد مريض بعد. ابدأ مقابلة.",
        "medical_history": "السجل الطبي",
        "drug_history": "الأدوية الحالية",
        "allergies": "الحساسية",
        "uncovered_count": "{count} نتيجة لم تكتشف بعد",
        "all_uncovered": "تم اكتشاف جميع النتائج",
        "encounter": "المقابلة",
        "phase": "المرحلة",
        "speak_to_patient": "تحدث مع المريض…",
        "start_info": "ابدأ مقابلة من الشريط الجانبي.",
        "debrief": "التقرير",
        "overall": "الإجمالي",
        "per_phase": "لكل مرحلة",
        "work_on": "للعمل عليه",
        "strengths": "نقاط القوة",
        "regrade": "إعادة التقييم",
        "grading_again": "جاري إعادة التقييم…",
        "regrade_failed": "فشل إعادة التقييم: {error}",
        "encounter_stopped": "توقفت المقابلة: {error}",
        "messages_kept": "رسائلك محفوظة. حاول مرة أخرى للمتابعة.",
        "not_for_clinical": "أداة تعليمية. ليست للاستخدام السريري أو نصيحة المريض.",
    },
}


class DentalState(TypedDict, total=False):
    """Everything the encounter carries from turn to turn.

    `llm` rides in state rather than a module global so tests substitute a fake
    without monkeypatching, and so the Streamlit app can supply one model per
    session. It is an injected dependency, not encounter data.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    phase: str
    patient: dict
    rubric: list
    phase_complete: bool
    phase_reason: str
    phase_reasons: dict
    turns_in_phase: int
    total_turns: int
    debrief: Any
    llm: Any
    language: str


def next_phase(phase: Phase) -> Phase:
    """The phase after `phase`. Raises on DEBRIEF, which has no successor."""
    if phase is Phase.DEBRIEF:
        raise ValueError("debrief is terminal and has no next phase")
    index = PHASE_ORDER.index(phase)
    if index + 1 >= len(PHASE_ORDER):
        return Phase.DEBRIEF
    return PHASE_ORDER[index + 1]


def initial_state() -> DentalState:
    return DentalState(
        messages=[],
        phase=Phase.COMPLAINT.value,
        patient={},
        rubric=[],
        phase_complete=False,
        phase_reason="",
        phase_reasons={},
        turns_in_phase=0,
        total_turns=0,
        debrief=None,
        language="en",
    )
