"""Reach the debrief with a realistic candidate."""
from dental.case_gen import generate_case
from dental.graph import get_graph
from dental.llm import get_llm
from dental.state import Phase
from langchain_core.messages import HumanMessage

KEYS=("messages","phase","patient","rubric","turns_in_phase","total_turns","phase_reasons","debrief","llm")

bundle = generate_case(difficulty="mid")
p = bundle.patient
print(f"CASE: {p.age} {p.sex} - {p.complaint}")
print(f"  dx: {p.intended_diagnosis}")

llm = get_llm(); g = get_graph()
state = {"messages": [], "phase": Phase.COMPLAINT.value,
         "patient": p.model_dump(),
         "rubric": [r.model_dump() for r in bundle.rubric],
         "turns_in_phase": 0, "total_turns": 0, "phase_reasons": {},
         "debrief": None, "llm": llm}
cfg = {"configurable": {"thread_id": "live-2"}, "recursion_limit": 100}

# A candidate who takes a decent history but never examines or investigates.
LEARNER = [
 "Hello, what can I do for you?",
 "Sorry to hear that. Can you tell me more about the pain?",
 "How would you describe it - is it sharp or dull? Does it keep you up at night?",
 "Is it triggered by anything specific, like cold or chewing?",
 "When did it start, and has it changed since?",
 "Have you had any treatment for this before?",
 "Are you taking any medications at the moment?",
 "Do you have any allergies I should know about?",
 "How is your general health - any diabetes, heart conditions, bleeding problems?",
 "Do you smoke, and how often do you brush?",
 "Right, I'd like to examine you now. Any discomfort when I press here?",
 "Open wide for me. Does that hurt?",
 "Let me take a radiograph and test the nerve response.",
 "Based on that, I think this is cracked tooth syndrome with early nerve involvement.",
 "Here's what I'd recommend - an onlay to strengthen the tooth, watching for nerve recovery.",
 "The risks are that the nerve may not recover and we'd need root canal treatment later.",
 "That would be around X for the onlay, and X if we end up needing the root canal.",
 "Do you have any questions for me?",
 "We'll review in six weeks. If the pain gets worse, come back sooner.",
 "That's everything, thank you.",
]

for i in range(30):
    if state["phase"] == Phase.DEBRIEF.value and state["debrief"]:
        break
    add = state["phase"] != Phase.DEBRIEF.value
    if add:
        state["messages"] = list(state["messages"]) + [HumanMessage(content=LEARNER[min(i, len(LEARNER)-1)])]
    out = g.invoke(dict(state), config=cfg)
    state = {k: out[k] for k in KEYS}

print(f"turns={state['total_turns']} phase={state['phase']}")
d = state["debrief"]
print()
print(f"DEBRIEF: {d['overall']}% - {d['band']}")
for s in d["phase_scores"]:
    print(f"  {s['phase']:<15} {s['score']}/{s['max_score']}  missed={s['missed'][:2]}")
print(f"  focus areas:")
for f in d["focus_areas"]: print(f"    - {f}")
print(f"  strengths:")
for s in d["strengths"]: print(f"    - {s}")
