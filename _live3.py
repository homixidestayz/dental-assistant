"""A real end-to-end encounter against the live model."""
from dental.case_gen import generate_case
from dental.graph import get_graph
from dental.llm import get_llm
from dental.state import Phase
from langchain_core.messages import HumanMessage

KEYS=("messages","phase","patient","rubric","turns_in_phase","total_turns","phase_reasons","debrief","llm")

bundle = generate_case(difficulty="mid")
p = bundle.patient
print(f"CASE: {p.age} {p.sex} - {p.complaint}")
print(f"  hidden: {p.hidden_findings}")
print(f"  dx: {p.intended_diagnosis}")
print(f"  meds: {p.drug_history} | allergies: {p.allergies}")
print()

llm = get_llm()
g = get_graph()
state = {"messages": [], "phase": Phase.COMPLAINT.value,
         "patient": p.model_dump(),
         "rubric": [r.model_dump() for r in bundle.rubric],
         "turns_in_phase": 0, "total_turns": 0, "phase_reasons": {},
         "debrief": None, "llm": llm}
cfg = {"configurable": {"thread_id": "live-1"}, "recursion_limit": 100}

# A deliberately sloppy candidate: opens, asks almost nothing, jumps to a guess.
LEARNER = [
    "Hi, what seems to be the problem?",
    "Any medications?",
    "Right, I'll take a look.",
    "So what's the treatment?",
    "How much is that going to cost?",
    "Can I get a filling?",
    "OK, thanks.",
    "One more thing.",
    "Do I need to come back?",
    "That's all.",
]

for i in range(24):
    if state["phase"] == Phase.DEBRIEF.value:
        if state["debrief"]: break
        add = False
    else:
        add = True
    if add:
        state["messages"] = list(state["messages"]) + [HumanMessage(content=LEARNER[i % len(LEARNER)])]
    out = g.invoke(dict(state), config=cfg)
    state = {k: out[k] for k in KEYS}
    if out["messages"]:
        last = out["messages"][-1]
        if add:
            print(f"[{out['phase']:<14}] LEARNER: {LEARNER[i % len(LEARNER)]}")
            print(f"[{out['phase']:<14}] PATIENT: {last.content[:160]}")
    if not add:
        break

print()
print(f"turns={state['total_turns']}  phase={state['phase']}")
print("phase reasons:")
for k, v in state["phase_reasons"].items(): print(f"   {k}: {v!r}")
d = state["debrief"]
print()
print(f"DEBRIEF: {d['overall']}% — {d['band']}")
for s in d["phase_scores"]:
    print(f"   {s['phase']:<15} {s['score']}/{s['max_score']}")
    print(f"      missed: {s['missed']}")
print(f"   focus: {d['focus_areas']}")
print(f"   strengths: {d['strengths']}")
