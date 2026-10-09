"""Graph assembly: the phase machine, one turn per invocation.

An invocation plays exactly one encounter turn and returns. The phase the next
invocation resumes at lives in `state["phase"]`, which `run_phase` advances.

That matters for a chat UI: the learner sends a message, the patient replies, control
returns. If the graph ran to the end of the encounter per invoke, every keystroke would
replay all seven phases and re-grade the transcript.

LangGraph Studio drives this the same way, invoking once per input.

Note the consequence: the graph has no self-loops. The turn loop belongs to the caller,
and the phase counter in state is where the next invocation resumes. A caller that
wants a whole encounter in one go must drive it turn by turn, as `app.py` does.
"""

from typing import Any, AsyncGenerator

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from dental.nodes import (
    closing,
    complaint,
    debrief,
    diagnosis,
    examination,
    history,
    investigations,
    management,
)
from dental.state import PHASE_ORDER, DentalState, Phase

NODE_MODULES = {
    Phase.COMPLAINT: complaint,
    Phase.HISTORY: history,
    Phase.EXAMINATION: examination,
    Phase.INVESTIGATIONS: investigations,
    Phase.DIAGNOSIS: diagnosis,
    Phase.MANAGEMENT: management,
    Phase.CLOSING: closing,
    Phase.DEBRIEF: debrief,
}


def make_node(phase: Phase):
    """The node callable for `phase`, closing over its own module.

    The model comes from state when present, so tests inject a fake without
    monkeypatching module globals.
    """
    fn = getattr(NODE_MODULES[phase], f"node_{phase.value}")

    def node(state: dict) -> dict:
        llm: Any = state.get("llm")
        return fn(state, llm=llm) if llm is not None else fn(state)

    node.__name__ = f"node_{phase.value}"
    return node


def entry_phase(state: dict) -> str:
    """Which node to start at: the phase the last invocation stopped in.

    Falls back to the complaint when there is nothing to resume, so a fresh state
    starts the encounter at the top.
    """
    saved = state.get("phase")
    if saved:
        try:
            return Phase(saved).value
        except ValueError:
            pass
    return Phase.COMPLAINT.value


def build_graph() -> StateGraph:
    """START -> the current phase -> END. One node, one turn, per invocation.

    Every phase is reachable as an entry point, because entry_phase routes to
    whichever one state says is current. No self-loops: the loop across turns is the
    caller's, and `phase` in state is where it resumes.
    """
    builder = StateGraph(DentalState)

    for phase in PHASE_ORDER:
        builder.add_node(phase.value, make_node(phase))
        builder.add_edge(phase.value, END)  # one turn per invocation

    builder.add_node(Phase.DEBRIEF.value, make_node(Phase.DEBRIEF))
    builder.add_edge(Phase.DEBRIEF.value, END)

    builder.add_conditional_edges(
        START, entry_phase, {phase.value: phase.value for phase in Phase}
    )
    return builder


# For langgraph.json — LangGraph Studio and the API supply their own checkpointer.
graph: CompiledStateGraph = build_graph().compile()

# The sibling projects' export alias.
ALIAS = "agent"


def get_graph(checkpointer=None) -> CompiledStateGraph:
    """For local runs (Streamlit): compiled with a checkpointer so the encounter
    persists across turns."""
    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
    return build_graph().compile(checkpointer=checkpointer)


async def stream_graph(state: dict, config: dict) -> AsyncGenerator[dict, None]:
    """Stream a single turn, yielding partial state updates as the model streams."""
    g = get_graph()
    async for chunk in g.astream(state, config=config, stream_mode="updates"):
        yield chunk
