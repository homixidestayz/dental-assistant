"""Graph assembly: the phase machine.

Each clinical phase is a node with a self-loop. `should_advance` decides whether a
phase is done, so the flow is agent-paced — the model decides, the graph enforces the
ceiling. The node set is derived from PHASE_ORDER, so adding a phase means adding a
node module and nothing here.
"""

from typing import Any

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
from dental.phase_runner import run_phase, should_advance
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


def build_graph() -> StateGraph:
    builder = StateGraph(DentalState)

    for phase in PHASE_ORDER:
        builder.add_node(phase.value, make_node(phase))

    builder.add_node(Phase.DEBRIEF.value, make_node(Phase.DEBRIEF))
    builder.add_edge(START, Phase.COMPLAINT.value)

    # One conditional edge per phase, and no static self-loop alongside it: both
    # would fire, writing `phase` twice in one step. should_advance returns either
    # the same phase (spend another turn here) or the next one.
    destinations = {phase.value: phase.value for phase in Phase}
    for phase in PHASE_ORDER:
        builder.add_conditional_edges(phase.value, should_advance, destinations)

    builder.add_edge(Phase.DEBRIEF.value, END)
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
