"""Graph node for the history phase.

History of the complaint, then medical, drug, and dental history, allergies, social
and habit history.
"""

from dental.phase_runner import run_phase
from dental.state import Phase


def node_history(state: dict, llm=None) -> dict:
    """One turn of the history phase."""
    return run_phase(state, Phase.HISTORY, llm=llm)
