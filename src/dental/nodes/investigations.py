"""Graph node for the investigations phase.

The investigations the candidate orders. Report a result only for a test they
actually order.
"""

from dental.phase_runner import run_phase
from dental.state import Phase


def node_investigations(state: dict, llm=None) -> dict:
    """One turn of the investigations phase."""
    return run_phase(state, Phase.INVESTIGATIONS, llm=llm)
