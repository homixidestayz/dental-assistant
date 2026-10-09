"""Graph node for the diagnosis phase.

The candidate telling you what they think is going on and what else it could be. A
patient does not know, so respond as one who is worried.
"""

from dental.phase_runner import run_phase
from dental.state import Phase


def node_diagnosis(state: dict, llm=None) -> dict:
    """One turn of the diagnosis phase."""
    return run_phase(state, Phase.DIAGNOSIS, llm=llm)
