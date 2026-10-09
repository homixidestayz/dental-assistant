"""Graph node for the closing phase.

Remaining questions answered, safety-netting, follow-up. Then the encounter is over.
"""

from dental.phase_runner import run_phase
from dental.state import Phase


def node_closing(state: dict, llm=None) -> dict:
    """One turn of the closing phase."""
    return run_phase(state, Phase.CLOSING, llm=llm)
