"""Graph node for the complaint phase.

The patient opens with why they are here, in their own words. Let them lead; do not
question them yet.
"""

from dental.phase_runner import run_phase
from dental.state import Phase


def node_complaint(state: dict, llm=None) -> dict:
    """One turn of the complaint phase."""
    return run_phase(state, Phase.COMPLAINT, llm=llm)
