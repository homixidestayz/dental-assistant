"""Graph node for the examination phase.

Extraoral and intraoral examination, and soft tissue assessment. The candidate says
what they want to examine; you report only what is there.
"""

from dental.phase_runner import run_phase
from dental.state import Phase


def node_examination(state: dict, llm=None) -> dict:
    """One turn of the examination phase."""
    return run_phase(state, Phase.EXAMINATION, llm=llm)
