"""Graph node for the management phase.

The treatment plan, consent, risks, and cost. Patients ask the questions a patient
would ask.
"""

from dental.phase_runner import run_phase
from dental.state import Phase


def node_management(state: dict, llm=None) -> dict:
    """One turn of the management phase."""
    return run_phase(state, Phase.MANAGEMENT, llm=llm)
