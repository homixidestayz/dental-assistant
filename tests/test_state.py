import pytest

from dental.state import (
    PHASE_ORDER,
    TURN_CEILING,
    Phase,
    initial_state,
    next_phase,
)


def test_phase_order_excludes_debrief():
    assert PHASE_ORDER == [
        Phase.COMPLAINT,
        Phase.HISTORY,
        Phase.EXAMINATION,
        Phase.INVESTIGATIONS,
        Phase.DIAGNOSIS,
        Phase.MANAGEMENT,
        Phase.CLOSING,
    ]


def test_next_phase_advances_through_order():
    assert next_phase(Phase.COMPLAINT) is Phase.HISTORY
    assert next_phase(Phase.CLOSING) is Phase.DEBRIEF


def test_next_phase_on_debrief_raises():
    with pytest.raises(ValueError):
        next_phase(Phase.DEBRIEF)


def test_turn_ceiling_is_eight():
    assert TURN_CEILING == 8


def test_initial_state_starts_at_complaint_with_zero_turns():
    state = initial_state()
    assert state["phase"] == Phase.COMPLAINT.value
    assert state["turns_in_phase"] == 0
    assert state["total_turns"] == 0
    assert state["messages"] == []
    assert state["debrief"] is None


def test_initial_state_has_empty_phase_reasons():
    assert initial_state()["phase_reasons"] == {}
