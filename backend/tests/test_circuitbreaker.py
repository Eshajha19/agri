import pytest
from error_recovery_middleware import CircuitBreakerState

def test_state_dict_prunes():
    cb = CircuitBreakerState(max_entries=5)
    for i in range(20):
        cb.set("GET", f"/api/items?id={i}", {"failures": i})
    # Should never exceed 5 entries
    assert len(cb._state) <= 5
