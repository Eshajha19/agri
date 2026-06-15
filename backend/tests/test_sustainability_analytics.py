import threading
import pytest
from backend.sustainability_analytics import SustainabilityAnalytics

def sample_payload():
    return {
        "crop_type": "wheat",
        "season": "rabi",
        "acreage": 2,
        "irrigation_type": "drip",
        "irrigation_events": 5,
        "fertilizer_n_kg": None,
        "fertilizer_p_kg": None,
        "fertilizer_k_kg": None,
        "machinery_hours": None,
        "diesel_liters": None,
        "organic_practices": False,
    }

def test_analyze_runs_without_history_lock_error():
    sa = SustainabilityAnalytics()
    result = sa.analyze(sample_payload())
    assert "water_footprint_m3" in result
    assert "carbon_emissions_kg" in result

def test_concurrent_analyze_threadsafe():
    sa = SustainabilityAnalytics()
    payload = sample_payload()

    def worker():
        sa.analyze(payload)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()

    # After concurrent calls, history should exist and not crash
    assert len(sa._history) > 0
