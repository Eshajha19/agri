import pytest
from backend.sustainability_analytics import SustainabilityAnalytics

def make_payload(acreage=10, rainfall_mm=0, effective_rainfall_mm=None):
    return {
        "crop_type": "wheat",
        "season": "rabi",
        "acreage": acreage,
        "irrigation_type": "drip",
        "irrigation_events": 5,
        "fertilizer_n_kg": None,
        "fertilizer_p_kg": None,
        "fertilizer_k_kg": None,
        "machinery_hours": None,
        "diesel_liters": None,
        "organic_practices": False,
        "user_id": "test-user",
        "rainfall_mm": rainfall_mm,
        "effective_rainfall_mm": effective_rainfall_mm,
        "location": "Rohtak"
    }

def test_no_rainfall():
    sa = SustainabilityAnalytics()
    result = sa.analyze(make_payload(rainfall_mm=0))
    assert result["water_footprint_m3"] > 0

def test_moderate_rainfall_reduces_irrigation():
    sa = SustainabilityAnalytics()
    result_no_rain = sa.analyze(make_payload(rainfall_mm=0))
    result_with_rain = sa.analyze(make_payload(rainfall_mm=200))
    assert result_with_rain["water_footprint_m3"] < result_no_rain["water_footprint_m3"]

def test_excessive_rainfall_clamps_to_zero_irrigation():
    sa = SustainabilityAnalytics()
    result = sa.analyze(make_payload(rainfall_mm=5000))
    # irrigation demand should not go negative
    assert result["water_footprint_m3"] >= 0
