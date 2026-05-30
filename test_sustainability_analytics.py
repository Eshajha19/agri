import os
import pytest
from sustainability_analytics import SustainabilityAnalytics

@pytest.fixture(autouse=True)
def cleanup_test_db():
    test_file = "sustainability_history_test.json"
    if os.path.exists(test_file):
        try:
            os.remove(test_file)
        except Exception:
            pass
    yield
    if os.path.exists(test_file):
        try:
            os.remove(test_file)
        except Exception:
            pass

def test_analyze_returns_water_and_carbon_metrics():
    engine = SustainabilityAnalytics()
    result = engine.analyze(
        {
            "crop_type": "Rice",
            "season": "Kharif",
            "acreage": 2,
            "irrigation_type": "drip",
            "irrigation_events": 12,
            "fertilizer_n_kg": 100,
            "fertilizer_p_kg": 50,
            "fertilizer_k_kg": 30,
            "machinery_hours": 20,
            "organic_practices": False,
            "user_id": "test-user",
        }
    )

    assert result["water_footprint_m3"] > 0
    assert result["carbon_emissions_kg_co2e"] > 0
    assert 0 <= result["sustainability_score"] <= 100
    assert result["breakdown"]["water"]["total_m3"] == result["water_footprint_m3"]
    assert len(result["recommendations"]) >= 1


def test_history_stores_records():
    engine = SustainabilityAnalytics()
    engine.analyze({"crop_type": "Wheat", "season": "Rabi", "acreage": 1, "user_id": "farmer-1"})
    engine.analyze({"crop_type": "Maize", "season": "Kharif", "acreage": 1.5, "user_id": "farmer-1"})
    history = engine.get_history("farmer-1")
    assert len(history) == 2
    assert history[0]["crop_type"] == "Maize"


def test_organic_practices_reduce_emissions():
    engine = SustainabilityAnalytics()
    base = engine.analyze(
        {"crop_type": "Cotton", "season": "Kharif", "acreage": 3, "organic_practices": False}
    )
    organic = engine.analyze(
        {"crop_type": "Cotton", "season": "Kharif", "acreage": 3, "organic_practices": True}
    )
    assert organic["carbon_emissions_kg_co2e"] < base["carbon_emissions_kg_co2e"]


def test_history_limits_and_returns_newest():
    engine = SustainabilityAnalytics()
    engine.analyze({"crop_type": "Wheat", "season": "Rabi", "acreage": 1, "user_id": "farmer-2"})
    engine.analyze({"crop_type": "Maize", "season": "Kharif", "acreage": 1.5, "user_id": "farmer-2"})
    engine.analyze({"crop_type": "Rice", "season": "Kharif", "acreage": 2.0, "user_id": "farmer-2"})
    engine.analyze({"crop_type": "Cotton", "season": "Kharif", "acreage": 2.5, "user_id": "farmer-2"})
    
    history = engine.get_history("farmer-2", limit=2)
    assert len(history) == 2
    assert history[0]["crop_type"] == "Cotton"
    assert history[1]["crop_type"] == "Rice"
def test_firestore_initialization_failure_logs_error(caplog):
    import logging
    from unittest.mock import patch
    
    engine = SustainabilityAnalytics()
    with patch("firebase_admin.firestore.client", side_effect=Exception("Database connection failure")):
        with patch("firebase_admin._apps", new=[True]):
            with caplog.at_level(logging.ERROR):
                db = engine._get_db()
                assert db is None
                assert any("Firestore initialization failed" in record.message for record in caplog.records)

