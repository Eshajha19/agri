from advisory_rules import generate_advisories


def test_advisory_warns_to_avoid_irrigation_when_rain_expected():
    alerts = generate_advisories(
        weather={"rainfall_next_24h": 12, "temperature": 30},
        soil={"nitrogen": "medium"},
        crop_type="rice",
    )

    assert any("Avoid irrigation" in alert["action"] for alert in alerts)


def test_advisory_recommends_fertilizer_for_low_nitrogen():
    alerts = generate_advisories(
        weather={"temperature": 31},
        soil={"nitrogen": "low", "phosphorus": "medium", "potassium": "medium"},
        crop_type="wheat",
    )

    assert any(alert["title"] == "Low nitrogen detected" for alert in alerts)


def test_advisory_recommends_watering_for_high_temperature():
    alerts = generate_advisories(
        weather={"temperature": 40},
        soil={"nitrogen": "medium"},
        crop_type="maize",
    )

    assert any("Water early morning" in alert["action"] for alert in alerts)


def test_crop_thresholds_consistency():
    import alert_rules
    import advisory_rules
    from backend.core.constants import CROP_THRESHOLDS
    
    assert alert_rules.CROP_THRESHOLDS is CROP_THRESHOLDS
    assert advisory_rules.CROP_THRESHOLDS is CROP_THRESHOLDS
    
    # Check that keys required by advisory_rules are present
    for crop, data in CROP_THRESHOLDS.items():
        assert "temp_ideal" in data
        assert "ph_range" in data
        assert "moisture_min" in data
        
    # Check that keys required by alert_rules are present
    for crop in ["rice", "wheat", "cotton", "maize", "sugarcane"]:
        data = CROP_THRESHOLDS[crop]
        assert "temp_min" in data
        assert "temp_max" in data
        assert "ph_min" in data
        assert "ph_max" in data
        assert "critical_stages" in data
