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
