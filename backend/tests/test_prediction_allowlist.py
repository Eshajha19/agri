def test_unknown_field_rejected(client):
    payload = {
        "crop_type": "wheat",
        "region": "north",
        "season": "summer",
        "soil_quality": 0.8,
        "rainfall": 120,
        "hacker_field": "boom"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422

def test_valid_payload_passes(client):
    payload = {
        "crop_type": "wheat",
        "region": "north",
        "season": "summer",
        "soil_quality": 0.8,
        "rainfall": 120
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200

def test_yield_lag_requires_lag_days(client):
    payload = {
        "crop_type": "wheat",
        "region": "north",
        "season": "summer",
        "soil_quality": 0.8,
        "rainfall": 120,
        "lag_days": 5
    }
    r = client.post("/predict-yield-lag", json=payload)
    assert r.status_code == 200
