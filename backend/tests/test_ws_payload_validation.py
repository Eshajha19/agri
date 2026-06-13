import pytest
from backend.realtime_notifications import validate_ws_payload

def test_valid_delivery_ack():
    payload = {"type": "delivery_ack", "notification_id": "123"}
    validated = validate_ws_payload(payload)
    assert validated.notification_id == "123"

def test_valid_subscribe_crops():
    payload = {"type": "subscribe_crops", "crops": ["wheat", "rice"]}
    validated = validate_ws_payload(payload)
    assert "wheat" in validated.crops

def test_unknown_type_rejected():
    payload = {"type": "hacker_msg", "foo": "bar"}
    with pytest.raises(ValueError):
        validate_ws_payload(payload)

def test_extra_keys_rejected():
    payload = {"type": "delivery_ack", "notification_id": "123", "extra": "boom"}
    with pytest.raises(Exception):
        validate_ws_payload(payload)
