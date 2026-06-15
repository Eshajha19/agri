import pytest
from backend.enums.message_type import MessageType
from backend.twilio_webhook_security import normalize_whatsapp_payload, STRICT_MODE

def test_enum_values():
    # Verify enum values match expected strings
    assert MessageType.TEXT.value == "text"
    assert MessageType.IMAGE.value == "image"
    assert MessageType.AUDIO.value == "audio"
    assert MessageType.VIDEO.value == "video"
    assert MessageType.DOCUMENT.value == "document"
    assert MessageType.STICKER.value == "sticker"
    assert MessageType.BUTTON.value == "button"

def test_normalize_text_message():
    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "id": "m1",
                        "from": "123",
                        "timestamp": "111111",
                        "type": "text",
                        "text": {"body": "Hello"}
                    }]
                }
            }]
        }]
    }
    msgs = normalize_whatsapp_payload(payload)
    assert len(msgs) == 1
    assert msgs[0]["text"] == "Hello"
    assert msgs[0]["media_id"] is None
    assert msgs[0]["type"] == MessageType.TEXT.value

def test_normalize_media_message():
    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "id": "m2",
                        "from": "123",
                        "timestamp": "222222",
                        "type": "image",
                        "image": {"id": "img123"}
                    }]
                }
            }]
        }]
    }
    msgs = normalize_whatsapp_payload(payload)
    assert len(msgs) == 1
    assert msgs[0]["text"] is None
    assert msgs[0]["media_id"] == "img123"
    assert msgs[0]["type"] == MessageType.IMAGE.value

def test_unknown_message_type_warning_mode(monkeypatch, caplog):
    monkeypatch.setenv("WHATSAPP_STRICT_MODE", "false")
    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "id": "m3",
                        "from": "123",
                        "timestamp": "333333",
                        "type": "unknown"
                    }]
                }
            }]
        }]
    }
    msgs = normalize_whatsapp_payload(payload)
    assert msgs == []  # skipped
    assert "Unknown WhatsApp message type" in caplog.text

def test_unknown_message_type_strict_mode(monkeypatch):
    monkeypatch.setenv("WHATSAPP_STRICT_MODE", "true")
    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "id": "m4",
                        "from": "123",
                        "timestamp": "444444",
                        "type": "unknown"
                    }]
                }
            }]
        }]
    }
    with pytest.raises(Exception):
        normalize_whatsapp_payload(payload)
