def test_sender_number_consistency(client):
    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{"id": "m1", "from": "15559999999", "text": {"body": "Hi"}}]
                }
            }]
        }]
    }
    raw_body = json.dumps(payload)

    # Matching sender_number → passes
    msgs = validate_and_parse(raw_body, sender_number="15559999999")
    assert len(msgs) == 1

    # Mismatched sender_number → raises
    with pytest.raises(ValueError):
        validate_and_parse(raw_body, sender_number="15550001111")


def test_unknown_message_type_strict_mode(monkeypatch):
    monkeypatch.setenv("WHATSAPP_STRICT_MODE", "true")
    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{"id": "m1", "from": "123", "type": "unknown"}]
                }
            }]
        }]
    }
    with pytest.raises(HTTPException):
        normalize_whatsapp_payload(payload)

def test_unknown_message_type_warning_mode(monkeypatch, caplog):
    monkeypatch.setenv("WHATSAPP_STRICT_MODE", "false")
    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{"id": "m1", "from": "123", "type": "unknown"}]
                }
            }]
        }]
    }
    msgs = normalize_whatsapp_payload(payload)
    assert msgs == []  # skipped
    assert "Unknown WhatsApp message type" in caplog.text
