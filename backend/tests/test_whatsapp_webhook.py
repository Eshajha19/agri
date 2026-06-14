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
