def test_multiple_messages_processed(client):
    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [
                        {"id": "m1", "from": "123", "text": {"body": "Hello"}},
                        {"id": "m2", "from": "123", "text": {"body": "World"}},
                    ]
                }
            }]
        }]
    }
    response = client.post("/api/whatsapp/webhook", json=payload)
    assert response.status_code == 200
    assert response.json()["processed"] == 2
