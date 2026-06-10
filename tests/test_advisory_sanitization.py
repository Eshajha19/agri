import asyncio

from backend.routers import advisory as advisory_router


def setup_function():
    advisory_router._stored_alerts.clear()
    advisory_router._graph_history.clear()


def test_create_advisory_recursively_sanitises_alerts(monkeypatch):
    malicious_alerts = [
        {
            "title": "<script>alert('x')</script>",
            "severity": 5,
            "metadata": {
                "source": "<img src=x onerror=alert(1)>",
                "values": [1, "<b>bold</b>", {"deep": "<svg onload=alert(2)>"}],
            },
        }
    ]

    async def fake_uid(_request):
        return "farmer-123"

    monkeypatch.setattr(advisory_router, "generate_advisories", lambda **kwargs: malicious_alerts)
    monkeypatch.setattr(advisory_router, "_get_authenticated_uid", fake_uid)

    payload = advisory_router.AdvisoryRequest(
        weather={"temperature": 31},
        soil={"nitrogen": "low"},
        crop_type="rice",
        store_alerts=True,
    )

    response = asyncio.run(advisory_router.create_advisory(payload, request=object()))

    assert response["data"][0]["title"] == "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;"
    assert response["data"][0]["severity"] == "5"
    assert response["data"][0]["metadata"]["source"] == "&lt;img src=x onerror=alert(1)&gt;"
    assert response["data"][0]["metadata"]["values"][0] == "1"
    assert response["data"][0]["metadata"]["values"][1] == "&lt;b&gt;bold&lt;/b&gt;"
    assert response["data"][0]["metadata"]["values"][2]["deep"] == "&lt;svg onload=alert(2)&gt;"
    assert response["data"][0]["sanitised"] == "true"

    stored_alerts = list(advisory_router._stored_alerts["farmer-123"])
    assert stored_alerts == response["data"]