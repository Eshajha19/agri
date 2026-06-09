import asyncio

from backend.routers import advisory as advisory_router


def setup_function():
    advisory_router._stored_alerts.clear()
    advisory_router._graph_history.clear()


async def _call_recommendation(payload, monkeypatch, uid="farmer-123"):
    async def fake_uid(_request):
        return uid

    monkeypatch.setattr(advisory_router, "_get_authenticated_uid", fake_uid)
    return await advisory_router.create_farm_intelligence(payload, request=object())


def test_farm_intelligence_builds_graph_and_cross_factor_reasoning(monkeypatch):
    payload = advisory_router.FarmIntelligenceRequest(
        crop_type="rice",
        weather={"temperature": 34, "humidity": 82, "rainfall_next_24h": 12},
        soil={"moisture": 22, "ph": 6.4, "nitrogen": "low"},
        pest={"pressure": "high", "observed": "brown plant hopper"},
        market={"commodity": "Rice", "price": 4200, "trend": "rising"},
        location="Punjab",
    )

    response = asyncio.run(_call_recommendation(payload, monkeypatch))

    assert response["success"] is True
    assert response["scores"]["pest_risk"] >= 60
    assert any(edge["from"] == "weather" and edge["to"] == "pest-risk" for edge in response["graph"]["edges"])
    assert any("irrigation" in item["title"].lower() for item in response["recommendations"])
    assert any("pest risk" in line.lower() for line in response["reasoning"])


def test_farm_intelligence_history_is_per_user(monkeypatch):
    first = advisory_router.FarmIntelligenceRequest(
        crop_type="wheat",
        weather={"temperature": 29, "humidity": 55, "rainfall_next_24h": 1},
        soil={"moisture": 30},
        pest={"pressure": "medium"},
        market={"commodity": "Wheat", "price": 3600, "trend": "stable"},
        location="Haryana",
    )
    second = advisory_router.FarmIntelligenceRequest(
        crop_type="cotton",
        weather={"temperature": 33, "humidity": 78, "rainfall_next_24h": 0},
        soil={"moisture": 18},
        pest={"pressure": "high"},
        market={"commodity": "Cotton", "price": 7800, "trend": "up"},
        location="Gujarat",
    )

    asyncio.run(_call_recommendation(first, monkeypatch, uid="farmer-123"))
    asyncio.run(_call_recommendation(second, monkeypatch, uid="farmer-123"))

    async def fake_uid(_request):
        return "farmer-123"

    monkeypatch.setattr(advisory_router, "_get_authenticated_uid", fake_uid)
    history = asyncio.run(advisory_router.get_my_farm_intelligence(request=object()))

    assert history["success"] is True
    assert len(history["data"]) == 2
    assert history["data"][0]["crop_type"] == "cotton"
    assert history["data"][1]["crop_type"] == "wheat"
