from backend.routers import voice_assistant as voice_assistant_router


def setup_function():
    voice_assistant_router._rate_limit_store.clear()
    voice_assistant_router._last_rate_limit_prune = 0.0


def test_rate_limit_prunes_expired_entries(monkeypatch):
    current_time = 100.0
    monkeypatch.setattr(voice_assistant_router, "monotonic", lambda: current_time)

    voice_assistant_router._rate_limit_store["stale-user"] = (
        3,
        current_time - voice_assistant_router.RATE_LIMIT_WINDOW - 1,
    )
    voice_assistant_router._rate_limit_store["fresh-user"] = (2, current_time)

    assert voice_assistant_router._check_rate_limit("fresh-user") is True
    assert "stale-user" not in voice_assistant_router._rate_limit_store
    assert voice_assistant_router._rate_limit_store["fresh-user"] == (3, current_time)