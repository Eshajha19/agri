import main


def test_normalize_dynamic_alerts_avoids_id_collisions():
    stored_notifications = main._notification_store.get_recent()
    dynamic_alerts = [
        {"id": 1, "type": "warning", "message": "advisory-1", "time": "2026-05-21T00:00:00"},
        {"id": 2, "type": "info", "message": "advisory-2", "time": "2026-05-21T00:00:01"},
    ]

    merged = stored_notifications + main._normalize_dynamic_alerts(dynamic_alerts)
    ids = [entry["id"] for entry in merged]

    assert len(ids) == len(set(ids))
    assert all(entry["id"] < 0 for entry in merged[len(stored_notifications):])
