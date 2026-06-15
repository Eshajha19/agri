from backend.realtime_notifications import NotificationEvent

def test_sha256_dedup_consistency():
    e1 = NotificationEvent(type="info", data={"msg": "hello"})
    e2 = NotificationEvent(type="info", data={"msg": "hello"})
    e3 = NotificationEvent(type="info", data={"msg": "different"})

    # Same content → same SHA-256 hash
    assert e1.get_content_hash() == e2.get_content_hash()

    # Different content → different SHA-256 hash
    assert e1.get_content_hash() != e3.get_content_hash()

def test_sha256_length():
    e = NotificationEvent(type="info", data={"msg": "test"})
    h = e.get_content_hash()
    # SHA-256 hex digest length is always 64
    assert len(h) == 64
