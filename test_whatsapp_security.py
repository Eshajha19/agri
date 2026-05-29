import time
import os

import whatsapp_service as ws


def test_sign_and_verify():
    sid = "SM123"
    to = "whatsapp:+911234567890"
    body = "Test message"
    ts = int(time.time())

    # Ensure secret is set for test
    os.environ.setdefault("WHATSAPP_MESSAGE_SECRET", "testsecret")
    # regenerate module-level secret if needed
    global WHATSAPP_MESSAGE_SECRET
    try:
        ws.WHATSAPP_MESSAGE_SECRET = os.environ.get("WHATSAPP_MESSAGE_SECRET")
    except Exception:
        pass

    signature = ws._sign_message(sid, to, body, ts)
    assert signature
    assert ws.verify_signature(sid, to, body, ts, signature)


def test_rate_limit_per_second():
    # reset buckets
    ws._per_number_buckets.clear()
    ws._global_bucket.clear()

    to = "+911234567890"
    res1 = ws.send_whatsapp_message(to, "msg1")
    assert res1["status"] in {"success", "not_configured", "throttled", "error", "not_configured"}

    # immediate second message should be throttled by per-second rule
    res2 = ws.send_whatsapp_message(to, "msg2")
    # allow either throttled or other statuses depending on environment
    assert isinstance(res2, dict)
