"""Tests for safe error detail utility — no raw exception text in responses."""

import logging
from error_utils import safe_detail


class TestSafeDetail:
    def test_returns_generic_message(self):
        msg = safe_detail(ValueError("sensitive path: /etc/passwd"), 400)
        assert msg == "Invalid request"
        assert "sensitive" not in msg
        assert "/etc/passwd" not in msg

    def test_returns_generic_for_500(self):
        msg = safe_detail(RuntimeError("model weights at /var/models/42"), 500)
        assert msg == "Internal server error"
        assert "/var/models" not in msg

    def test_returns_generic_for_403(self):
        msg = safe_detail(PermissionError("admin override token: xyz123"), 403)
        assert msg == "Access denied"

    def test_returns_generic_for_404(self):
        msg = safe_detail(KeyError("user_private_data"), 404)
        assert msg == "Resource not found"

    def test_returns_generic_for_unknown_status(self):
        msg = safe_detail(Exception("anything"), 999)
        assert msg == "An error occurred"

    def test_logs_full_exception(self, caplog):
        caplog.set_level(logging.ERROR)
        safe_detail(ValueError("real detail: path=/secret/key"), 400, request_id="req-123")
        assert "req-123" in caplog.text
        assert "ValueError" in caplog.text
        assert "real detail" in caplog.text
        assert "/secret/key" in caplog.text

    def test_accepts_exc_without_request_id(self):
        msg = safe_detail(Exception("test"), 500)
        assert msg == "Internal server error"
