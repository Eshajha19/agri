"""
Tests for ErrorRecoveryMiddleware payload detection.

Covers:
- looks_like_binary: gzip/zip magic bytes, non-printable heuristic
- looks_like_base64: true binary-in-base64, plain-text false-positives
- _contains_suspicious_content: real injections vs benign prose
"""

import base64
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from error_recovery_middleware import (
    looks_like_binary,
    looks_like_base64,
    ErrorRecoveryMiddleware,
)


# ── looks_like_binary ────────────────────────────────────────────────────────

class TestLooksLikeBinary:
    def test_gzip_magic_bytes(self):
        assert looks_like_binary(b"\x1F\x8B" + b"\x00" * 20) is True

    def test_zip_magic_bytes(self):
        assert looks_like_binary(b"PK\x03\x04" + b"\x00" * 20) is True

    def test_high_non_printable_ratio(self):
        # Mostly bytes in the non-printable range
        payload = bytes([0x01, 0x02, 0x03, 0x04]) * 50 + b"hello"
        assert looks_like_binary(payload) is True

    def test_plain_text_not_binary(self):
        assert looks_like_binary(b"Hello, this is a normal crop report.") is False

    def test_json_not_binary(self):
        assert looks_like_binary(b'{"crop": "wheat", "area": 5.2}') is False

    def test_empty_payload(self):
        assert looks_like_binary(b"") is False


# ── looks_like_base64 ────────────────────────────────────────────────────────

class TestLooksLikeBase64:
    def test_base64_of_binary_data_flagged(self):
        # Bytes 0-255 → definitely not valid UTF-8 when decoded
        binary_data = bytes(range(256)) * 3
        encoded = base64.b64encode(binary_data).decode()
        assert looks_like_base64(encoded) is True

    def test_base64_of_plain_text_not_flagged(self):
        # Encodes to base64 but decodes back to valid UTF-8 — not suspicious
        encoded = base64.b64encode(b"normal crop report content " * 5).decode()
        assert looks_like_base64(encoded) is False

    def test_short_string_not_flagged(self):
        # Under 100 chars — gate prevents false positives on short tokens
        sample = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo="
        assert len(sample) < 100
        assert looks_like_base64(sample) is False

    def test_plain_alphanumeric_not_flagged(self):
        # Long alphanumeric string that isn't base64 of binary
        plain = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" * 3
        assert looks_like_base64(plain) is False

    def test_uuid_list_not_flagged(self):
        # A list of UUIDs could look base64-ish but decodes to valid UTF-8
        uuids = "a1b2c3d4-e5f6-7890-abcd-ef1234567890" * 4
        assert looks_like_base64(uuids) is False

    def test_invalid_base64_not_flagged(self):
        # Contains characters outside the base64 alphabet
        assert looks_like_base64("this is not base64!!! " * 10) is False


# ── _contains_suspicious_content ─────────────────────────────────────────────

class TestSuspiciousContent:
    """Tests run through the middleware method directly (no HTTP stack needed)."""

    @pytest.fixture
    def mw(self):
        # Minimal fake ASGI app; we only call the detection method, not dispatch
        async def dummy_app(scope, receive, send):
            pass
        return ErrorRecoveryMiddleware(dummy_app)

    # ── Benign strings that must NOT be flagged ──────────────────────────────

    def test_prose_drop_table(self, mw):
        # Common agricultural instruction — "drop table salt"
        assert mw._contains_suspicious_content(
            "How do I drop table salt evenly across my field?"
        ) is False

    def test_prose_union_select(self, mw):
        assert mw._contains_suspicious_content(
            "I want to union select the best wheat and rice varieties."
        ) is False

    def test_prose_script_word(self, mw):
        assert mw._contains_suspicious_content(
            "Here is the script for our agricultural training video."
        ) is False

    def test_normal_json(self, mw):
        assert mw._contains_suspicious_content(
            '{"name": "Ravi Kumar", "crop": "wheat", "area": 5.2}'
        ) is False

    def test_empty_string(self, mw):
        assert mw._contains_suspicious_content("") is False

    def test_crop_notes_with_select(self, mw):
        assert mw._contains_suspicious_content(
            "Please select the best irrigation method for our crop."
        ) is False

    # ── Real injection attempts that MUST be flagged ─────────────────────────

    def test_sql_drop_table_with_semicolon(self, mw):
        assert mw._contains_suspicious_content("'; DROP TABLE users;") is True

    def test_sql_union_select_with_from(self, mw):
        assert mw._contains_suspicious_content(
            "' UNION SELECT username, password FROM users--"
        ) is True

    def test_sql_or_tautology(self, mw):
        assert mw._contains_suspicious_content("' OR '1'='1") is True

    def test_sql_comment_prefix(self, mw):
        assert mw._contains_suspicious_content("-- DROP TABLE crops") is True

    def test_xss_script_tag(self, mw):
        assert mw._contains_suspicious_content("<script>alert('xss')</script>") is True

    def test_xss_script_tag_with_attrs(self, mw):
        assert mw._contains_suspicious_content(
            '<script src="https://evil.com/x.js"></script>'
        ) is True

    def test_xss_javascript_uri(self, mw):
        assert mw._contains_suspicious_content("javascript:alert(document.cookie)") is True

    def test_xss_onerror(self, mw):
        assert mw._contains_suspicious_content('<img src=x onerror=alert(1)>') is True

    def test_xss_onload(self, mw):
        assert mw._contains_suspicious_content('<body onload=alert(1)>') is True