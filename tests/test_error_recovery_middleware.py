"""Tests for error recovery middleware — encoding-aware suspicious-content scan."""

import base64
import pytest
from error_recovery_middleware import (
    SuspiciousContentScanner,
    ConfidenceLevel,
    ScanResult,
)


class TestSuspiciousContentScanner:
    """SuspiciousContentScanner in isolation."""

    @pytest.fixture
    def scanner(self):
        return SuspiciousContentScanner()

    # ── is_binary ───────────────────────────────────────────────────────

    @pytest.mark.parametrize("payload,expected", [
        (b"hello world", False),
        (b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f" * 10, True),
        (b"mostly text with a \x00 null byte", False),  # single null < 5 %
        (b"\xff\xfe\xfd\xfc" * 50, False),  # valid Latin-1 extended chars
    ])
    def test_is_binary(self, payload, expected):
        assert SuspiciousContentScanner._is_binary(payload) == expected

    # ── is_base64_like ──────────────────────────────────────────────────

    @pytest.mark.parametrize("payload,expected", [
        (base64.b64encode(b'{"hello":"world"}'), True),
        (b"SGVsbG8gV29ybGQ=", True),
        (b"short", False),
        (b"not base64!!! with \x00 binary!!", False),
        (b"QmFzZTY0IGVuY29kZWQgdGV4dA==", True),
    ])
    def test_is_base64_like(self, payload, expected):
        assert SuspiciousContentScanner._is_base64_like(payload) == expected

    # ── decode_with_confidence ──────────────────────────────────────────

    def test_decode_utf8_clean(self):
        text, conf = SuspiciousContentScanner._decode_with_confidence(b"hello world")
        assert text == "hello world"
        assert conf == ConfidenceLevel.HIGH

    def test_decode_utf8_with_replacement(self):
        raw = "hello \udc80 world".encode("utf-8", errors="surrogatepass")
        raw = raw.replace(b"\xed\xb2\x80", b"\xff")
        text, conf = SuspiciousContentScanner._decode_with_confidence(raw)
        assert isinstance(text, str)
        assert conf == ConfidenceLevel.MEDIUM

    def test_decode_latin1_fallback(self):
        raw = bytes(range(0x80, 0x100))
        text, conf = SuspiciousContentScanner._decode_with_confidence(raw)
        assert isinstance(text, str)
        # Latin-1 can decode this but it has no UTF-8 structure → MEDIUM
        assert conf == ConfidenceLevel.MEDIUM

    def test_decode_base64_auto(self):
        raw = base64.b64encode(b"hello from base64")
        text, conf = SuspiciousContentScanner._decode_with_confidence(raw)
        assert "hello from base64" in text
        assert conf == ConfidenceLevel.HIGH

    def test_decode_binary_low_confidence(self):
        raw = b"\x00\x01\x02\x03\x04" * 20
        text, conf = SuspiciousContentScanner._decode_with_confidence(raw)
        assert conf == ConfidenceLevel.LOW

    def test_decode_base64_nested_xss(self):
        malicious = b"<script>alert('xss')</script>"
        encoded = base64.b64encode(malicious)
        text, conf = SuspiciousContentScanner._decode_with_confidence(encoded)
        assert "<script>" in text
        assert conf == ConfidenceLevel.HIGH

    # ── scan ────────────────────────────────────────────────────────────

    def test_scan_empty(self, scanner):
        result = scanner.scan(b"")
        assert not result.suspicious
        assert result.confidence == ConfidenceLevel.HIGH

    def test_scan_clean_text(self, scanner):
        result = scanner.scan(b"what is the weather today")
        assert not result.suspicious
        assert result.confidence == ConfidenceLevel.HIGH

    def test_scan_sql_injection(self, scanner):
        result = scanner.scan(b"' OR '1'='1")
        assert result.suspicious
        assert result.threat_type is not None

    def test_scan_xss(self, scanner):
        result = scanner.scan(b"<script>alert(1)</script>")
        assert result.suspicious
        assert result.threat_type is not None

    def test_scan_shell_injection(self, scanner):
        result = scanner.scan(b"|| sh -c 'ls'")
        assert result.suspicious

    def test_scan_ssti_jinja2(self, scanner):
        result = scanner.scan(b"{{ 7*7 }}")
        assert result.suspicious

    def test_scan_binary_skipped(self, scanner):
        """Binary payloads should be SKIPped, not false-positive."""
        result = scanner.scan(b"\x00\x01\x02\x03" * 50)
        assert not result.suspicious
        assert result.confidence == ConfidenceLevel.SKIP

    def test_scan_sql_in_base64(self, scanner):
        """SQL injection hidden in base64 must still be caught."""
        payload = base64.b64encode(b"' OR '1'='1' --")
        result = scanner.scan(payload)
        assert result.suspicious
        assert result.threat_type is not None

    def test_scan_latin1_does_not_false_positive(self, scanner):
        """Latin-1 bytes that happen to look like a pattern must not trigger."""
        raw = bytes(range(0xC0, 0x100)) * 4  # accented chars
        result = scanner.scan(raw)
        assert not result.suspicious  # not really attack content

    def test_scan_text_containing_comment_marker(self, scanner):
        """Benign text with '--' (double dash) must not trigger SQLi."""
        result = scanner.scan(b"the value range is 10--20 degrees")
        assert not result.suspicious

    def test_scan_mixed_encoding_medium_confidence(self, scanner):
        """Mixed valid UTF-8 + invalid bytes → MEDIUM → still scanned."""
        raw = b"hello \xff\xfe world ' OR '1'='1"
        result = scanner.scan(raw)
        assert result.suspicious
        assert result.confidence == ConfidenceLevel.MEDIUM

    # ── Confidence threshold override ───────────────────────────────────

    def test_scan_below_threshold_skips(self):
        """With HIGH threshold, MEDIUM-confidence content is skipped."""
        scanner = SuspiciousContentScanner(confidence_threshold=ConfidenceLevel.HIGH)
        raw = bytes(range(0x80, 0x100)) + b" OR '1'='1"
        result = scanner.scan(raw)
        # Confidence is MEDIUM (Latin-1 fallback) → below HIGH → skip
        assert not result.suspicious
        assert result.confidence == ConfidenceLevel.SKIP

    def test_scan_low_threshold_catches_all(self):
        """With LOW threshold, even garbled binary is scanned."""
        scanner = SuspiciousContentScanner(confidence_threshold=ConfidenceLevel.LOW)
        raw = b"\x00\x01\x02 ' OR '1'='1"
        result = scanner.scan(raw)
        assert result.suspicious
        assert result.confidence == ConfidenceLevel.LOW
