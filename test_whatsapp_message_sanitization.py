"""
Tests for WhatsApp message content sanitization: control character stripping,
safe formatting, and valid output across special character inputs.
"""

import re

import pytest
from pydantic import BaseModel, Field, validator

from backend.routers.alerts import AlertTriggerRequest as AlertsRequest
from whatsapp_service import format_alert_message, sanitise_message


# Inline model to test without importing main.py (which has side effects)
class _TestRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)

    @validator("message")
    def strip_control_chars(cls, v):
        return sanitise_message(v)


class TestSanitiseMessage:
    def test_strips_null_bytes(self):
        assert sanitise_message("hello\x00world") == "helloworld"

    def test_strips_bell_and_escape_chars(self):
        assert sanitise_message("a\x07b\x1bc") == "abc"

    def test_preserves_newlines_tabs_carriage_returns(self):
        result = sanitise_message("line1\nline2\tcol2\r\nline3")
        assert result == "line1\nline2\tcol2\r\nline3"

    def test_strips_all_ascii_control_except_nl_cr_tab(self):
        all_controls = bytes(i for i in range(0x20)).decode("latin-1")
        result = sanitise_message(all_controls)
        assert sorted(result) == sorted("\n\r\t")

    def test_strips_delete_char(self):
        assert sanitise_message("abc\x7fdef") == "abcdef"

    def test_preserves_emojis(self):
        result = sanitise_message("Hello 🌾🚜🐛✅")
        assert "🌾" in result
        assert "🚜" in result
        assert "🐛" in result

    def test_empty_string_returns_empty(self):
        assert sanitise_message("") == ""


class TestPydanticSanitization:
    def test_control_chars_stripped_from_message(self):
        req = _TestRequest(message="flood\x00warning")
        assert req.message == "floodwarning"

    def test_newlines_preserved(self):
        req = _TestRequest(message="line1\nline2")
        assert req.message == "line1\nline2"

    def test_emojis_allowed(self):
        req = _TestRequest(message="🐛 outbreak detected")
        assert "🐛" in req.message

    def test_after_strip_empty_raises(self):
        with pytest.raises(Exception):
            _TestRequest(message="")

    def test_control_chars_full_range(self):
        dirty = "safe" + "".join(chr(i) for i in range(0x20) if i not in (9, 10, 13)) + "safe"
        req = _TestRequest(message=dirty)
        assert req.message == "safesafe"

    def test_delete_char_stripped(self):
        req = _TestRequest(message="hi\x7fthere")
        assert req.message == "hithere"


class TestBackendAlertsModel:
    """Verifies the actual backend/routers/alerts.py model has the validator."""

    def test_control_chars_stripped(self):
        req = AlertsRequest(alert_type="weather", message="alert\x00test")
        assert req.message == "alerttest"

    def test_normal_message_passes(self):
        req = AlertsRequest(alert_type="pest", message="Pest detected in north region")
        assert req.message == "Pest detected in north region"


class TestFormatAlertMessage:
    def test_control_chars_in_content_are_stripped(self):
        result = format_alert_message("weather", "storm\x00surge")
        assert "\x00" not in result
        assert "storm" in result
        assert "surge" in result

    def test_newlines_preserved_in_output(self):
        result = format_alert_message("advisory", "line1\nline2")
        assert "line1\nline2" in result

    def test_emojis_preserved(self):
        result = format_alert_message("pest", "🐛 detected")
        assert "🐛" in result

    def test_weather_format_contains_weather_warning(self):
        result = format_alert_message("weather", "Heavy rain")
        assert "Weather Warning" in result
        assert "⛈️" in result

    def test_pest_format_contains_pest_alert(self):
        result = format_alert_message("pest", "Locusts spotted")
        assert "Pest Outbreak Alert" in result
        assert "🐛" in result

    def test_advisory_format_contains_advisory(self):
        result = format_alert_message("advisory", "Fertilize now")
        assert "Farming Advisory" in result
        assert "📝" in result

    def test_unknown_type_uses_fallback(self):
        result = format_alert_message("unknown", "test")
        assert "Notification" in result
        assert "📢" in result

    def test_footer_present(self):
        result = format_alert_message("weather", "test")
        assert "Stay safe and stay informed with Fasal Saathi" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
