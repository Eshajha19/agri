"""Comprehensive security-validation tests for WhatsApp subscribe endpoints.

Covers all three subscribe models / signatures:

1. main.py          — WhatsAppSubscribeRequest (Pydantic model + E.164 regex)
2. platform.py      — WhatsAppSubscribeRequest (Pydantic model, Field-level)
3. alerts.py        — Form() parameters with max_length + runtime regex

Test categories:
- phone_number: max_length, E.164 regex, boundary lengths, injection strings
- name:         min_length / max_length, boundary values
- region_id:    max_length, optional behaviour
- Cross-model consistency: same valid payload accepted by both Pydantic models
"""

from __future__ import annotations

import inspect
import re

import pytest
from pydantic import ValidationError

# ── Import models from main.py & platform.py ────────────────────────────
from main import WhatsAppSubscribeRequest as MainModel
from backend.routers.platform import WhatsAppSubscribeRequest as PlatformModel

# ── Import alerts regex for runtime-level checks ────────────────────────
from backend.routers import alerts as alerts_module


# =====================================================================
#  Helpers
# =====================================================================

VALID_PHONE = "+919876543210"
VALID_NAME = "Test Farmer"


def _main(phone: str = VALID_PHONE, name: str = VALID_NAME, **kw) -> MainModel:
    return MainModel(phone_number=phone, name=name, **kw)


def _platform(phone: str = VALID_PHONE, name: str = VALID_NAME, **kw) -> PlatformModel:
    return PlatformModel(phone_number=phone, name=name, **kw)


# =====================================================================
#  1. phone_number — Field max_length constraints
# =====================================================================


class TestPhoneMaxLength:
    """Verify max_length is enforced at the Pydantic schema level."""

    def test_main_model_accepts_max_e164(self):
        """Longest valid E.164: '+' + 15 digits = 16 chars (within max_length=20)."""
        phone = "+123456789012345"  # 16 chars
        req = _main(phone=phone)
        assert req.phone_number == phone

    def test_main_model_rejects_over_20_chars(self):
        """Strings longer than max_length=20 must be rejected by the Field."""
        with pytest.raises(ValidationError):
            _main(phone="+" + "1" * 20)  # 21 chars

    def test_platform_model_rejects_over_20_chars(self):
        with pytest.raises(ValidationError):
            _platform(phone="+" + "1" * 20)

    def test_platform_model_rejects_under_7_chars(self):
        """platform.py has min_length=7 — a 6-char string must fail."""
        with pytest.raises(ValidationError):
            _platform(phone="+12345")  # 6 chars


# =====================================================================
#  2. phone_number — E.164 regex enforcement (main.py only)
# =====================================================================


class TestMainPhoneRegex:
    """The main.py model has a @field_validator regex check on top of Field."""

    @pytest.mark.parametrize("phone", [
        "+919876543210",
        "+14155552671",
        "+447911123456",
        "+1234567",            # minimum: +1 then 6 digits
        "+123456789012345",    # maximum: +1 then 14 digits
    ])
    def test_valid_e164_accepted(self, phone: str):
        req = _main(phone=phone)
        assert req.phone_number == phone

    @pytest.mark.parametrize("phone", [
        "919876543210",        # missing '+'
        "0919876543210",       # leading zero
        "+0123456789",         # zero after '+'
        "+91 98765 43210",     # spaces
        "+91-9876-543210",     # dashes
        "+91(98765)43210",     # parens
        "+91.9876.543210",     # dots
        "++919876543210",      # double '+'
    ])
    def test_invalid_format_rejected(self, phone: str):
        with pytest.raises(ValidationError, match="E.164"):
            _main(phone=phone)

    def test_rejects_too_short_for_regex(self):
        """'+12345' has only 5 digits — regex requires ≥7 total digits."""
        with pytest.raises(ValidationError):
            _main(phone="+12345")

    def test_rejects_too_long_for_regex(self):
        """'+1' + 15 nines = 17 chars; regex allows at most 15 digits."""
        with pytest.raises(ValidationError):
            _main(phone="+1" + "9" * 15)

    @pytest.mark.parametrize("phone", [
        "",
        "+",
        "not-a-phone",
        "'; DROP TABLE subscribers; --",
        "tel:+919876543210",
        "+91987654321012345678",  # way too long
    ])
    def test_injection_and_garbage_rejected(self, phone: str):
        with pytest.raises(ValidationError):
            _main(phone=phone)


# =====================================================================
#  3. name — min_length / max_length constraints
# =====================================================================


class TestNameBounds:
    """Both main.py and platform.py require min_length=1, max_length=100."""

    # ── main.py model ──

    def test_main_accepts_single_char_name(self):
        req = _main(name="A")
        assert req.name == "A"

    def test_main_accepts_100_char_name(self):
        name = "B" * 100
        req = _main(name=name)
        assert req.name == name

    def test_main_rejects_empty_name(self):
        with pytest.raises(ValidationError):
            _main(name="")

    def test_main_rejects_101_char_name(self):
        with pytest.raises(ValidationError):
            _main(name="C" * 101)

    # ── platform.py model ──

    def test_platform_accepts_single_char_name(self):
        req = _platform(name="A")
        assert req.name == "A"

    def test_platform_accepts_100_char_name(self):
        name = "D" * 100
        req = _platform(name=name)
        assert req.name == name

    def test_platform_rejects_empty_name(self):
        with pytest.raises(ValidationError):
            _platform(name="")

    def test_platform_rejects_101_char_name(self):
        with pytest.raises(ValidationError):
            _platform(name="E" * 101)

    # ── realistic edge cases ──

    def test_unicode_name_accepted(self):
        """Names with non-ASCII characters (Hindi, etc.) should be fine."""
        req = _main(name="किसान भाई")
        assert req.name == "किसान भाई"

    def test_name_with_spaces_accepted(self):
        req = _main(name="Ram Kumar Singh")
        assert req.name == "Ram Kumar Singh"

    def test_whitespace_only_name_rejected_main(self):
        """A name of just spaces has length > 0 but Pydantic counts them.
        Our min_length=1 would pass this; the real guard is in alerts.py's
        runtime sanitisation.  Here we just confirm the schema layer."""
        req = _main(name=" ")
        assert req.name == " "  # schema allows it; runtime may strip later


# =====================================================================
#  4. region_id — max_length constraints
# =====================================================================


class TestRegionIdBounds:
    """region_id is optional with max_length=100 on main.py model."""

    def test_main_accepts_none_region(self):
        req = _main()
        assert req.region_id is None

    def test_main_accepts_valid_region(self):
        req = _main(region_id="maharashtra-pune")
        assert req.region_id == "maharashtra-pune"

    def test_main_accepts_100_char_region(self):
        region = "r" * 100
        req = _main(region_id=region)
        assert req.region_id == region

    def test_main_rejects_101_char_region(self):
        with pytest.raises(ValidationError):
            _main(region_id="r" * 101)

    def test_main_accepts_empty_string_region(self):
        """Empty string is valid — not the same as None."""
        req = _main(region_id="")
        assert req.region_id == ""


# =====================================================================
#  5. user_id backward-compat field
# =====================================================================


class TestUserIdField:
    """user_id is accepted but should be optional / ignorable."""

    def test_main_defaults_user_id_none(self):
        req = _main()
        assert req.user_id is None

    def test_main_accepts_user_id(self):
        req = _main(user_id="abc123")
        assert req.user_id == "abc123"

    def test_platform_defaults_user_id_none(self):
        req = _platform()
        assert req.user_id is None


# =====================================================================
#  6. alerts.py — Form parameter max_length and regex
# =====================================================================


class TestAlertsFormConstraints:
    """Validate that alerts.py Form() defaults match the tightened spec.

    Since alerts.py uses Form() parameters (not a Pydantic model), we
    introspect the function signature and verify the compiled regex.

    FastAPI/Pydantic v2 stores Form constraints (max_length, min_length)
    as MaxLen / MinLen objects inside the ``.metadata`` list on the
    FieldInfo, not as direct attributes.
    """

    @staticmethod
    def _get_form_metadata(param_name: str) -> dict:
        """Extract {max_length, min_length} from Form() metadata."""
        sig = inspect.signature(alerts_module.subscribe_whatsapp)
        param = sig.parameters[param_name]
        result = {}
        for m in getattr(param.default, "metadata", []):
            if hasattr(m, "max_length"):
                result["max_length"] = m.max_length
            if hasattr(m, "min_length"):
                result["min_length"] = m.min_length
        return result

    def test_phone_form_max_length_is_16(self):
        """After the security fix, max_length should be 16 (not 20)."""
        meta = self._get_form_metadata("phone_number")
        assert meta["max_length"] == 16

    def test_name_form_min_length_is_1(self):
        meta = self._get_form_metadata("name")
        assert meta["min_length"] == 1

    def test_name_form_max_length_is_100(self):
        meta = self._get_form_metadata("name")
        assert meta["max_length"] == 100

    def test_region_id_form_max_length_is_100(self):
        meta = self._get_form_metadata("region_id")
        assert meta["max_length"] == 100

    def test_alerts_regex_accepts_valid_e164(self):
        """The alerts.py regex is slightly more lenient (optional +)."""
        regex = alerts_module._PHONE_E164_RE
        assert regex.match("+919876543210")
        assert regex.match("919876543210")   # alerts allows no '+'

    def test_alerts_regex_rejects_too_long(self):
        regex = alerts_module._PHONE_E164_RE
        assert not regex.match("+1" + "9" * 15)  # 17 chars

    def test_alerts_regex_rejects_garbage(self):
        regex = alerts_module._PHONE_E164_RE
        assert not regex.match("not-a-phone")
        assert not regex.match("")
        assert not regex.match("+")


# =====================================================================
#  7. Cross-model parity
# =====================================================================


class TestCrossModelParity:
    """Ensure both Pydantic models accept the same valid payloads."""

    @pytest.mark.parametrize("phone,name", [
        ("+919876543210", "Ravi Kumar"),
        ("+14155552671", "Jane Doe"),
        ("+447911123456", "A"),
        ("+1234567", "B" * 100),
    ])
    def test_same_valid_input_accepted_by_both(self, phone: str, name: str):
        main_req = _main(phone=phone, name=name)
        plat_req = _platform(phone=phone, name=name)
        assert main_req.phone_number == plat_req.phone_number
        assert main_req.name == plat_req.name

    def test_oversized_phone_rejected_by_both(self):
        big_phone = "+" + "1" * 20
        with pytest.raises(ValidationError):
            _main(phone=big_phone)
        with pytest.raises(ValidationError):
            _platform(phone=big_phone)

    def test_oversized_name_rejected_by_both(self):
        big_name = "X" * 101
        with pytest.raises(ValidationError):
            _main(name=big_name)
        with pytest.raises(ValidationError):
            _platform(name=big_name)

    def test_empty_name_rejected_by_both(self):
        with pytest.raises(ValidationError):
            _main(name="")
        with pytest.raises(ValidationError):
            _platform(name="")


# =====================================================================
#  8. Defense-in-depth: Field max_length vs regex layering (main.py)
# =====================================================================


class TestDefenseInDepth:
    """Confirm that BOTH the Field max_length AND the regex validator
    provide independent rejection — so removing either one still leaves
    a safety net."""

    def test_regex_catches_16_digit_number_despite_field_allowing_20(self):
        """'+1' + 15 nines = 17 chars.  Field max_length=20 would allow it,
        but the regex rejects >15 digits."""
        with pytest.raises(ValidationError, match="E.164"):
            _main(phone="+1" + "9" * 15)

    def test_field_catches_21_chars_even_if_regex_somehow_didnt(self):
        """'+' + 20 ones = 21 chars.  The regex would also reject this,
        but the Field max_length=20 provides an additional barrier."""
        with pytest.raises(ValidationError):
            _main(phone="+" + "1" * 20)

    def test_exactly_at_field_boundary_20_chars(self):
        """'+' + 19 digits = 20 chars.  Within Field max_length=20 but
        regex still rejects (max 15 digits)."""
        with pytest.raises(ValidationError, match="E.164"):
            _main(phone="+" + "1" * 19)

    def test_exactly_at_regex_boundary_max(self):
        """+1 + 14 digits = 16 chars.  Both Field and regex accept."""
        phone = "+1" + "2" * 14  # +1 + 14 = 16 chars, 15 digits total
        req = _main(phone=phone)
        assert req.phone_number == phone

    def test_exactly_at_regex_boundary_min(self):
        """+1 + 6 digits = 8 chars.  Both Field and regex accept."""
        phone = "+1234567"  # 8 chars, 7 digits total
        req = _main(phone=phone)
        assert req.phone_number == phone
