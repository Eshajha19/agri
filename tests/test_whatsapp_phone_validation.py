"""Tests for WhatsAppSubscribeRequest E.164 phone number validation.

Verifies that:
- Valid E.164 numbers are accepted.
- Numbers missing the + prefix are rejected.
- Numbers with spaces, dashes, or parentheses are rejected.
- Numbers that are too short or too long are rejected.
- Arbitrary strings are rejected.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# Import directly from main to test the actual model used by the endpoint.
from main import WhatsAppSubscribeRequest


def _make(phone: str) -> WhatsAppSubscribeRequest:
    return WhatsAppSubscribeRequest(phone_number=phone, name="Test Farmer")


# ---------------------------------------------------------------------------
# Valid numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phone",
    [
        "+919876543210",   # India (12 chars)
        "+14155552671",    # US (11 chars)
        "+447911123456",   # UK (12 chars)
        "+12125551234",    # US with area code
        "+85212345678",    # Hong Kong (11 chars)
        "+1234567",        # minimum valid: +1 followed by 6 digits (7 total digits)
    ],
)
def test_valid_e164_numbers(phone: str) -> None:
    req = _make(phone)
    assert req.phone_number == phone


# ---------------------------------------------------------------------------
# Missing or wrong prefix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phone",
    [
        "919876543210",     # missing + prefix
        "0919876543210",    # leading zero instead of +
        "00919876543210",   # double-zero prefix
    ],
)
def test_rejects_missing_plus_prefix(phone: str) -> None:
    with pytest.raises(ValidationError, match="E.164"):
        _make(phone)


# ---------------------------------------------------------------------------
# Formatting characters not allowed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phone",
    [
        "+91 98765 43210",     # spaces
        "+91-9876-543210",     # dashes
        "+91(98765)43210",     # parentheses
        "+91.9876.543210",     # dots
    ],
)
def test_rejects_formatted_numbers(phone: str) -> None:
    with pytest.raises(ValidationError, match="E.164"):
        _make(phone)


# ---------------------------------------------------------------------------
# Length constraints
# ---------------------------------------------------------------------------


def test_rejects_too_short() -> None:
    # + followed by only 5 digits -- fewer than the minimum 6 after country code
    with pytest.raises(ValidationError):
        _make("+12345")


def test_rejects_too_long() -> None:
    # 17-character string exceeds max_length=16
    with pytest.raises(ValidationError):
        _make("+1" + "9" * 15)


# ---------------------------------------------------------------------------
# Arbitrary / injection strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phone",
    [
        "",
        "not-a-phone",
        "+",
        "++919876543210",
        "+91987654321012345678",   # way too long
        "'; DROP TABLE subscribers; --",
    ],
)
def test_rejects_arbitrary_strings(phone: str) -> None:
    with pytest.raises(ValidationError):
        _make(phone)


# ---------------------------------------------------------------------------
# name field bounds (added as part of this fix)
# ---------------------------------------------------------------------------


def test_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        WhatsAppSubscribeRequest(phone_number="+919876543210", name="")


def test_rejects_name_over_100_chars() -> None:
    with pytest.raises(ValidationError):
        WhatsAppSubscribeRequest(phone_number="+919876543210", name="A" * 101)
