"""
Unit tests for region normalization, matching, and subscription security.

Covers all three acceptance criteria from issue #2370:

  AC-1  Unit tests for normalization of malformed region inputs.
  AC-2  Ensure WS subscribe_crops/regions and HTTP endpoints behave identically.
  AC-3  Tests ensuring unauthorized broadcast regions can't be subscribed.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from geo_alerts import (
    is_valid_region_identifier,
    normalize_region_identifier,
    profile_can_broadcast_region,
    profile_regions,
    region_matches,
    notification_matches_regions,
    resolve_subscription_regions,
)


# ═══════════════════════════════════════════════════════════════════════════════
# AC-1 — Normalization of malformed region inputs
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsValidRegionIdentifier:
    """is_valid_region_identifier must reject all degenerate tokens."""

    @pytest.mark.parametrize("token", [
        "",          # empty
        " ",         # whitespace only
        "-",         # bare hyphen
        ":",         # bare colon
        "--",        # double hyphen
        "::",        # double colon
        "-abc",      # leading hyphen
        "abc-",      # trailing hyphen
        ":abc",      # leading colon
        "abc:",      # trailing colon
        "ABC",       # uppercase not allowed (tokens are lowercased, raw uppercase is invalid)
        "a b",       # space
        "*",         # wildcard must be rejected as a user identifier
        "a/b",       # slash (use colon for hierarchy)
        "a_b",       # underscore (use hyphen)
    ])
    def test_invalid_tokens_rejected(self, token: str) -> None:
        assert is_valid_region_identifier(token) is False, f"Expected {token!r} to be invalid"

    @pytest.mark.parametrize("token", [
        "punjab",
        "state-punjab",
        "state:punjab",
        "state:punjab:ludhiana",
        "in",
        "region1",
        "a",
    ])
    def test_valid_tokens_accepted(self, token: str) -> None:
        assert is_valid_region_identifier(token) is True, f"Expected {token!r} to be valid"


class TestNormalizeRegionIdentifier:
    """normalize_region_identifier must return '' for any invalid/empty input."""

    @pytest.mark.parametrize("raw,expected", [
        (None,          ""),
        ("",            ""),
        ("   ",         ""),
        ("---",         ""),
        ("::",          ""),
        ("--abc--",     "abc"),        # strip leading/trailing junk
        ("Punjab",      "punjab"),     # lowercase
        ("state_up",    "state-up"),   # underscore → hyphen
        ("state/up",    "state:up"),   # slash → colon
        ("state::up",   "state:up"),   # collapse double colon
        ("state--up",   "state-up"),   # collapse double hyphen
        ("  ludhiana ", "ludhiana"),   # strip whitespace
        ("*",           ""),           # wildcard is not a valid user-supplied identifier
        (123,           "123"),        # numeric coercion
        (["a"],         ""),           # list is not a scalar — returns ""
    ])
    def test_normalization(self, raw, expected: str) -> None:
        assert normalize_region_identifier(raw) == expected

    def test_empty_string_never_broadens_scope(self) -> None:
        """An empty normalized token must never match anything."""
        assert region_matches("", "punjab") is False
        assert region_matches("punjab", "") is False
        assert region_matches("", "") is False


# ═══════════════════════════════════════════════════════════════════════════════
# AC-1 (continued) — region_matches edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegionMatches:
    def test_exact_match(self) -> None:
        assert region_matches("punjab", "punjab") is True

    def test_prefix_hierarchy(self) -> None:
        assert region_matches("state:punjab", "state:punjab:ludhiana") is True

    def test_no_false_prefix(self) -> None:
        # "state:pun" must NOT match "state:punjab"
        assert region_matches("state:pun", "state:punjab") is False

    def test_wildcard_subscription_matches_anything(self) -> None:
        assert region_matches("*", "state:punjab") is True

    def test_wildcard_notification_matches_anything(self) -> None:
        assert region_matches("state:punjab", "*") is True

    def test_empty_subscription_matches_nothing(self) -> None:
        assert region_matches("", "state:punjab") is False

    def test_empty_notification_matches_nothing(self) -> None:
        assert region_matches("state:punjab", "") is False

    def test_malformed_raw_input_rejected(self) -> None:
        # Raw malformed tokens must be normalized to "" before comparison
        assert region_matches("---", "punjab") is False
        assert region_matches("punjab", "---") is False


# ═══════════════════════════════════════════════════════════════════════════════
# AC-1 (continued) — notification_matches_regions
# ═══════════════════════════════════════════════════════════════════════════════

class TestNotificationMatchesRegions:
    def test_empty_regions_matches_broadcast_notifications(self) -> None:
        """No region filter → match only notifications with no region scope."""
        assert notification_matches_regions({}, None) is True
        assert notification_matches_regions({}, []) is True

    def test_empty_regions_does_not_match_scoped_notification(self) -> None:
        notif = {"region_id": "punjab"}
        assert notification_matches_regions(notif, None) is False

    def test_malformed_region_in_filter_is_discarded(self) -> None:
        """Invalid tokens in the filter set must be silently discarded,
        leaving an empty filter that matches broadcast-only notifications."""
        notif_broadcast = {}
        notif_scoped = {"region_id": "punjab"}
        # All provided tokens are malformed — they normalize to ""
        assert notification_matches_regions(notif_broadcast, ["---", "::"]) is True
        assert notification_matches_regions(notif_scoped, ["---", "::"]) is False

    def test_valid_region_filter_matches_correct_notification(self) -> None:
        notif = {"region_id": "state:punjab"}
        assert notification_matches_regions(notif, ["state:punjab"]) is True

    def test_valid_region_filter_no_match(self) -> None:
        notif = {"region_id": "state:maharashtra"}
        assert notification_matches_regions(notif, ["state:punjab"]) is False


# ═══════════════════════════════════════════════════════════════════════════════
# AC-3 — Unauthorized broadcast regions can't be subscribed
# ═══════════════════════════════════════════════════════════════════════════════

class TestResolveSubscriptionRegions:
    """resolve_subscription_regions must enforce ownership, not just pass-through."""

    def test_no_profile_no_regions(self) -> None:
        """An unauthenticated / profileless user gets nothing."""
        assert resolve_subscription_regions(None) == set()
        assert resolve_subscription_regions(None, ["state:punjab"]) == set()
        assert resolve_subscription_regions({}) == set()

    def test_no_owned_regions_cannot_request_any(self) -> None:
        """A farmer with no regions on their profile must not get requested regions."""
        profile = {"role": "farmer"}  # no region keys
        result = resolve_subscription_regions(profile, ["state:punjab", "state:haryana"])
        assert result == set(), (
            "A user with no owned regions must not be granted any requested scope"
        )

    def test_farmer_limited_to_owned_regions(self) -> None:
        profile = {"role": "farmer", "region_id": "state:punjab"}
        result = resolve_subscription_regions(profile, ["state:punjab"])
        assert result == {"state:punjab"}

    def test_farmer_cannot_subscribe_unowned_region(self) -> None:
        profile = {"role": "farmer", "region_id": "state:punjab"}
        result = resolve_subscription_regions(profile, ["state:maharashtra"])
        assert result == set(), (
            "Farmer must not receive a region they don't own"
        )

    def test_farmer_partial_intersection(self) -> None:
        """Only owned regions from the requested set are granted."""
        profile = {"role": "farmer", "regions": ["state:punjab", "state:haryana"]}
        result = resolve_subscription_regions(
            profile, ["state:punjab", "state:maharashtra"]
        )
        assert result == {"state:punjab"}

    def test_admin_gets_wildcard(self) -> None:
        profile = {"role": "admin"}
        result = resolve_subscription_regions(profile)
        assert "*" in result

    def test_expert_gets_wildcard_plus_requested(self) -> None:
        profile = {"role": "expert"}
        result = resolve_subscription_regions(profile, ["state:punjab"])
        assert "*" in result
        assert "state:punjab" in result

    def test_malformed_requested_regions_discarded(self) -> None:
        """Invalid tokens in the requested list must not widen the scope."""
        profile = {"role": "farmer", "region_id": "state:punjab"}
        result = resolve_subscription_regions(profile, ["---", "::", "state:punjab"])
        assert result == {"state:punjab"}
        # Malformed tokens must not appear in the result
        assert "---" not in result
        assert "::" not in result

    def test_empty_string_requested_region_discarded(self) -> None:
        profile = {"role": "farmer", "region_id": "state:punjab"}
        result = resolve_subscription_regions(profile, [""])
        # Empty string after normalization → treated as no request → return owned
        assert result == {"state:punjab"}

    def test_wildcard_requested_by_farmer_not_granted(self) -> None:
        """A farmer requesting '*' must not receive wildcard authority."""
        profile = {"role": "farmer", "region_id": "state:punjab"}
        result = resolve_subscription_regions(profile, ["*"])
        # "*" normalizes to "" (invalid for user-supplied identifiers)
        # so the requested set becomes empty → falls back to owned regions
        assert "*" not in result
        assert result == {"state:punjab"}


class TestProfileCanBroadcastRegion:
    def test_admin_can_broadcast_anywhere(self) -> None:
        assert profile_can_broadcast_region({"role": "admin"}, "state:punjab") is True

    def test_expert_can_broadcast_anywhere(self) -> None:
        assert profile_can_broadcast_region({"role": "expert"}, "state:kerala") is True

    def test_farmer_can_broadcast_owned_region(self) -> None:
        profile = {"role": "farmer", "region_id": "state:punjab"}
        assert profile_can_broadcast_region(profile, "state:punjab") is True

    def test_farmer_cannot_broadcast_unowned_region(self) -> None:
        profile = {"role": "farmer", "region_id": "state:punjab"}
        assert profile_can_broadcast_region(profile, "state:maharashtra") is False

    def test_empty_region_id_rejected(self) -> None:
        assert profile_can_broadcast_region({"role": "admin"}, "") is False
        assert profile_can_broadcast_region({"role": "admin"}, None) is False

    def test_malformed_region_id_rejected(self) -> None:
        """'---' normalizes to '' → must be rejected even for admin."""
        assert profile_can_broadcast_region({"role": "admin"}, "---") is False

    def test_no_profile_cannot_broadcast(self) -> None:
        assert profile_can_broadcast_region(None, "state:punjab") is False


# ═══════════════════════════════════════════════════════════════════════════════
# AC-2 — WS subscribe_regions validation parity with HTTP
# ═══════════════════════════════════════════════════════════════════════════════

class TestWSSubscribeRegionsValidation:
    """
    The WS subscribe_regions message must validate identically to the HTTP
    region_id field: only accept structurally valid, non-empty region tokens.

    These tests exercise the _validate_subscribe_regions static method that
    was added to NotificationBroadcastHub.
    """

    @pytest.fixture
    def validate(self):
        from realtime_notifications import NotificationBroadcastHub
        return NotificationBroadcastHub._validate_subscribe_regions

    def test_valid_message_accepted(self, validate) -> None:
        ok, err = validate({"type": "subscribe_regions", "regions": ["state:punjab"]})
        assert ok is True
        assert err == ""

    def test_multiple_valid_regions_accepted(self, validate) -> None:
        ok, err = validate({"type": "subscribe_regions", "regions": ["state:punjab", "state:haryana"]})
        assert ok is True

    def test_empty_regions_list_rejected(self, validate) -> None:
        ok, err = validate({"type": "subscribe_regions", "regions": []})
        assert ok is False

    def test_too_many_regions_rejected(self, validate) -> None:
        ok, err = validate({"type": "subscribe_regions", "regions": ["r"] * 51})
        assert ok is False

    def test_non_string_region_rejected(self, validate) -> None:
        ok, err = validate({"type": "subscribe_regions", "regions": [123]})
        assert ok is False

    def test_empty_string_region_rejected(self, validate) -> None:
        ok, err = validate({"type": "subscribe_regions", "regions": [""]})
        assert ok is False

    def test_unknown_keys_rejected(self, validate) -> None:
        ok, err = validate({"type": "subscribe_regions", "regions": ["punjab"], "extra": "x"})
        assert ok is False

    def test_missing_regions_key_rejected(self, validate) -> None:
        ok, err = validate({"type": "subscribe_regions"})
        assert ok is False


class TestWSSubscribeCropsValidation:
    """subscribe_crops must also enforce the same structural rules."""

    @pytest.fixture
    def validate(self):
        from realtime_notifications import NotificationBroadcastHub
        return NotificationBroadcastHub._validate_subscribe_crops

    def test_valid_message_accepted(self, validate) -> None:
        ok, err = validate({"type": "subscribe_crops", "crops": ["wheat", "rice"]})
        assert ok is True

    def test_empty_crops_list_rejected(self, validate) -> None:
        ok, err = validate({"type": "subscribe_crops", "crops": []})
        assert ok is False

    def test_non_string_crop_rejected(self, validate) -> None:
        ok, err = validate({"type": "subscribe_crops", "crops": [None]})
        assert ok is False

    def test_unknown_keys_rejected(self, validate) -> None:
        ok, err = validate({"type": "subscribe_crops", "crops": ["wheat"], "foo": "bar"})
        assert ok is False


# ═══════════════════════════════════════════════════════════════════════════════
# AC-2 — HTTP trigger-alert region_id parity
# ═══════════════════════════════════════════════════════════════════════════════

class TestHTTPRegionNormalizationParity:
    """
    The HTTP trigger-alert and subscribe endpoints must reject non-empty
    region_ids that normalize to an empty string — the same behaviour the
    WS path enforces through _validate_subscribe_regions.
    """

    @pytest.mark.parametrize("bad_region", ["---", "::", "-:-", "   ", "*"])
    def test_malformed_region_normalizes_to_empty(self, bad_region: str) -> None:
        """Any input that the HTTP layer would reject must normalize to ''."""
        assert normalize_region_identifier(bad_region) == "", (
            f"Expected {bad_region!r} to normalize to '' so the HTTP layer rejects it"
        )

    @pytest.mark.parametrize("good_region", [
        "punjab", "state:punjab", "state:punjab:ludhiana", "in", "region1"
    ])
    def test_valid_region_survives_normalization(self, good_region: str) -> None:
        result = normalize_region_identifier(good_region)
        assert result != "", f"Expected {good_region!r} to survive normalization"
        assert is_valid_region_identifier(result), (
            f"Normalized form {result!r} must be a valid identifier"
        )