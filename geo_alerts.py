"""Shared helpers for region-scoped alerts and subscriptions."""

from __future__ import annotations

import re as _re
from collections.abc import Iterable, Mapping
from typing import Any, Optional, Set


_PROFILE_REGION_KEYS = (
    "region_id",
    "region",
    "state",
    "district",
    "managed_regions",
    "allowed_regions",
    "regions",
)

_NOTIFICATION_REGION_KEYS = (
    "region_id",
    "region_ids",
    "region",
    "regions",
    "state",
    "district",
)

# ---------------------------------------------------------------------------
# Region ID validation
# ---------------------------------------------------------------------------

# A valid region token contains only lowercase alphanumeric characters,
# hyphens, and colons (the canonical hierarchy separator).
# It must start and end with an alphanumeric character.
_VALID_REGION_RE = _re.compile(r'^[a-z0-9]+([:\-][a-z0-9]+)*$')


def is_valid_region_identifier(value: str) -> bool:
    """Return True when *value* is a non-empty, well-formed region token.

    A valid token:
    - is non-empty
    - contains only lowercase letters, digits, hyphens, and colons
    - starts and ends with an alphanumeric character
    - is not the bare wildcard ``*`` (callers must handle that explicitly)

    Rejects inputs like ``"--"``, ``"::"``, ``" "``, ``"*"``, or anything
    that normalizes to an empty string.
    """
    if not value:
        return False
    if value == "*":
        # Wildcard is handled explicitly by privileged callers; it is not a
        # valid user-supplied region identifier.
        return False
    return bool(_VALID_REGION_RE.match(value))


def normalize_region_identifier(value: Any) -> str:
    """Return a canonical region token for matching.

    Returns an empty string for any input that cannot be normalized into a
    valid token (e.g. ``None``, ``""``, ``"---"``, ``"::"``, ``"*"``).
    Callers that need to distinguish *missing* from *invalid* should check
    ``is_valid_region_identifier`` on the non-empty result.
    """
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    for old, new in ((" ", "-"), ("_", "-"), ("/", ":")):
        text = text.replace(old, new)
    while "::" in text:
        text = text.replace("::", ":")
    while "--" in text:
        text = text.replace("--", "-")
    text = text.strip(":-")
    # After normalization the token must still be non-empty and structurally
    # valid; if it isn't (e.g. input was "---" or "::") return "" so callers
    # treat it identically to a missing value.
    if not text or not is_valid_region_identifier(text):
        return ""
    return text


def _extract_regions(value: Any) -> Set[str]:
    regions: Set[str] = set()
    if value is None:
        return regions
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            normalized = normalize_region_identifier(item)
            if normalized:
                regions.add(normalized)
        return regions
    normalized = normalize_region_identifier(value)
    if normalized:
        regions.add(normalized)
    return regions


def _extract_prefixed_regions(prefix: str, value: Any) -> Set[str]:
    regions: Set[str] = set()
    if value is None:
        return regions
    values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
    for item in values:
        normalized = normalize_region_identifier(item)
        if normalized:
            regions.add(normalized)
            regions.add(f"{prefix}:{normalized}")
    return regions


def profile_regions(profile: Mapping[str, Any] | None) -> Set[str]:
    """Collect region scopes declared on a Firestore user profile."""
    if not profile:
        return set()

    regions: Set[str] = set()
    for key in _PROFILE_REGION_KEYS:
        value = profile.get(key)
        if key == "state" and value is not None:
            regions.update(_extract_prefixed_regions("state", value))
        elif key == "district" and value is not None:
            regions.update(_extract_prefixed_regions("district", value))
        else:
            regions.update(_extract_regions(value))
    return regions


def notification_regions(notification: Mapping[str, Any] | None) -> Set[str]:
    """Collect region scopes declared on a notification payload."""
    if not notification:
        return set()

    regions: Set[str] = set()
    for key in _NOTIFICATION_REGION_KEYS:
        value = notification.get(key)
        if key == "state" and value is not None:
            regions.update(_extract_prefixed_regions("state", value))
        elif key == "district" and value is not None:
            regions.update(_extract_prefixed_regions("district", value))
        else:
            regions.update(_extract_regions(value))
    return regions


def region_matches(subscription_region: str, notification_region: str) -> bool:
    """Return True when the subscription scope covers the notification scope.

    Wildcards (``"*"``) are checked on the raw input before normalization
    because ``normalize_region_identifier("*")`` returns ``""`` — the
    wildcard is not a valid user-supplied identifier but IS a valid internal
    scope token assigned by ``resolve_subscription_regions`` to admins/experts.
    """
    # Check wildcards on raw input before normalization strips them.
    if str(subscription_region).strip() == "*" or str(notification_region).strip() == "*":
        return True
    subscription_region = normalize_region_identifier(subscription_region)
    notification_region = normalize_region_identifier(notification_region)
    if not subscription_region or not notification_region:
        return False
    if subscription_region == notification_region:
        return True
    return notification_region.startswith(subscription_region + ":")


def notification_matches_regions(
    notification: Mapping[str, Any], regions: Iterable[str] | None
) -> bool:
    """Return True if a notification matches at least one requested region."""
    normalized_regions = {
        normalize_region_identifier(region)
        for region in (regions or [])
        if normalize_region_identifier(region)
    }
    if not normalized_regions:
        return not notification_regions(notification)

    targets = notification_regions(notification)
    if not targets:
        return True

    for requested in normalized_regions:
        if requested == "*":
            return True
        for target in targets:
            if region_matches(requested, target):
                return True
    return False


def profile_can_broadcast_region(
    profile: Mapping[str, Any] | None, region_id: str
) -> bool:
    """Return True when a profile is allowed to broadcast to a region."""
    if not region_id:
        return False

    normalized_region = normalize_region_identifier(region_id)
    if not normalized_region:
        return False

    role = normalize_region_identifier((profile or {}).get("role"))
    if role in {"admin", "expert"}:
        return True

    for owned_region in profile_regions(profile):
        if region_matches(owned_region, normalized_region):
            return True
    return False


def resolve_subscription_regions(
    profile: Mapping[str, Any] | None,
    requested_regions: Iterable[str] | None = None,
) -> Set[str]:
    """Resolve the websocket subscription scopes for a profile.

    Security contract
    -----------------
    * Admin and expert roles may subscribe to any scope (including ``*``).
    * All other callers are constrained to regions their profile owns.
    * A caller with **no owned regions** and **no privileged role** receives
      an **empty set** regardless of what they requested — granting unowned
      regions would allow arbitrary scope escalation via a crafted payload.
    * Invalid or empty region tokens are always discarded before the
      authorization intersection so they cannot broaden the result.
    """
    # Discard any token that does not normalize to a valid identifier.
    requested: Set[str] = {
        normalized
        for normalized in (
            normalize_region_identifier(region)
            for region in (requested_regions or [])
        )
        if normalized
    }

    role = normalize_region_identifier((profile or {}).get("role"))
    if role in {"admin", "expert"}:
        return {"*"} if not requested else requested | {"*"}

    owned_regions = profile_regions(profile)

    # *** Security fix: no owned regions → no subscription authority. ***
    # Previously this fell through to ``return requested or owned_regions``
    # which would have granted all requested regions to an unprivileged user
    # with an empty profile, potentially broadening their subscription scope.
    if not owned_regions:
        return set()

    if requested:
        # Only grant the intersection: scopes the caller actually owns.
        return {
            region
            for region in requested
            if any(region_matches(owned, region) for owned in owned_regions)
        }

    # No explicit request → fall back to the full owned set.
    return owned_regions