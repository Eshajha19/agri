"""
role_sync.py — Firebase custom-claim synchronisation for role-based rules.

Firestore security rules now use request.auth.token.role (a custom claim
embedded in the JWT) instead of calling get() on the users document on
every rule evaluation.  This module provides the single function that must
be called whenever a user's role changes so the JWT claim stays in sync
with the Firestore users/{uid}.role field.

Usage
-----
    from role_sync import sync_role_claim

    # After writing role to Firestore:
    await sync_role_claim(uid, new_role)

The function is intentionally synchronous-safe (runs the Firebase Admin
SDK call in a thread-pool executor when called from async context) and
idempotent — calling it with the same role twice is harmless.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Valid roles — must match the values used in firestore.rules and rbac.py.
VALID_ROLES = frozenset({"admin", "expert", "farmer", "vendor", "system", "guest"})


def _validate_uid(uid: str) -> None:
    """Raise ValueError if uid is not a non-empty string."""
    if not isinstance(uid, str) or not uid.strip():
        raise ValueError(
            f"uid must be a non-empty string, got {type(uid).__name__!r}"
        )


def _set_claim_sync(uid: str, role: str) -> None:
    """Blocking call to Firebase Admin SDK.  Run in a thread-pool from async code."""
    import firebase_admin
    from firebase_admin import auth as firebase_auth

    if not firebase_admin._apps:
        raise RuntimeError("Firebase Admin SDK is not initialised")

    firebase_auth.set_custom_user_claims(uid, {"role": role})
    logger.info("Custom claim set: uid=%s role=%s", uid, role)


def _revoke_refresh_tokens_sync(uid: str) -> None:
    """Invalidate outstanding refresh tokens so ID tokens are re-issued promptly."""
    import firebase_admin
    from firebase_admin import auth as firebase_auth

    if not firebase_admin._apps:
        raise RuntimeError("Firebase Admin SDK is not initialised")

    firebase_auth.revoke_refresh_tokens(uid)
    logger.info("Refresh tokens revoked: uid=%s", uid)


async def sync_role_claim(uid: str, role: str, *, revoke_sessions: bool = True) -> None:
    """
    Set the 'role' custom claim on the Firebase Auth user identified by uid.

    This must be called after every role change so that Firestore security
    rules (which read request.auth.token.role) stay consistent with the
    Firestore users/{uid}.role field.

    By default this also revokes refresh tokens so the user must sign in again
    and cannot keep using a stale JWT with the previous ``role`` claim (closes
    the demotion / privilege-escalation window documented in issue #1130).

    Authoritative role source for the API is Firestore ``users/{uid}.role``.
    Custom claims are a mirror for Firestore security rules only.

    Raises ValueError if uid is not a non-empty string or role is invalid.
    Raises RuntimeError if Firebase Admin is not initialised.
    Raises firebase_admin.auth.UserNotFoundError if uid does not exist.
    """
    _validate_uid(uid)
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role '{role}'. Must be one of: {sorted(VALID_ROLES)}")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _set_claim_sync, uid, role)
    if revoke_sessions:
        await loop.run_in_executor(None, _revoke_refresh_tokens_sync, uid)


def sync_role_claim_sync(uid: str, role: str, *, revoke_sessions: bool = True) -> None:
    """
    Synchronous variant for use in non-async contexts (e.g. startup scripts,
    Celery tasks, or test fixtures).

    Raises ValueError if uid is not a non-empty string or role is invalid.
    """
    _validate_uid(uid)
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role '{role}'. Must be one of: {sorted(VALID_ROLES)}")
    _set_claim_sync(uid, role)
    if revoke_sessions:
        _revoke_refresh_tokens_sync(uid)


def backfill_role_claims(db_client, batch_size: int = 100) -> dict:
    """
    One-time backfill: read every user document from Firestore and set the
    'role' custom claim on the corresponding Firebase Auth user.

    Call this once after deploying the rules change to ensure all existing
    users have the claim set.  Safe to call multiple times — idempotent.

    Returns a summary dict: {"processed": N, "errors": M, "skipped": K}
    """
    import firebase_admin
    from firebase_admin import auth as firebase_auth

    if not firebase_admin._apps:
        raise RuntimeError("Firebase Admin SDK is not initialised")

    processed = 0
    errors = 0
    skipped = 0

    try:
        docs = db_client.collection("users").stream()
        for doc in docs:
            try:
                _validate_uid(doc.id)
            except ValueError:
                logger.warning(
                    "backfill: invalid uid %r in Firestore document — skipping", doc.id
                )
                skipped += 1
                continue

            data = doc.to_dict() or {}
            role = data.get("role", "farmer")
            if role not in VALID_ROLES:
                logger.warning(
                    "backfill: uid=%s has unknown role '%s', defaulting to 'farmer'",
                    doc.id, role,
                )
                role = "farmer"
            try:
                firebase_auth.set_custom_user_claims(doc.id, {"role": role})
                processed += 1
            except firebase_admin.auth.UserNotFoundError:
                # Firestore document exists but the Auth user was deleted.
                logger.warning("backfill: Auth user not found for uid=%s — skipping", doc.id)
                skipped += 1
            except Exception as exc:
                logger.error("backfill: failed to set claim for uid=%s: %s", doc.id, exc)
                errors += 1
    except Exception as exc:
        logger.error("backfill: Firestore stream failed: %s", exc)
        raise

    logger.info(
        "backfill complete: processed=%d errors=%d skipped=%d",
        processed, errors, skipped,
    )
    return {"processed": processed, "errors": errors, "skipped": skipped}