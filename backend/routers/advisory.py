"""Rule-based farmer advisory API."""
import threading
from collections import defaultdict, deque
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from advisory_rules import generate_advisories


router = APIRouter()
_MAX_STORED_ALERTS = 50
_stored_alerts: dict[str, deque] = defaultdict(lambda: deque(maxlen=_MAX_STORED_ALERTS))
_store_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Dependency injection — wired in main.py lifespan
# ---------------------------------------------------------------------------
_verify_role_fn = None


def init_advisory(verify_role_fn) -> None:
    global _verify_role_fn
    _verify_role_fn = verify_role_fn


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AdvisoryRequest(BaseModel):
    weather: dict[str, Any] = Field(default_factory=dict)
    soil: dict[str, Any] = Field(default_factory=dict)
    crop_type: Optional[str] = Field(default=None, max_length=50)
    # user_id is no longer accepted from the request body.
    # The authoritative identity is always derived from the verified
    # Firebase ID token so a caller cannot store alerts under another
    # user's UID.
    store_alerts: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/advisory")
async def create_advisory(payload: AdvisoryRequest, request: Request):
    """
    Generate rule-based farm advisories for the authenticated user.

    If store_alerts is True the generated alerts are persisted server-side
    under the caller's verified Firebase UID — never under a client-supplied
    user_id — so they can be retrieved later via GET /advisory/me.

    Authentication is required when store_alerts is True so that:
    1. Alerts are always bound to a verified identity.
    2. An unauthenticated caller cannot pollute another user's alert store
       by guessing or enumerating Firebase UIDs (IDOR).

    Unauthenticated callers may still generate transient advisories
    (store_alerts=False) for the climate simulator and public widgets.
    """
    alerts = generate_advisories(
        weather=payload.weather,
        soil=payload.soil,
        crop_type=payload.crop_type,
    )

    stored = False
    if payload.store_alerts:
        if _verify_role_fn is None:
            raise HTTPException(status_code=500, detail="Advisory service not initialized")

        # Derive uid from the verified token — never from the request body.
        token_data = await _verify_role_fn(request)
        uid = token_data.get("uid")

        with _store_lock:
            _stored_alerts[uid].extend(alerts)
        stored = True

    return {
        "success": True,
        "data": alerts,
        "count": len(alerts),
        "stored": stored,
    }


@router.get("/advisory/me")
async def get_my_advisories(request: Request):
    """
    Return stored advisories for the authenticated caller.

    Authentication is required — a caller can only read their own stored
    alerts. The previous GET /advisory/{user_id} endpoint accepted any
    user_id as a path parameter with no token check, allowing any caller
    to read another user's alerts (IDOR).

    The endpoint is now /advisory/me so the caller's identity is always
    derived from the verified Firebase token, not from a URL parameter.
    """
    if _verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Advisory service not initialized")

    token_data = await _verify_role_fn(request)
    uid = token_data.get("uid")

    with _store_lock:
        data = list(_stored_alerts.get(uid, []))

    return {"success": True, "data": data}
