"""Farmer referral and village growth router."""
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from firebase_admin import auth, firestore

router = APIRouter()

get_db_fn = None


class RedeemReferralRequest(BaseModel):
    referral_code: str = Field(..., min_length=4, max_length=32)


def init_referrals(db_resolver):
    global get_db_fn
    get_db_fn = db_resolver


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _normalize_referral_code(code: str) -> str:
    if not code:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(code).upper().strip())


def _generate_referral_code(uid: str, attempt: int = 0) -> str:
    import hashlib

    digest = hashlib.sha256(f"{uid}:{attempt}".encode("utf-8")).hexdigest().upper()
    return f"FS{digest[:10]}"


def _referral_badge(referral_count: int) -> str:
    if referral_count >= 10:
        return "Village Mentor"
    if referral_count >= 5:
        return "Community Champion"
    if referral_count >= 3:
        return "Seed Builder"
    if referral_count >= 1:
        return "First Harvester"
    return "Starter"


def _referral_points_for_count(referral_count: int) -> int:
    return referral_count * 50


def _community_label(user_data: Optional[Dict[str, Any]]) -> str:
    if not user_data:
        return "Unknown village"
    return (
        user_data.get("villageName")
        or user_data.get("village")
        or user_data.get("address")
        or user_data.get("locationName")
        or "Unknown village"
    )


def _require_db():
    if get_db_fn is None:
        raise HTTPException(status_code=500, detail="Referral service not initialized")
    return get_db_fn()


async def _get_uid_from_request(request: Request) -> str:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")

    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing auth token")

    try:
        decoded = auth.verify_id_token(token, check_revoked=True)
        uid = decoded.get("uid")
        if not uid:
            raise HTTPException(status_code=401, detail="Invalid auth token")
        return uid
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid auth token")


def _ensure_user_referral_code(db, uid: str, user_data: Optional[Dict[str, Any]] = None) -> str:
    user_ref = db.collection("users").document(uid)
    data = user_data
    if data is None:
        user_snap = user_ref.get()
        data = user_snap.to_dict() if user_snap.exists else {}

    existing_code = _normalize_referral_code((data or {}).get("referralCode", ""))
    if existing_code:
        code_ref = db.collection("referral_codes").document(existing_code)
        code_snap = code_ref.get()
        if not code_snap.exists or code_snap.to_dict().get("uid") == uid:
            code_ref.set(
                {
                    "uid": uid,
                    "displayName": (data or {}).get("displayName") or "Farmer",
                    "updatedAt": _now_iso(),
                },
                merge=True,
            )
            if (data or {}).get("referralCode") != existing_code:
                user_ref.set(
                    {
                        "referralCode": existing_code,
                        "referralCodeIssuedAt": _now_iso(),
                    },
                    merge=True,
                )
            return existing_code

    for attempt in range(5):
        generated_code = _generate_referral_code(uid, attempt)
        code_ref = db.collection("referral_codes").document(generated_code)
        code_snap = code_ref.get()
        if code_snap.exists and code_snap.to_dict().get("uid") != uid:
            continue

        code_ref.set(
            {
                "uid": uid,
                "displayName": (data or {}).get("displayName") or "Farmer",
                "createdAt": _now_iso(),
                "updatedAt": _now_iso(),
            },
            merge=True,
        )
        user_ref.set(
            {
                "referralCode": generated_code,
                "referralCodeIssuedAt": _now_iso(),
            },
            merge=True,
        )
        return generated_code

    raise HTTPException(status_code=500, detail="Failed to generate a referral code")


def _history_entry(doc_snapshot) -> Dict[str, Any]:
    data = doc_snapshot.to_dict() if doc_snapshot else {}
    return {
        "id": getattr(doc_snapshot, "id", None),
        "inviteeUid": data.get("inviteeUid"),
        "inviteeName": data.get("inviteeName", "Farmer"),
        "inviteeLocation": data.get("inviteeLocation", "Unknown village"),
        "createdAt": data.get("createdAt"),
        "status": data.get("status", "redeemed"),
        "rewardPoints": data.get("rewardPoints", 0),
    }


def _leaderboards(db, limit: int = 10):
    leaders = []
    communities: Dict[str, Dict[str, Any]] = {}

    try:
        docs = (
            db.collection("users")
            .order_by("referralCount", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .get()
        )
    except Exception:
        docs = db.collection("users").get()
        docs = sorted(
            docs,
            key=lambda snap: int((snap.to_dict() or {}).get("referralCount", 0)),
            reverse=True,
        )[:limit]

    for doc_snapshot in docs:
        data = doc_snapshot.to_dict() or {}
        count = int(data.get("referralCount", 0) or 0)
        if count <= 0:
            continue

        community = _community_label(data)
        leaders.append(
            {
                # uid is intentionally omitted from the leaderboard payload.
                # Exposing Firebase UIDs in a public/unauthenticated response
                # enables account enumeration and targeted attacks against
                # specific users' Firestore documents.
                "displayName": data.get("displayName") or "Farmer",
                "referralCount": count,
                "referralPoints": int(data.get("referralPoints", _referral_points_for_count(count)) or 0),
                "referralBadge": data.get("referralBadge") or _referral_badge(count),
                "community": community,
            }
        )

        community_entry = communities.setdefault(
            community,
            {"community": community, "referrals": 0, "farmers": 0},
        )
        community_entry["referrals"] += count
        community_entry["farmers"] += 1

    community_board = sorted(communities.values(), key=lambda item: item["referrals"], reverse=True)[:limit]
    return leaders, community_board


@router.get("/dashboard")
async def get_referral_dashboard(request: Request):
    db = _require_db()
    uid = await _get_uid_from_request(request)

    user_ref = db.collection("users").document(uid)
    user_snap = user_ref.get()
    user_data = user_snap.to_dict() if user_snap.exists else {}

    referral_code = _ensure_user_referral_code(db, uid, user_data)

    referral_count = int((user_data or {}).get("referralCount", 0) or 0)
    referral_points = int((user_data or {}).get("referralPoints", _referral_points_for_count(referral_count)) or 0)
    referral_badge = (user_data or {}).get("referralBadge") or _referral_badge(referral_count)

    app_url = os.getenv("REFERRAL_APP_URL") or str(request.base_url).rstrip("/")
    referral_link = f"{app_url}/login?ref={referral_code}"

    history_docs = (
        db.collection("referrals")
        .where("inviterUid", "==", uid)
        .order_by("createdAt", direction=firestore.Query.DESCENDING)
        .limit(20)
        .get()
    )
    history = [_history_entry(item) for item in history_docs]

    top_farmers, village_board = _leaderboards(db, limit=10)

    milestones = [1, 3, 5, 10]
    unlocked_milestones = [m for m in milestones if referral_count >= m]
    next_milestone = next((m for m in milestones if referral_count < m), None)

    return {
        "success": True,
        "data": {
            "referralCode": referral_code,
            "referralLink": referral_link,
            "share": {
                "whatsapp": f"https://wa.me/?text=Join%20Fasal%20Saathi%20using%20my%20referral%20code%20{referral_code}%20-%20{referral_link}",
                "sms": f"sms:?body=Join%20Fasal%20Saathi%20using%20my%20referral%20code%20{referral_code}%20-%20{referral_link}",
            },
            "stats": {
                "referralCount": referral_count,
                "referralPoints": referral_points,
                "referralBadge": referral_badge,
                "community": _community_label(user_data),
                "unlockedPremium": referral_count >= 5,
            },
            "milestones": {
                "all": milestones,
                "unlocked": unlocked_milestones,
                "next": next_milestone,
            },
            "history": history,
            "leaderboard": {
                "farmers": top_farmers,
                "villages": village_board,
            },
        },
    }


@router.post("/code")
async def generate_referral_code(request: Request):
    db = _require_db()
    uid = await _get_uid_from_request(request)

    user_ref = db.collection("users").document(uid)
    user_snap = user_ref.get()
    user_data = user_snap.to_dict() if user_snap.exists else {}

    code = _ensure_user_referral_code(db, uid, user_data)
    return {"success": True, "code": code}


@router.post("/redeem")
async def redeem_referral_code(payload: RedeemReferralRequest, request: Request):
    db = _require_db()
    invitee_uid = await _get_uid_from_request(request)

    normalized_code = _normalize_referral_code(payload.referral_code)
    if not normalized_code:
        raise HTTPException(status_code=400, detail="Invalid referral code")

    # Resolve the inviter from the referral code before entering the transaction.
    # This read does not need to be inside the transaction because referral codes
    # are immutable once created — the uid they map to never changes.
    code_ref = db.collection("referral_codes").document(normalized_code)
    code_doc = code_ref.get()
    if not code_doc.exists:
        raise HTTPException(status_code=404, detail="Referral code not found")

    inviter_uid = (code_doc.to_dict() or {}).get("uid")
    if not inviter_uid:
        raise HTTPException(status_code=404, detail="Referral code is invalid")
    if inviter_uid == invitee_uid:
        raise HTTPException(status_code=400, detail="Self referral is not allowed")

    invitee_ref = db.collection("users").document(invitee_uid)
    inviter_ref = db.collection("users").document(inviter_uid)
    referral_doc_id = f"{inviter_uid}_{invitee_uid}"
    referral_ref = db.collection("referrals").document(referral_doc_id)
    reward_history_ref = db.collection("reward_history").document()

    created_at = _now_iso()
    reward_points = 50

    # All six Firestore operations are wrapped in a single transaction so that:
    #   1. The duplicate-redemption check and the referral write are atomic —
    #      concurrent redemptions of the same code cannot both pass the guard.
    #   2. The inviter's referralCount increment and the subsequent badge/premium
    #      read are performed inside the same transaction, guaranteeing that the
    #      count we read is exactly the value after our increment (Firestore
    #      transactions return the post-commit state of documents read within
    #      them, and the server-side Increment is applied before the read is
    #      returned to the transaction).
    @firestore.transactional
    def _run_redemption(transaction, invitee_ref, inviter_ref, referral_ref, reward_history_ref):
        # --- reads (must come before any writes in a Firestore transaction) ---
        invitee_snap = invitee_ref.get(transaction=transaction)
        invitee_data = invitee_snap.to_dict() if invitee_snap.exists else {}

        if (invitee_data or {}).get("referredByUid") or (invitee_data or {}).get("referralRedeemedAt"):
            raise HTTPException(status_code=409, detail="Referral already redeemed for this account")

        referral_snap = referral_ref.get(transaction=transaction)
        if referral_snap.exists:
            raise HTTPException(status_code=409, detail="Duplicate referral attempt blocked")

        inviter_snap = inviter_ref.get(transaction=transaction)
        inviter_data = inviter_snap.to_dict() if inviter_snap.exists else {}

        # Compute the new count from the current persisted value plus our increment
        # so that badge and premium status are derived from the correct post-commit
        # count without requiring a second round-trip after the transaction.
        current_count = int((inviter_data or {}).get("referralCount", 0) or 0)
        new_count = current_count + 1
        new_points = int((inviter_data or {}).get("referralPoints", 0) or 0) + reward_points

        # --- writes ---
        transaction.set(
            referral_ref,
            {
                "inviterUid": inviter_uid,
                "inviteeUid": invitee_uid,
                "inviteeName": (invitee_data or {}).get("displayName") or "Farmer",
                "inviteeLocation": _community_label(invitee_data),
                "referralCode": normalized_code,
                "status": "redeemed",
                "rewardPoints": reward_points,
                "createdAt": created_at,
                "updatedAt": created_at,
            },
        )

        transaction.set(
            invitee_ref,
            {
                "referredByUid": inviter_uid,
                "referredByCode": normalized_code,
                "referralRedeemedAt": created_at,
            },
            merge=True,
        )

        transaction.set(
            inviter_ref,
            {
                "referralCount": new_count,
                "referralPoints": new_points,
                "referralBadge": _referral_badge(new_count),
                "premiumUnlocked": new_count >= 5,
                "updatedAt": created_at,
            },
            merge=True,
        )

        transaction.set(
            reward_history_ref,
            {
                "uid": inviter_uid,
                "type": "referral_reward",
                "points": reward_points,
                "inviteeUid": invitee_uid,
                "inviteeName": (invitee_data or {}).get("displayName") or "Farmer",
                "referralCode": normalized_code,
                "createdAt": created_at,
            },
        )

        return new_count

    transaction = db.transaction()
    inviter_count = _run_redemption(
        transaction, invitee_ref, inviter_ref, referral_ref, reward_history_ref
    )

    return {
        "success": True,
        "message": "Referral redeemed successfully",
        "data": {
            "inviterUid": inviter_uid,
            "inviteeUid": invitee_uid,
            "rewardPoints": reward_points,
            "inviterReferralCount": inviter_count,
            "inviterBadge": _referral_badge(inviter_count),
        },
    }


@router.get("/history")
async def get_referral_history(request: Request, limit: int = Query(default=20, ge=1, le=100)):
    db = _require_db()
    uid = await _get_uid_from_request(request)

    docs = (
        db.collection("referrals")
        .where("inviterUid", "==", uid)
        .order_by("createdAt", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .get()
    )

    return {"success": True, "data": [_history_entry(item) for item in docs]}


@router.get("/leaderboard")
async def get_referral_leaderboard(request: Request, limit: int = Query(default=10, ge=3, le=50)):
    """
    Public referral leaderboard — requires authentication.

    Authentication is required to prevent unauthenticated enumeration of
    user accounts. UIDs are not included in the response regardless.
    """
    db = _require_db()
    await _get_uid_from_request(request)   # raises 401 if token is missing/invalid
    farmers, villages = _leaderboards(db, limit=limit)
    return {"success": True, "data": {"farmers": farmers, "villages": villages}}
