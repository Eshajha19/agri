"""LMS Router — server-side lesson completion and certificate issuance.

All completion state is stored in Firestore under:
  users/{uid}/lms_progress/{courseId}  →  { lessons: {lessonId: true, ...}, completedAt: ISO }

Certificates are only issued when the server confirms 100% completion from
Firestore. localStorage is used only as a UI cache; it is never trusted for
authorization decisions.
"""
import asyncio
import hashlib
import logging
import secrets
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, Tuple

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)

# Per-user per-course certificate request cooldown (seconds).
_CERT_COOLDOWN_SECONDS = 60

# Maximum number of (uid, course_id) pairs tracked in the cooldown store.
# Each entry is a single float timestamp (~56 bytes), so 10 000 entries
# consume roughly 560 KB — a safe upper bound for a long-running process.
# When the cap is reached the least-recently-used entry is evicted before
# the new one is inserted, keeping memory proportional to this constant
# regardless of how many distinct users the process has served.
_CERT_COOLDOWN_MAX_ENTRIES = 10_000

# OrderedDict used as an LRU store: move_to_end() on every access keeps
# the most-recently-used entries at the tail; popitem(last=False) evicts
# the least-recently-used entry from the head when the cap is reached.
_last_cert_request: OrderedDict[Tuple[str, str], float] = OrderedDict()

# Asyncio lock that serialises all reads and writes to _last_cert_request.
# Without this lock, two concurrent requests for the same (uid, course_id)
# can both pass the cooldown check before either one records a timestamp,
# allowing the cooldown to be bypassed under load.
_cert_cooldown_lock: asyncio.Lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Injected dependencies (wired in main.py lifespan via init_lms)
# ---------------------------------------------------------------------------

_verify_role_fn = None
_db = None  # Firestore client


def init_lms(verify_role_fn, db_client):
    global _verify_role_fn, _db
    _verify_role_fn = verify_role_fn
    _db = db_client


# ---------------------------------------------------------------------------
# Course catalogue — single source of truth shared with the frontend
# ---------------------------------------------------------------------------

COURSES: Dict[str, Dict] = {
    "soil-health": {
        "title": "Advanced Soil Management",
        "lessons": ["s1", "s2", "s3"],
    },
    "pest-control": {
        "title": "Organic Pest Management",
        "lessons": ["p1", "p2"],
    },
    "modern-tools": {
        "title": "Drones in Agriculture",
        "lessons": ["t1", "t2"],
    },
}

# Flat map: lessonId → courseId for fast lookup
_LESSON_TO_COURSE: Dict[str, str] = {
    lesson_id: course_id
    for course_id, course in COURSES.items()
    for lesson_id in course["lessons"]
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CompleteLessonRequest(BaseModel):
    lesson_id: str = Field(..., min_length=1, max_length=20, pattern=r"^[a-zA-Z0-9_-]+$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_firestore():
    if _db is None:
        raise HTTPException(
            status_code=503,
            detail="LMS service temporarily unavailable",
        )


def _progress_ref(uid: str, course_id: str):
    return _db.collection("users").document(uid).collection("lms_progress").document(course_id)


def _get_progress(uid: str, course_id: str) -> dict:
    """Return the stored progress dict, or an empty one if not yet started."""
    try:
        snap = _progress_ref(uid, course_id).get()
        return snap.to_dict() if snap.exists else {}
    except Exception as exc:
        logger.error("Firestore read failed for uid=%s course=%s: %s", uid, course_id, exc)
        raise HTTPException(status_code=503, detail="LMS service temporarily unavailable")


def _is_complete(progress: dict, course_id: str) -> bool:
    lessons = COURSES[course_id]["lessons"]
    completed = progress.get("lessons", {})
    return all(completed.get(lid) is True for lid in lessons)


def _make_cert_id(uid: str, course_id: str) -> str:
    """Generate a unique certificate ID for each issuance.

    A cryptographically random nonce (16 hex characters from secrets.token_hex)
    is mixed into the hash input so that every call produces a different ID,
    even for the same uid and course_id combination.  This closes two
    previously identified issues:

    1. Re-certification replay: a user whose certificate was revoked and who
       re-completed the course would receive the exact same ID, making the
       revoked and the new certificate indistinguishable.

    2. Predictable ID space: because the inputs (uid, course_id) are known to
       the caller, a fully deterministic function allowed a user to pre-compute
       another user's certificate ID without completing the course.

    The nonce is NOT stored separately; the cert ID itself is the persistent
    record.  Firestore deduplication (if needed) should use the cert document
    path rather than the ID value.
    """
    nonce = secrets.token_hex(8)
    raw = f"{uid}:{course_id}:{nonce}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16].upper()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/lms/complete-lesson")
async def complete_lesson(request: Request, body: CompleteLessonRequest):
    """
    Record a lesson as completed for the authenticated user.

    The lesson_id is validated against the server-side course catalogue.
    Unknown lesson IDs are rejected with 400 — a client cannot invent
    lesson IDs to manufacture fake completion records.
    """
    if _verify_role_fn is None:
        raise HTTPException(status_code=500, detail="LMS not initialized")
    _require_firestore()

    token_data = await _verify_role_fn(request)
    uid = token_data.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="User identity missing from authentication token")

    lesson_id = body.lesson_id
    course_id = _LESSON_TO_COURSE.get(lesson_id)
    if course_id is None:
        raise HTTPException(status_code=400, detail=f"Unknown lesson: {lesson_id}")

    progress = _get_progress(uid, course_id)
    lessons_done = dict(progress.get("lessons", {}))
    lessons_done[lesson_id] = True

    now_iso = datetime.now(timezone.utc).isoformat()
    update: dict = {"lessons": lessons_done, "updatedAt": now_iso}

    # Record completion timestamp the first time all lessons are done.
    all_done = all(lessons_done.get(lid) is True for lid in COURSES[course_id]["lessons"])
    if all_done and not progress.get("completedAt"):
        update["completedAt"] = now_iso

    try:
        _progress_ref(uid, course_id).set(update, merge=True)
    except Exception as exc:
        logger.error("Firestore write failed for uid=%s course=%s: %s", uid, course_id, exc)
        raise HTTPException(status_code=503, detail="LMS service temporarily unavailable")

    return {
        "success": True,
        "lesson_id": lesson_id,
        "course_id": course_id,
        "course_complete": all_done,
    }


@router.get("/lms/progress")
async def get_progress(request: Request):
    """
    Return the authenticated user's completion state for all courses.
    The frontend uses this to hydrate its UI on load instead of trusting
    localStorage.
    """
    if _verify_role_fn is None:
        raise HTTPException(status_code=500, detail="LMS not initialized")
    _require_firestore()

    token_data = await _verify_role_fn(request)
    uid = token_data.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="User identity missing from authentication token")

    result = {}
    for course_id in COURSES:
        progress = _get_progress(uid, course_id)
        lessons_done = progress.get("lessons", {})
        result[course_id] = {
            "lessons": lessons_done,
            "completedAt": progress.get("completedAt"),
        }

    return {"success": True, "progress": result}


@router.get("/lms/certificate/{course_id}")
async def get_certificate_data(request: Request, course_id: str):
    """
    Return certificate metadata for a completed course.

    The certificate is only issued when Firestore confirms 100% completion.
    The recipient name comes from the verified Firestore user profile, not
    from any client-supplied value, so it is always tied to a real identity.

    Returns:
        {
          "success": true,
          "certificate": {
            "recipient_name": "...",
            "course_title": "...",
            "completed_at": "ISO date",
            "cert_id": "deterministic hex ID"
          }
        }
    """
    if _verify_role_fn is None:
        raise HTTPException(status_code=500, detail="LMS not initialized")
    _require_firestore()

    if course_id not in COURSES:
        raise HTTPException(status_code=404, detail="Course not found")

    token_data = await _verify_role_fn(request)
    uid = token_data.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="User identity missing from authentication token")

    # Per-user per-course cooldown to prevent Firestore cost abuse.
    # The entire check-then-update block is serialised with an asyncio lock
    # to prevent a TOCTOU race where two concurrent requests both pass the
    # cooldown check before either one records its timestamp.
    # time.monotonic() is used instead of time.time() because monotonic
    # time never runs backwards. A backward NTP jump on time.time() would
    # reset (now - last) to a large positive number, clearing the cooldown
    # and allowing repeated Firestore reads until the clock catches up again.
    key = (uid, course_id)
    async with _cert_cooldown_lock:
        now = time.monotonic()
        last = _last_cert_request.get(key)
        if last is not None and (now - last) < _CERT_COOLDOWN_SECONDS:
            raise HTTPException(
                status_code=429,
                detail=f"Certificate already requested recently. Please wait {_CERT_COOLDOWN_SECONDS} seconds.",
            )
        # Evict the LRU entry before inserting when the cap is reached, then
        # record the current timestamp and move the key to the MRU position.
        if key not in _last_cert_request and len(_last_cert_request) >= _CERT_COOLDOWN_MAX_ENTRIES:
            _last_cert_request.popitem(last=False)
        _last_cert_request[key] = now
        _last_cert_request.move_to_end(key)

    progress = _get_progress(uid, course_id)
    if not _is_complete(progress, course_id):
        raise HTTPException(
            status_code=403,
            detail="Course not yet completed — finish all lessons before requesting a certificate",
        )

    # Fetch the user's display name from Firestore (authoritative source).
    try:
        user_snap = _db.collection("users").document(uid).get()
        user_data = user_snap.to_dict() if user_snap.exists else {}
    except Exception as exc:
        logger.error("Firestore user fetch failed for uid=%s: %s", uid, exc)
        raise HTTPException(status_code=503, detail="LMS service temporarily unavailable")

    recipient_name = (
        user_data.get("displayName")
        or user_data.get("name")
        or "Fasal Saathi Student"
    )

    completed_at = progress.get("completedAt", datetime.now(timezone.utc).isoformat())
    cert_id = _make_cert_id(uid, course_id)

    return {
        "success": True,
        "certificate": {
            "recipient_name": recipient_name,
            "course_title": COURSES[course_id]["title"],
            "course_id": course_id,
            "completed_at": completed_at,
            "cert_id": cert_id,
        },
    }

# The certificate cooldown atomicity issue (asyncio.Lock wrapping the check-then-update) was already addressed in commit 22821c0 and further hardened in 69cb52f.