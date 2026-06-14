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
from google.cloud import firestore
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)

# Per-user per-course certificate request cooldown (seconds).
_CERT_COOLDOWN_SECONDS = 60

# Maximum number of (uid, course_id) pairs tracked in the cooldown store.
_CERT_COOLDOWN_MAX_ENTRIES = 10_000

# Last cleanup timestamp for stale entry removal.
_LAST_CLEANUP_TIME = time.monotonic()
_CLEANUP_INTERVAL = 300  # Clean up stale entries every 5 minutes

_last_cert_request: OrderedDict[Tuple[str, str], float] = OrderedDict()
_cert_cooldown_lock: asyncio.Lock = asyncio.Lock()


def _cleanup_stale_cooldown_entries(now: float) -> None:
    """Remove expired cooldown entries from the registry.
    
    Entries older than _CERT_COOLDOWN_SECONDS are removed to prevent
    unbounded memory growth. This is called periodically during requests
    to maintain a bounded in-memory registry.
    """
    global _LAST_CLEANUP_TIME
    
    # Only clean up if interval has passed to avoid excessive iteration.
    if (now - _LAST_CLEANUP_TIME) < _CLEANUP_INTERVAL:
        return
    
    cutoff_time = now - _CERT_COOLDOWN_SECONDS
    expired_keys = [
        key for key, timestamp in _last_cert_request.items()
        if timestamp < cutoff_time
    ]
    
    for key in expired_keys:
        del _last_cert_request[key]
    
    if expired_keys:
        logger.debug(
            "Cleaned up %d stale certificate cooldown entries",
            len(expired_keys),
        )
    
    _LAST_CLEANUP_TIME = now

_verify_role_fn = None
_db = None


def init_lms(verify_role_fn, db_client):
    global _verify_role_fn, _db
    _verify_role_fn = verify_role_fn
    _db = db_client


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

_LESSON_TO_COURSE: Dict[str, str] = {
    lesson_id: course_id
    for course_id, course in COURSES.items()
    for lesson_id in course["lessons"]
}


class CompleteLessonRequest(BaseModel):
    lesson_id: str = Field(..., min_length=1, max_length=20, pattern=r"^[a-zA-Z0-9_-]+$")


def _require_firestore():
    if _db is None:
        raise HTTPException(
            status_code=503,
            detail="LMS service temporarily unavailable",
        )


def _progress_ref(uid: str, course_id: str):
    return _db.collection("users").document(uid).collection("lms_progress").document(course_id)


def _get_progress(uid: str, course_id: str) -> dict:
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

def _analyze_learning_drift(progress: dict, course_id: str) -> dict:
    lessons = COURSES[course_id]["lessons"]
    completed = progress.get("lessons", {})

    completed_count = sum(
        1 for lesson in lessons
        if completed.get(lesson)
    )

    completion_rate = round(
        (completed_count / len(lessons)) * 100,
        2,
    )

    inactivity_days = 0

    updated_at = progress.get("updatedAt")
    if updated_at:
        try:
            last_activity = datetime.fromisoformat(updated_at)

            if last_activity.tzinfo is None:
                last_activity = last_activity.replace(
                    tzinfo=timezone.utc
                )

            inactivity_days = (
                datetime.now(timezone.utc) - last_activity
            ).days
        except Exception:
            inactivity_days = 0

    drift_reasons = []

    if inactivity_days >= 7:
        drift_reasons.append("extended_inactivity")

    if completion_rate < 50:
        drift_reasons.append("slow_completion_trend")

    drift_detected = len(drift_reasons) > 0

    if inactivity_days == 0:
        streak = "active"
    elif inactivity_days <= 3:
        streak = "warning"
    else:
        streak = "broken"

    return {
        "drift_detected": drift_detected,
        "inactivity_days": inactivity_days,
        "completion_rate": completion_rate,
        "learning_streak_status": streak,
        "drift_reasons": drift_reasons,
    }


def _make_cert_id(uid: str, course_id: str) -> str:
    nonce = secrets.token_hex(8)
    raw = f"{uid}:{course_id}:{nonce}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16].upper()


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

    ref = _progress_ref(uid, course_id)
    all_done = False

    try:
        transaction = _db.transaction()

        @firestore.transactional
        def update_progress(transaction):
            nonlocal all_done
            snapshot = ref.get(transaction=transaction)
            current = snapshot.to_dict() if snapshot.exists else {}
            lessons_done = dict(current.get("lessons", {}))
            lessons_done[lesson_id] = True

            now_iso = datetime.now(timezone.utc).isoformat()
            update: dict = {"lessons": lessons_done, "updatedAt": now_iso}

            all_done = all(lessons_done.get(lid) is True for lid in COURSES[course_id]["lessons"])
            if all_done and not current.get("completedAt"):
                update["completedAt"] = now_iso

            transaction.set(ref, update, merge=True)

        update_progress(transaction)
    except Exception as exc:
        logger.error("Firestore transaction failed for uid=%s course=%s: %s", uid, course_id, exc)
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
            "drift_analysis": _analyze_learning_drift(
                progress,
                course_id,
            ),
        }

    return {"success": True, "progress": result}


@router.get("/lms/certificate/{course_id}")
async def get_certificate_data(request: Request, course_id: str):
    """
    Return certificate metadata for a completed course.
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

    key = (uid, course_id)
    async with _cert_cooldown_lock:
        now = time.monotonic()
        
        # Clean up stale entries periodically to prevent unbounded growth.
        _cleanup_stale_cooldown_entries(now)
        
        last = _last_cert_request.get(key)
        if last is not None and (now - last) < _CERT_COOLDOWN_SECONDS:
            raise HTTPException(
                status_code=429,
                detail=f"Certificate already requested recently. Please wait {_CERT_COOLDOWN_SECONDS} seconds.",
            )
        
        # If at capacity, remove oldest entry to make room for new one.
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