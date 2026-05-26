"""LMS Router — server-side lesson completion and certificate issuance.

All completion state is stored in Firestore under:
  users/{uid}/lms_progress/{courseId}  →  { lessons: {lessonId: true, ...}, completedAt: ISO }

Certificates are only issued when the server confirms 100% completion from
Firestore. localStorage is used only as a UI cache; it is never trusted for
authorization decisions.
"""
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)

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
    """Deterministic certificate ID — uid + course_id only, so repeated calls
    for the same user and course always return the same ID.  The mutable
    completed_at timestamp is intentionally excluded from the hash to prevent
    users from minting unlimited unique cert IDs by editing Firestore."""
    raw = f"{uid}:{course_id}"
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
