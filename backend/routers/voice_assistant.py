"""Voice Assistant Router - FastAPI endpoints for voice interaction.

Security fix: all stateful endpoints now require Firebase token
verification.  The caller's uid is derived exclusively from the verified
token — never from a client-supplied user_id field.

Affected endpoints (previously open):
  POST /sessions/create  — now requires auth; uid from token
  POST /query            — now requires auth; uid from token
  POST /audio-upload     — now requires auth; uid from token; rate-limit
                           keyed on verified uid instead of client field
    POST /query-analyze    — now requires auth
"""

from error_utils import safe_detail
  GET  /sessions/{id}    — now requires auth; caller may only read their
                           own sessions
  GET  /offline-cache    — now requires auth (read of internal cache)

Unchanged (intentionally public):
  GET  /health           — service health probe
  GET  /languages        — static list
  GET  /supported-intents — static list
  GET  /docs-voice       — documentation

Still requires admin/system role:
  POST /sync-cache       — unchanged
"""
from collections import OrderedDict
from datetime import datetime, timezone
import os
import re
import sys
import logging
from threading import Lock
from time import monotonic
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, validator

router = APIRouter()
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models
# ============================================================================

def _build_recovery_response(
    message: str,
    error_type: str,
    session_id: Optional[str] = None,
) -> dict:
    recovery_prompts = {
        "audio": "Please upload a supported audio file and try again.",
        "session": "Try creating a new session and repeat your request.",
        "query": "Try rephrasing your question with more details.",
        "rate_limit": "Too many requests detected. Please wait before retrying.",
        "system": "Please retry in a few moments.",
    }

    return {
        "success": False,
        "error_type": error_type,
        "message": message,
        "recovery_prompt": recovery_prompts.get(
            error_type,
            "Please try again."
        ),
        "retry_supported": True,
        "session_id": session_id,
    }

class VoiceQueryRequest(BaseModel):
    """Request model for voice queries.

    user_id is intentionally absent — the authoritative identity is
    always derived from the verified Firebase ID token.
    """
    transcript: str = Field(..., min_length=1, max_length=500)
    language_code: str = Field(default="hi", pattern="^(hi|bho|mr|gu|kn|te|ta|en)$")
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

    @validator("transcript")
    def sanitize_transcript(cls, v):
        v = v.strip()
        if len(v) < 1:
            raise ValueError("Transcript cannot be empty")
        return v[:500]


class AudioUploadRequest(BaseModel):
    """Request for audio file upload.

    user_id removed — derived from token.
    """
    language_code: str = Field(default="hi", pattern="^(hi|bho|mr|gu|kn|te|ta|en)$")
    session_id: Optional[str] = None


class VoiceResponseData(BaseModel):
    success: bool
    response_text: str
    language_code: str
    intent: str
    session_id: str
    offline_mode: bool
    metadata: Optional[Dict[str, Any]] = None


class SessionCreateRequest(BaseModel):
    """Request to create new voice session.

    user_id removed — derived from token.
    """
    language_code: str = Field(default="hi", pattern="^(hi|bho|mr|gu|kn|te|ta|en)$")


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    language_code: str
    offline_mode: bool
    created_at: str


class LanguageListResponse(BaseModel):
    languages: Dict[str, Dict[str, str]]


class HealthCheckResponse(BaseModel):
    status: str
    voice_assistant_ready: bool
    offline_mode: bool
    supported_languages: int


# ============================================================================
# Global State
# ============================================================================

voice_assistant = None
cache_manager = None
verify_role_fn = None


def init_voice_assistant(va, cm, vr_fn=None):
    global voice_assistant, cache_manager, verify_role_fn
    voice_assistant = va
    cache_manager = cm
    verify_role_fn = vr_fn
    logger.info("Voice Assistant router initialized")


# ============================================================================
# Auth helper
# ============================================================================

async def _require_auth(request: Request) -> str:
    """Verify the Firebase ID token and return the caller's uid.

    Raises HTTP 401 if verify_role_fn is not configured or the token is
    missing/invalid.  Raises HTTP 500 if the router was not initialized.
    """
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Authorization system not initialized")
    token_data = await verify_role_fn(request)
    uid = token_data.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return uid


# ============================================================================
# Rate limiting (keyed on verified uid)
# ============================================================================

MAX_FILE_SIZE = 25 * 1024 * 1024   # 25 MB
CHUNK_SIZE = 1024 * 1024            # 1 MB
TEMP_UPLOAD_DIR = "./temp_audio_uploads"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".webm", ".m4a"}

_rate_limit_store: Dict[str, tuple] = {}
_rate_limit_lock = Lock()
_last_rate_limit_prune = 0.0
RATE_LIMIT_COUNT = 10   # max uploads per window
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_PRUNE_INTERVAL = RATE_LIMIT_WINDOW
RATE_LIMIT_MAX_ENTRIES = 10_000

# Session cleanup constants
MAX_SESSIONS = 1000
SESSION_TTL_SECONDS = 3600  # 1 hour
_session_created_at: OrderedDict[str, float] = OrderedDict()
_session_lock = Lock()

def _prune_rate_limit_store(now: float) -> None:
    """Drop expired rate-limit entries to bound in-memory state."""
    global _last_rate_limit_prune

    if now - _last_rate_limit_prune < RATE_LIMIT_PRUNE_INTERVAL:
        return

    expired_uids = [
        uid
        for uid, (_, window_start) in _rate_limit_store.items()
        if now - window_start > RATE_LIMIT_WINDOW
    ]
    for uid in expired_uids:
        _rate_limit_store.pop(uid, None)

    _last_rate_limit_prune = now


def _check_rate_limit(uid: str) -> bool:
    """Return True if the request is within the rate limit for this uid."""
    now = monotonic()
    with _rate_limit_lock:
        _prune_rate_limit_store(now)

        if len(_rate_limit_store) >= RATE_LIMIT_MAX_ENTRIES:
            sorted_uids = sorted(
                _rate_limit_store,
                key=lambda u: _rate_limit_store[u][1],
            )
            for uid_candidate in sorted_uids[: max(1, len(sorted_uids) // 4)]:
                _rate_limit_store.pop(uid_candidate, None)

        if uid not in _rate_limit_store:
            _rate_limit_store[uid] = (1, now)
            return True

        count, window_start = _rate_limit_store[uid]
        if now - window_start > RATE_LIMIT_WINDOW:
            _rate_limit_store[uid] = (1, now)
            return True

        if count >= RATE_LIMIT_COUNT:
            return False

        _rate_limit_store[uid] = (count + 1, window_start)
        return True
def _cleanup_sessions(now: float) -> None:
    """Evict expired and oldest sessions to bound memory."""
    if voice_assistant is None:
        return
    with _session_lock:
        expired = [
            sid for sid, created in _session_created_at.items()
            if now - created > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            _session_created_at.pop(sid, None)
            voice_assistant.sessions.pop(sid, None)
        while len(_session_created_at) >= MAX_SESSIONS:
            sid, _ = _session_created_at.popitem(last=False)
            voice_assistant.sessions.pop(sid, None)


def _register_session(session_id: str) -> None:
    now = monotonic()
    with _session_lock:
        _cleanup_sessions(now)
        _session_created_at[session_id] = now

def _validate_filename(filename: str) -> str:
    if not filename:
        raise ValueError("No file provided")
    filename = re.sub(r"[^\w\-.]", "_", filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    return filename


# ============================================================================
# Public endpoints (no auth required)
# ============================================================================

@router.get("/health", response_model=HealthCheckResponse, tags=["Voice"])
async def health_check():
    """Check voice assistant health — public."""
    if voice_assistant is None:
        raise HTTPException(status_code=500, detail="Voice Assistant not initialized")
    return HealthCheckResponse(
        status="ok",
        voice_assistant_ready=True,
        offline_mode=voice_assistant.offline_mode,
        supported_languages=len(voice_assistant.language_model.language_models),
    )


@router.get("/languages", response_model=LanguageListResponse, tags=["Voice"])
async def list_languages():
    """List supported languages — public."""
    from voice_assistant import SUPPORTED_LANGUAGES
    return LanguageListResponse(languages=SUPPORTED_LANGUAGES)


@router.get("/supported-intents", tags=["Voice"])
async def list_supported_intents():
    """List all supported voice intents — public."""
    from voice_assistant import INTENT_PATTERNS
    return {
        "success": True,
        "intents": {
            intent: {
                "name": intent.replace("_", " ").title(),
                "pattern_count": len(patterns),
            }
            for intent, patterns in INTENT_PATTERNS.items()
        },
    }


@router.get("/docs-voice", tags=["Voice"])
async def voice_assistant_docs():
    """Get documentation for voice assistant — public."""
    return {
        "title": "Voice Assistant for Farmers",
        "version": "1.0.0",
        "features": [
            "Multilingual voice interaction (Hindi, Bhojpuri, Marathi, Gujarati, Kannada, Telugu, Tamil)",
            "Offline-capable language understanding",
            "Intent detection (crop health, weather, fertilizer, irrigation, etc.)",
            "Entity extraction (crop type, disease, etc.)",
            "Session management",
            "Offline knowledge cache",
        ],
        "languages": [
            {"code": "hi",  "name": "Hindi",    "native": "हिंदी"},
            {"code": "bho", "name": "Bhojpuri", "native": "भोजपुरी"},
            {"code": "mr",  "name": "Marathi",  "native": "मराठी"},
            {"code": "gu",  "name": "Gujarati", "native": "ગુજરાતી"},
            {"code": "kn",  "name": "Kannada",  "native": "ಕನ್ನಡ"},
            {"code": "te",  "name": "Telugu",   "native": "తెలుగు"},
            {"code": "ta",  "name": "Tamil",    "native": "தமிழ்"},
            {"code": "en",  "name": "English",  "native": "English"},
        ],
        "endpoints": [
            "/api/voice/health - Check service status",
            "/api/voice/languages - List supported languages",
            "/api/voice/sessions/create - Create session (auth required)",
            "/api/voice/query - Process voice query (auth required)",
            "/api/voice/audio-upload - Upload audio (auth required)",
            "/api/voice/query-analyze - Analyze query (auth required)",
            "/api/voice/offline-cache - Get offline knowledge (auth required)",
        ],
    }


# ============================================================================
# Authenticated endpoints
# ============================================================================

@router.post("/sessions/create", response_model=SessionResponse, tags=["Voice"])
async def create_session(request: Request, data: SessionCreateRequest):
    """Create a new voice session.

    Requires authentication.  The session is created under the caller's
    verified uid — not a client-supplied user_id.
    """
    if voice_assistant is None:
        raise HTTPException(status_code=500, detail="Voice Assistant not initialized")

    uid = await _require_auth(request)

    try:
        new_session = voice_assistant.create_session(uid, data.language_code)
        session_id = new_session.session_id
        _register_session(session_id)
        return SessionResponse(
            session_id=session_id,
            user_id=uid,
            language_code=data.language_code,
            offline_mode=voice_assistant.offline_mode,
            created_at=new_session.start_time,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        raise HTTPException(status_code=500, detail=safe_detail(e, 500))


@router.post("/query", response_model=VoiceResponseData, tags=["Voice"])
async def process_voice_query(request: Request, data: VoiceQueryRequest):
    """Process a voice query and generate a response.

    Requires authentication.  The session is owned by the verified uid.
    """
    if voice_assistant is None:
        raise HTTPException(status_code=500, detail="Voice Assistant not initialized")

    uid = await _require_auth(request)

    try:
        from voice_assistant import VoiceInput, detect_language

        detected_lang = detect_language(data.transcript)
        language_code = data.language_code if data.language_code else detected_lang

        session_id = data.session_id
        if not session_id:
            session = voice_assistant.create_session(uid, language_code)
            session_id = session.session_id
            _register_session(session_id)
        else:
            # Verify the session belongs to the authenticated user.
            if session_id not in voice_assistant.sessions:
                raise HTTPException(status_code=404, detail="Session not found")
            session = voice_assistant.sessions[session_id]
            if getattr(session, "user_id", None) != uid:
                raise HTTPException(status_code=403, detail="Access denied: session belongs to another user")

        voice_input = VoiceInput(
            audio_bytes=b"",
            language_code=language_code,
            transcript=data.transcript,
        )

        response = voice_assistant.process_voice_input(
            voice_input=voice_input,
            session_id=session_id,
            context=data.context,
        )

        return VoiceResponseData(
            success=True,
            response_text=response.text,
            language_code=response.language_code,
            intent=response.intent,
            session_id=session_id,
            offline_mode=voice_assistant.offline_mode,
            metadata=response.metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice query error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))


@router.post("/audio-upload", tags=["Voice"])
async def upload_audio(
    request: Request,
    file: UploadFile = File(...),
    language_code: str = "hi",
    session_id: Optional[str] = None,
):
    """Upload an audio file for transcription.

    Requires authentication.  Rate limiting is keyed on the verified uid
    so it cannot be bypassed by rotating a client-supplied user_id.

    Supported formats: WAV, MP3, OGG, WebM, M4A
    Max file size: 25 MB
    """
    if voice_assistant is None:
        raise HTTPException(status_code=500, detail="Voice Assistant not initialized")

    uid = await _require_auth(request)

    if not _check_rate_limit(uid):
        return _build_recovery_response(
            "Rate limit exceeded.",
            "rate_limit",
            session_id,
        )

    try:
        safe_filename = _validate_filename(file.filename or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    # Sanitize uid before embedding it in a filesystem path. Firebase UIDs are
    # currently 28-character alphanumeric strings, but a defensive strip prevents
    # path traversal if the auth provider is ever changed or the token is crafted
    # to contain special characters.
    safe_uid = re.sub(r"[^a-zA-Z0-9_-]", "_", uid)
    temp_path = os.path.join(TEMP_UPLOAD_DIR, f"{safe_uid}_{safe_filename}")
    bytes_written = 0

    try:
        with open(temp_path, "wb") as f:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_FILE_SIZE:
                    f.flush()
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (max {MAX_FILE_SIZE // (1024 * 1024)} MB)",
                    )
                f.write(chunk)

        if bytes_written == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        if not session_id:
            new_session = voice_assistant.create_session(uid, language_code)
            session_id = new_session.session_id
            _register_session(session_id)


        logger.info("Audio uploaded: uid=%s file=%s bytes=%d", uid, safe_filename, bytes_written)

        return {
            "success": True,
            "message": "Audio received — transcription in progress",
            "session_id": session_id,
            "filename": safe_filename,
            "size_bytes": bytes_written,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio upload error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


@router.get("/sessions/{session_id}", tags=["Voice"])
async def get_session_history(request: Request, session_id: str):
    """Retrieve session history.

    Requires authentication.  A caller may only read sessions they own.
    """
    if voice_assistant is None:
        raise HTTPException(status_code=500, detail="Voice Assistant not initialized")

    uid = await _require_auth(request)

    try:
        if session_id not in voice_assistant.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = voice_assistant.sessions[session_id]
        if getattr(session, "user_id", None) != uid:
            # Return 404 rather than 403 to avoid leaking session existence
            # to callers who do not own the session.
            raise HTTPException(status_code=404, detail="Session not found")

        history = voice_assistant.get_session_history(session_id)
        return {"success": True, "session": history}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session retrieval error: {e}")
        raise HTTPException(status_code=404, detail=safe_detail(e, 404))


@router.post("/query-analyze", tags=["Voice"])
async def analyze_query(request: Request, data: VoiceQueryRequest):
    """Analyze a voice query for quality and intent.

    Requires authentication.
    """
    if voice_assistant is None:
        raise HTTPException(status_code=500, detail="Voice Assistant not initialized")

    await _require_auth(request)

    try:
        from voice_assistant import VoiceQueryAnalyzer, detect_language

        detected_lang = detect_language(data.transcript)
        language_code = data.language_code or detected_lang

        analysis = VoiceQueryAnalyzer.analyze(data.transcript, language_code)
        intent, confidence = voice_assistant.language_model.detect_intent(data.transcript)

        return {
            "success": True,
            "query": data.transcript,
            "analysis": analysis,
            "intent": intent,
            "intent_confidence": confidence,
            "language": language_code,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query analysis error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))


@router.get("/offline-cache", tags=["Voice"])
async def get_offline_cache(request: Request):
    """Return lightweight cache metadata.

    Restricted to admin/system roles.  Returns entry count and a shallow
    byte estimate instead of raw cache content or expensive full
    serialization.
    """
    if voice_assistant is None:
        raise HTTPException(status_code=500, detail="Voice Assistant not initialized")
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Authorization system not initialized")

    await verify_role_fn(request, required_roles=["admin", "system"])

    try:
        cache = voice_assistant.offline_cache
        return {
            "success": True,
            "cache_entries": len(cache),
            "cache_size_bytes": sys.getsizeof(cache),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache retrieval error: {e}")
        raise HTTPException(status_code=500, detail=safe_detail(e, 500))


@router.post("/sync-cache", tags=["Voice"])
async def sync_offline_cache(request: Request):
    """Sync offline cache — requires admin or system role (unchanged)."""
    if cache_manager is None:
        raise HTTPException(status_code=500, detail="Cache manager not initialized")
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Authorization system not initialized")

    try:
        await verify_role_fn(request, required_roles=["admin", "system"])
        cache_manager.save_cache(voice_assistant.offline_cache)
        return {"success": True, "message": "Offline cache synchronized"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache sync error: {e}")
        raise HTTPException(status_code=500, detail=safe_detail(e, 500))
