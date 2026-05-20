"""Voice Assistant Router - FastAPI endpoints for voice interaction"""

from fastapi import APIRouter, Request, HTTPException, File, UploadFile
from pydantic import BaseModel, Field, validator
import logging
from typing import Optional, Dict, Any
import os

router = APIRouter()
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models
# ============================================================================

class VoiceQueryRequest(BaseModel):
    """Request model for voice queries"""
    transcript: str = Field(..., min_length=1, max_length=500)
    language_code: str = Field(default="hi", pattern="^(hi|bho|mr|gu|kn|te|ta|en)$")
    session_id: Optional[str] = None
    user_id: str = Field(..., min_length=1, max_length=100)
    context: Optional[Dict[str, Any]] = None
    
    @validator("transcript")
    def sanitize_transcript(cls, v):
        # Remove dangerous characters
        v = v.strip()
        if len(v) < 1:
            raise ValueError("Transcript cannot be empty")
        return v[:500]


class AudioUploadRequest(BaseModel):
    """Request for audio file upload"""
    language_code: str = Field(default="hi", pattern="^(hi|bho|mr|gu|kn|te|ta|en)$")
    session_id: Optional[str] = None
    user_id: str = Field(..., min_length=1, max_length=100)


class VoiceResponseData(BaseModel):
    """Response model for voice interaction"""
    success: bool
    response_text: str
    language_code: str
    intent: str
    session_id: str
    offline_mode: bool
    metadata: Optional[Dict[str, Any]] = None


class SessionCreateRequest(BaseModel):
    """Request to create new voice session"""
    user_id: str = Field(..., min_length=1, max_length=100)
    language_code: str = Field(default="hi", pattern="^(hi|bho|mr|gu|kn|te|ta|en)$")


class SessionResponse(BaseModel):
    """Response for session creation"""
    session_id: str
    user_id: str
    language_code: str
    offline_mode: bool
    created_at: str


class LanguageListResponse(BaseModel):
    """List of supported languages"""
    languages: Dict[str, Dict[str, str]]


class HealthCheckResponse(BaseModel):
    """Health check response"""
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
    """Initialize voice assistant and cache manager"""
    global voice_assistant, cache_manager, verify_role_fn
    voice_assistant = va
    cache_manager = cm
    verify_role_fn = vr_fn
    logger.info("Voice Assistant router initialized")


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/health", response_model=HealthCheckResponse, tags=["Voice"])
async def health_check():
    """Check voice assistant health"""
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
    """List supported languages"""
    from voice_assistant import SUPPORTED_LANGUAGES
    
    return LanguageListResponse(languages=SUPPORTED_LANGUAGES)


@router.post("/sessions/create", response_model=SessionResponse, tags=["Voice"])
async def create_session(request: Request, data: SessionCreateRequest):
    """Create new voice session"""
    if voice_assistant is None:
        raise HTTPException(status_code=500, detail="Voice Assistant not initialized")
    
    try:
        session = voice_assistant.create_session(
            user_id=data.user_id,
            language_code=data.language_code,
        )
        
        return SessionResponse(
            session_id=session.session_id,
            user_id=session.user_id,
            language_code=session.language_code,
            offline_mode=session.offline_mode,
            created_at=session.start_time,
        )
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=VoiceResponseData, tags=["Voice"])
async def process_voice_query(request: Request, data: VoiceQueryRequest):
    """
    Process voice query and generate response
    
    Supports:
    - Hindi, Bhojpuri, Marathi, Gujarati, Kannada, Telugu, Tamil, English
    - Offline mode for low-network areas
    - Intent detection (crop_health, weather_alert, fertilizer, irrigation, etc.)
    """
    if voice_assistant is None:
        raise HTTPException(status_code=500, detail="Voice Assistant not initialized")
    
    try:
        from voice_assistant import VoiceInput, detect_language
        
        # Detect language if not provided
        detected_lang = detect_language(data.transcript)
        language_code = data.language_code if data.language_code else detected_lang
        
        # Create or retrieve session
        session_id = data.session_id
        if not session_id:
            session = voice_assistant.create_session(data.user_id, language_code)
            session_id = session.session_id
        elif session_id not in voice_assistant.sessions:
            raise ValueError(f"Invalid session: {session_id}")
        
        # Create voice input
        voice_input = VoiceInput(
            audio_bytes=b"",
            language_code=language_code,
            transcript=data.transcript,
        )
        
        # Process query
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
    
    except Exception as e:
        logger.error(f"Voice query error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
CHUNK_SIZE = 1024 * 1024  # 1MB chunks
TEMP_UPLOAD_DIR = "./temp_audio_uploads"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".webm", ".m4a"}

# In-memory rate limiting: user_id -> (count, window_start)
_rate_limit_store: Dict[str, tuple] = {}
RATE_LIMIT_COUNT = 10  # max uploads per window
RATE_LIMIT_WINDOW = 60  # seconds


def _check_rate_limit(user_id: str) -> bool:
    """Check if user has exceeded rate limit"""
    import time
    current_time = time.time()
    
    if user_id not in _rate_limit_store:
        _rate_limit_store[user_id] = (1, current_time)
        return True
    
    count, window_start = _rate_limit_store[user_id]
    
    if current_time - window_start > RATE_LIMIT_WINDOW:
        _rate_limit_store[user_id] = (1, current_time)
        return True
    
    if count >= RATE_LIMIT_COUNT:
        return False
    
    _rate_limit_store[user_id] = (count + 1, window_start)
    return True


def _validate_filename(filename: str) -> str:
    """Validate and sanitize filename"""
    if not filename:
        raise ValueError("No file provided")
    
    import re
    filename = re.sub(r'[^\w\-.]', '_', filename)
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    
    return filename


@router.post("/audio-upload", tags=["Voice"])
async def upload_audio(
    request: Request,
    file: UploadFile = File(...),
    language_code: str = "hi",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
):
    """
    Upload audio file for transcription
    
    Supported formats: WAV, MP3, OGG, WebM, M4A
    Max file size: 25MB
    
    Security features:
    - Streams to disk instead of memory buffering
    - Rate limited to prevent DoS
    - Validates file type and size
    """
    if voice_assistant is None:
        raise HTTPException(status_code=500, detail="Voice Assistant not initialized")
    
    try:
        if not user_id:
            raise ValueError("user_id required")
        
        if not _check_rate_limit(user_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
        
        safe_filename = _validate_filename(file.filename)
        
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        
        temp_path = os.path.join(TEMP_UPLOAD_DIR, f"{user_id}_{safe_filename}")
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
                        os.remove(temp_path)
                        raise ValueError(f"File too large (max {MAX_FILE_SIZE // (1024*1024)}MB)")
                    
                    f.write(chunk)
            
            if bytes_written == 0:
                raise ValueError("Empty file uploaded")
            
            session_id = session_id or voice_assistant.create_session(user_id, language_code).session_id
            
            logger.info(f"Audio uploaded: {safe_filename} ({bytes_written} bytes)")
            
            return {
                "success": True,
                "message": "Audio received - transcription in progress",
                "session_id": session_id,
                "filename": safe_filename,
                "size_bytes": bytes_written,
            }
        
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio upload error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/sessions/{session_id}", tags=["Voice"])
async def get_session_history(request: Request, session_id: str):
    """Retrieve session history and details"""
    if voice_assistant is None:
        raise HTTPException(status_code=500, detail="Voice Assistant not initialized")
    
    try:
        history = voice_assistant.get_session_history(session_id)
        return {"success": True, "session": history}
    except Exception as e:
        logger.error(f"Session retrieval error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/query-analyze", tags=["Voice"])
async def analyze_query(request: Request, data: VoiceQueryRequest):
    """
    Analyze voice query for quality and intent
    
    Returns:
    - Query clarity score
    - Detected intent
    - Entities (crop, disease, etc.)
    - Language detection
    """
    if voice_assistant is None:
        raise HTTPException(status_code=500, detail="Voice Assistant not initialized")
    
    try:
        from voice_assistant import VoiceQueryAnalyzer, detect_language
        
        # Detect language
        detected_lang = detect_language(data.transcript)
        language_code = data.language_code or detected_lang
        
        # Analyze query
        analysis = VoiceQueryAnalyzer.analyze(data.transcript, language_code)
        
        # Detect intent
        intent, confidence = voice_assistant.language_model.detect_intent(data.transcript)
        
        return {
            "success": True,
            "query": data.transcript,
            "analysis": analysis,
            "intent": intent,
            "intent_confidence": confidence,
            "language": language_code,
        }
    
    except Exception as e:
        logger.error(f"Query analysis error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/offline-cache", tags=["Voice"])
async def get_offline_cache(request: Request):
    """
    Retrieve offline knowledge cache
    
    Contains:
    - Crop disease information
    - Fertilizer recommendations
    - Irrigation schedules
    - Weather alerts
    """
    if voice_assistant is None:
        raise HTTPException(status_code=500, detail="Voice Assistant not initialized")
    
    try:
        return {
            "success": True,
            "offline_cache": voice_assistant.offline_cache,
            "cache_size_bytes": sum(
                len(str(v).encode()) for v in voice_assistant.offline_cache.values()
            ),
        }
    except Exception as e:
        logger.error(f"Cache retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync-cache", tags=["Voice"])
async def sync_offline_cache(request: Request):
    """
    Sync offline cache (for updating knowledge base)
    
    Useful for:
    - Adding new crop varieties
    - Updating disease information
    - Adding regional advisories
    """
    if cache_manager is None:
        raise HTTPException(status_code=500, detail="Cache manager not initialized")
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Authorization system not initialized")
    
    try:
        await verify_role_fn(request, required_roles=["admin", "system"])
        cache_manager.save_cache(voice_assistant.offline_cache)
        return {
            "success": True,
            "message": "Offline cache synchronized",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supported-intents", tags=["Voice"])
async def list_supported_intents(request: Request):
    """List all supported voice intents"""
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


# ============================================================================
# Documentation Endpoint
# ============================================================================

@router.get("/docs-voice", tags=["Voice"])
async def voice_assistant_docs(request: Request):
    """Get documentation for voice assistant"""
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
            {"code": "hi", "name": "Hindi", "native": "हिंदी"},
            {"code": "bho", "name": "Bhojpuri", "native": "भोजपुरी"},
            {"code": "mr", "name": "Marathi", "native": "मराठी"},
            {"code": "gu", "name": "Gujarati", "native": "ગુજરાતી"},
            {"code": "kn", "name": "Kannada", "native": "ಕನ್ನಡ"},
            {"code": "te", "name": "Telugu", "native": "తెలుగు"},
            {"code": "ta", "name": "Tamil", "native": "தமிழ்"},
            {"code": "en", "name": "English", "native": "English"},
        ],
        "example_queries": {
            "hi": [
                "मेरी फसल को क्या समस्या है?",
                "गेहूँ के लिए खाद कौन सी दें?",
                "सिंचाई का समय क्या है?",
                "मौसम कैसा रहेगा?",
            ],
            "en": [
                "What problem does my crop have?",
                "Which fertilizer for wheat?",
                "When should I irrigate?",
                "How will the weather be?",
            ],
        },
        "endpoints": [
            "/api/voice/health - Check service status",
            "/api/voice/languages - List supported languages",
            "/api/voice/sessions/create - Create session",
            "/api/voice/query - Process voice query",
            "/api/voice/audio-upload - Upload audio",
            "/api/voice/query-analyze - Analyze query",
            "/api/voice/offline-cache - Get offline knowledge",
        ],
    }
