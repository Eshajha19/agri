"""
Voice Assistant for Farmers
Multilingual voice input/output with offline support

Features:
- Hindi + Regional languages (Bhojpuri, Marathi, Gujarati, Kannada, Telugu, Tamil)
- Voice-to-text and text-to-speech
- Offline and low-network functionality
- Integration with crop recommendations, weather alerts, soil analysis
"""

from __future__ import annotations
import json
import logging
import os
import threading
import time
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import re
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# Error Handling & Validation
# ============================================================================

class AudioFormat(str, Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    OPUS = "opus"


@dataclass
class AudioValidationResult:
    """Result of audio validation"""
    is_valid: bool
    error_message: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    bitrate: Optional[int] = None
    format_type: Optional[str] = None


class AudioValidator:
    """Validate audio format and specifications"""

    SUPPORTED_FORMATS = {AudioFormat.WAV, AudioFormat.MP3, AudioFormat.OPUS}
    MIN_BITRATE = 8000  # 8 kHz
    MAX_BITRATE = 48000  # 48 kHz
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

    @classmethod
    def validate_format(cls, file_path: str, bitrate: int = 16000) -> AudioValidationResult:
        """Validate audio file format"""
        try:
            # Check file size
            if not os.path.exists(file_path):
                return AudioValidationResult(
                    is_valid=False,
                    error_message="Audio file not found"
                )

            file_size = os.path.getsize(file_path)
            if file_size > cls.MAX_FILE_SIZE:
                return AudioValidationResult(
                    is_valid=False,
                    error_message=f"File too large: {file_size / 1024 / 1024:.1f}MB (max 100MB)",
                    suggestions=["Compress the audio file", "Split into smaller chunks"]
                )

            # Check format
            _, ext = os.path.splitext(file_path)
            ext = ext.lower().lstrip('.')

            if ext not in {fmt.value for fmt in cls.SUPPORTED_FORMATS}:
                return AudioValidationResult(
                    is_valid=False,
                    error_message=f"Unsupported format: {ext}",
                    suggestions=[
                        f"Supported formats: {', '.join(fmt.value for fmt in cls.SUPPORTED_FORMATS)}",
                        "Use FFmpeg to convert your audio file"
                    ]
                )

            # Check bitrate
            if bitrate < cls.MIN_BITRATE or bitrate > cls.MAX_BITRATE:
                return AudioValidationResult(
                    is_valid=False,
                    error_message=f"Invalid bitrate: {bitrate}Hz (must be 8000-48000Hz)",
                    suggestions=[
                        f"Resample audio to 16000Hz (recommended)",
                        "Use: ffmpeg -i input.wav -ar 16000 output.wav"
                    ],
                    bitrate=bitrate,
                    format_type=ext
                )

            return AudioValidationResult(
                is_valid=True,
                bitrate=bitrate,
                format_type=ext
            )

        except Exception as e:
            logger.error(f"Audio validation error: {e}")
            return AudioValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}"
            )


class VoiceAssistantError(Exception):
    """Base class for voice assistant errors"""
    def __init__(self, error_code: str, message: str, suggestions: List[str] = None):
        self.error_code = error_code
        self.message = message
        self.suggestions = suggestions or []
        super().__init__(self.message)


class PermissionError(VoiceAssistantError):
    """Microphone permission error"""
    def __init__(self):
        super().__init__(
            error_code="PERMISSION_DENIED",
            message="Microphone permission not granted",
            suggestions=[
                "Grant microphone permissions in device settings",
                "Check if browser is allowed to access microphone",
                "Restart the application"
            ]
        )


class TranscriptionError(VoiceAssistantError):
    """Transcription failure error"""
    def __init__(self, retry_count: int = 0):
        super().__init__(
            error_code="TRANSCRIPTION_FAILED",
            message=f"Failed to transcribe audio (attempt {retry_count})",
            suggestions=[
                "Check audio quality and noise levels",
                "Speak clearly and slowly",
                "Try using a quieter environment"
            ]
        )


class IntentParsingError(VoiceAssistantError):
    """Intent parsing error"""
    def __init__(self, confidence: float):
        super().__init__(
            error_code="LOW_CONFIDENCE",
            message=f"Could not understand intent (confidence: {confidence:.1%})",
            suggestions=[
                "Please rephrase your question",
                "Try using different wording",
                "Speak more clearly"
            ]
        )


@dataclass
class ErrorAnalytics:
    """Analytics for error tracking"""
    error_type: str
    count: int = 0
    last_occurrence: Optional[str] = None
    affected_languages: List[str] = field(default_factory=list)

    def record_error(self, language: str):
        """Record an error occurrence"""
        self.count += 1
        self.last_occurrence = datetime.now().isoformat()
        if language not in self.affected_languages:
            self.affected_languages.append(language)


class ErrorAnalyticsManager:
    """Manage and track error analytics"""
    def __init__(self):
        self.error_stats: Dict[str, ErrorAnalytics] = {}
        self.lock = threading.Lock()

    def record_error(self, error_type: str, language: str = "en"):
        """Record error for analytics"""
        with self.lock:
            if error_type not in self.error_stats:
                self.error_stats[error_type] = ErrorAnalytics(error_type=error_type)
            self.error_stats[error_type].record_error(language)
            logger.info(f"Error recorded: {error_type} (total: {self.error_stats[error_type].count})")

    def get_stats(self) -> Dict:
        """Get error statistics"""
        with self.lock:
            return {
                error_type: asdict(stats)
                for error_type, stats in self.error_stats.items()
            }

    def check_error_spike(self, error_type: str, threshold: int = 5) -> bool:
        """Check if error rate has spiked"""
        with self.lock:
            if error_type in self.error_stats:
                return self.error_stats[error_type].count >= threshold
        return False


error_analytics = ErrorAnalyticsManager()


class RetryHandler:
    """Handle retry logic with exponential backoff"""

    MAX_RETRIES = 3
    INITIAL_DELAY = 0.5  # seconds
    MAX_DELAY = 5  # seconds

    @classmethod
    def should_retry(cls, error: Exception, attempt: int) -> bool:
        """Determine if error should be retried"""
        if attempt >= cls.MAX_RETRIES:
            return False

        # Retry on transient errors (network, timeout)
        transient_errors = (TimeoutError, ConnectionError, IOError)
        return isinstance(error, transient_errors)

    @classmethod
    def get_retry_delay(cls, attempt: int) -> float:
        """Get delay before next retry (exponential backoff)"""
        delay = cls.INITIAL_DELAY * (2 ** attempt)
        return min(delay, cls.MAX_DELAY)


# ============================================================================
# Language & Voice Configuration
# ============================================================================

SUPPORTED_LANGUAGES = {
    "hi": {"name": "Hindi", "label": "हिंदी"},
    "bho": {"name": "Bhojpuri", "label": "भोजपुरी"},
    "mr": {"name": "Marathi", "label": "मराठी"},
    "gu": {"name": "Gujarati", "label": "ગુજરાતી"},
    "kn": {"name": "Kannada", "label": "ಕನ್ನಡ"},
    "te": {"name": "Telugu", "label": "తెలుగు"},
    "ta": {"name": "Tamil", "label": "தமிழ்"},
    "en": {"name": "English", "label": "English"},
}

# Allowed safe voice assistant intents
SAFE_VOICE_INTENTS = {
    "crop_health",
    "weather_alert",
    "fertilizer_guide",
    "irrigation_advice",
    "yield_prediction",
    "pest_management",
    "general_query",
}

# Dangerous command injection patterns
COMMAND_INJECTION_PATTERNS = [
    r";",
    r"&&",
    r"\|\|",
    r"`.*`",
    r"\$\(.*\)",
    r"rm\s+-rf",
    r"sudo",
    r"wget\s+",
    r"curl\s+",
    r"chmod\s+",
    r"exec\s*\(",
    r"eval\s*\(",
]

# Query intent mapping for voice commands
INTENT_PATTERNS = {
    "crop_health": [
        r"(?:meri|mere|mera)\s+(?:fasal|crop|paudhe?)\s+(?:ko\s+)?kya\s+(?:problem|issue|bimari)",
        r"(?:fasal|crop)\s+(?:se)?(?:prega|problem|disease)",
        r"(?:fasal|crop|paudha?)\s+(?:peedle|sick|halki|kamzor)",
        r"what.*problem.*my.*crop",
        r"why.*crop.*dying",
        r"(?:मेरी|मेरे|मेरा|माझी)\s+(?:फसल|पिके?|पौधे?)\s+(?:ko|को|ला\s+)?(?:kya|क्या|कया|काय|समस्या|बीमारी|परेशानी|आजारी)",
        r"समस्या",
        r"बीमारी",
    ],
    "weather_alert": [
        r"(?:mausam|weather)\s+(?:kaisa|kya|how)",
        r"(?:baarish|rain|tufaan|storm)\s+(?:aa|aayega|aayega)",
        r"(?:temperature|garmi|garmi)\s+(?:kitni|how much)",
        r"weather.*today|tomorrow|this week",
        r"मौसम",
        r"हवामान",
        r"बारिश",
    ],
    "fertilizer_guide": [
        r"(?:khad|fertilizer|nutrients?)\s+(?:kaunsi|kaun|which)",
        r"(?:fasal|crop)\s+(?:ke|ko)\s+(?:liye|for)\s+(?:khad|fertilizer)",
        r"(?:nutrient|nitrogen|phosphorus|potassium)\s+guidance",
        r"what.*fertilizer.*my.*crop",
        r"which.*fertilizer",
        r"fertilizer.*wheat",
        r"fertilizer",
        r"खाद",
        r"खत",
    ],
    "irrigation_advice": [
        r"(?:pani|water)\s+(?:kitna|how much|when)",
        r"(?:sinchai|irrigation)\s+(?:schedule|table)",
        r"(?:how|when)\s+to\s+irrigate",
        r"सिंचाई",
        r"पानी",
        r"पाणी",
    ],
    "yield_prediction": [
        r"(?:paidavaari|yield|production)\s+(?:kitni|how much)",
        r"(?:expected|munday|aashayit)\s+(?:paidavaari|yield)",
        r"(?:crop)\s+(?:utpadan|production)\s+forecast",
        r"पैदावार",
        r"उत्पादन",
    ],
    "pest_management": [
        r"(?:keeda|pest|insect|bug)\s+(?:se|from)\s+(?:kaise|how)",
        r"(?:pest|कीड़े)\s+control\s+(?:method|tarika)",
        r"कीड़ों",
        r"किडे",
        r"कीट",
    ],
}

def validate_voice_command(transcript: str) -> str:
    """
    Validate and sanitize voice commands to prevent
    command injection and unauthorized execution.
    """

    sanitized = transcript.strip().lower()

    for pattern in COMMAND_INJECTION_PATTERNS:
        if re.search(pattern, sanitized):
            logger.warning(
                "Potential command injection attempt detected: %s",
                sanitized,
            )

            raise ValueError(
                "Potential command injection detected"
            )

    return sanitized


def validate_voice_intent(intent: str) -> str:
    """
    Restrict execution to approved voice assistant intents.
    """

    if intent not in SAFE_VOICE_INTENTS:
        raise ValueError(
            f"Unauthorized voice intent: {intent}"
        )

    return intent

# Response templates in multiple languages
RESPONSE_TEMPLATES = {
    "hi": {
        "crop_health": "आपकी {crop} में {disease} का संकेत है। सुझाव: {advice}",
        "weather_alert": "मौसम अपडेट: {condition}। सावधानी: {warning}",
        "fertilizer": "आपकी {crop} को {fertilizer} की आवश्यकता है। मात्रा: {dose}",
        "irrigation": "सिंचाई का समय: {schedule}। मात्रा: {quantity}",
        "greeting": "नमस्ते! मैं आपके खेत के लिए यहाँ हूँ।",
        "error": "क्षमा करें, मुझे समझ नहीं आया। कृपया दोहराएं।",
    },
    "bho": {
        "crop_health": "आपरे {crop} में {disease} के लच्छन बा। सलाह: {advice}",
        "weather_alert": "मौसम की खबर: {condition}। सावधान: {warning}",
        "fertilizer": "आपरे {crop} को {fertilizer} चाहिए। मात्रा: {dose}",
        "irrigation": "पानी का वक्त: {schedule}। मात्रा: {quantity}",
        "greeting": "राम राम! मैं आपरे खेत के लिए हूँ।",
        "error": "माफ करिहे, मुझे समझ न आइल। फिर से कहिहे।",
    },
    "mr": {
        "crop_health": "तुमच्या {crop} ला {disease} दिसत आहे। सुचना: {advice}",
        "weather_alert": "हवामान अपडेट: {condition}। सावधानता: {warning}",
        "fertilizer": "तुमच्या {crop} ला {fertilizer} हवे. प्रमाण: {dose}",
        "irrigation": "सिंचन वेळ: {schedule}. प्रमाण: {quantity}",
        "greeting": "नमस्कार! मी तुमच्या शेतीसाठी येथे आहे.",
        "error": "क्षमस्व, मला समजले नाही. कृपया पुन्हा सांगा.",
    },
    "en": {
        "crop_health": "Your {crop} shows signs of {disease}. Advice: {advice}",
        "weather_alert": "Weather Update: {condition}. Warning: {warning}",
        "fertilizer": "Your {crop} needs {fertilizer}. Dosage: {dose}",
        "irrigation": "Irrigation Schedule: {schedule}. Amount: {quantity}",
        "greeting": "Hello! I'm here to help with your farm.",
        "error": "Sorry, I didn't understand. Please repeat.",
    },
}

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class VoiceInput:
    """Represents voice input from user"""
    audio_bytes: bytes
    language_code: str
    confidence: float = 0.0
    transcript: str = ""
    intent: Optional[str] = None


@dataclass
class VoiceResponse:
    """Represents voice response to user"""
    text: str
    language_code: str
    audio_bytes: Optional[bytes] = None
    intent: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    offline_available: bool = True


@dataclass
class VoiceSession:
    """Session tracking for voice interactions"""
    session_id: str
    user_id: str
    language_code: str
    start_time: str
    last_query: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    offline_mode: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


# ============================================================================
# Offline Language Model
# ============================================================================

class OfflineLanguageModel:
    """Lightweight offline language understanding"""
    
    def __init__(self):
        self.intent_patterns = INTENT_PATTERNS
        self.language_models = self._init_language_models()
    
    def _init_language_models(self) -> Dict[str, Dict]:
        """Initialize offline language models"""
        return {
            "hi": {"vocab_size": 5000, "model_type": "rule_based"},
            "bho": {"vocab_size": 3000, "model_type": "rule_based"},
            "mr": {"vocab_size": 4000, "model_type": "rule_based"},
            "en": {"vocab_size": 8000, "model_type": "rule_based"},
        }
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize input text by:
        1. Converting to lowercase
        2. Replacing line breaks, tabs, and carriage returns with spaces
        3. Replacing multiple spaces with a single space
        4. Removing typical punctuation marks
        """
        if not text:
            return ""
        
        # Lowercase
        normalized = text.lower()
        
        # Replace line breaks, tabs, and carriage returns with spaces
        normalized = re.sub(r'[\r\n\t]+', ' ', normalized)
        
        # Replace multiple spaces with a single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove typical punctuation marks
        normalized = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()?\"\'’]', '', normalized)
        
        return normalized.strip()

    def detect_intent(self, text: str) -> Tuple[str, float]:
        """
        Detect intent from input text using offline patterns
        Returns: (intent, confidence)
        """
        normalized_text = self._normalize_text(text)
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, normalized_text):
                    return intent, 0.85  # Offline confidence
        
        return "general_query", 0.5
    
    def extract_entities(self, text: str, intent: str) -> Dict[str, str]:
        """Extract entities from text based on intent"""
        entities = {}
        normalized_text = self._normalize_text(text)
        
        # Simple entity extraction with expanded vocabulary (Devanagari + Latin)
        crops = [
            "rice", "wheat", "sugarcane", "cotton", "maize", "chawal", "gehun",
            "धान", "चावल", "गेहूँ", "गेहूं", "गन्ना", "कपास", "मक्का", "भात", "गहू", "ऊस"
        ]
        for crop in crops:
            if crop in normalized_text:
                entities["crop"] = crop
                break
        
        diseases = [
            "fungal", "bacterial", "viral", "blight", "rust", "leaf spot",
            "फंगल", "बैक्टीरियल", "वायरस", "ब्लाइट", "रस्ट", "धब्बा", "करपा", "तांबेरा"
        ]
        for disease in diseases:
            if disease in normalized_text:
                entities["disease"] = disease
                break
        
        return entities


# ============================================================================
# Voice Assistant Core
# ============================================================================

SESSION_TTL = 1800       # 30 min inactivity timeout
MAX_SESSIONS = 1000       # hard cap to prevent unbounded growth


class VoiceAssistant:
    """Main voice assistant for farmers"""
    SESSION_TIMEOUT_MINUTES = 30
    MAX_HISTORY_SIZE = 20
    def __init__(self, offline_mode: bool = True):
        self.offline_mode = offline_mode
        self.language_model = OfflineLanguageModel()
        self.sessions: Dict[str, VoiceSession] = {}
        self._session_lock = threading.Lock()

    def _evict_stale_sessions(self):
        """Remove expired and excess sessions."""
        now = datetime.now()
        cutoff = now.timestamp() - SESSION_TTL
        stale_keys = []
        for sid, sess in self.sessions.items():
            ts = sess.last_activity or sess.start_time
            try:
                last = datetime.fromisoformat(ts).timestamp()
            except (ValueError, TypeError):
                last = 0
            if last < cutoff:
                stale_keys.append(sid)
        for sid in stale_keys:
            del self.sessions[sid]
        if len(self.sessions) > MAX_SESSIONS:
            sorted_sids = sorted(
                self.sessions.keys(),
                key=lambda s: self.sessions[s].start_time or "",
            )
            for sid in sorted_sids[:len(self.sessions) - MAX_SESSIONS]:
                del self.sessions[sid]
        self.offline_cache = self._init_offline_cache()
        logger.info(f"Voice Assistant initialized - Offline mode: {offline_mode}")
    
    def _init_offline_cache(self) -> Dict[str, Any]:
        """Initialize offline knowledge cache"""
        return {
            "crop_diseases": {
                "rice": ["blast", "sheath blight", "brown spot", "leaf scald"],
                "wheat": ["leaf rust", "stem rust", "powdery mildew", "septoria"],
                "cotton": ["wilt", "leaf curl", "boll rot", "thrips damage"],
                "maize": ["gray leaf spot", "southern corn leaf blight", "common rust"],
            },
            "fertilizer_recommendations": {
                "rice": {"nitrogen": "40-60 kg/acre", "phosphorus": "30-40 kg/acre", "potassium": "20-30 kg/acre"},
                "wheat": {"nitrogen": "60-80 kg/acre", "phosphorus": "40-50 kg/acre", "potassium": "30-40 kg/acre"},
                "cotton": {"nitrogen": "50-70 kg/acre", "phosphorus": "35-45 kg/acre", "potassium": "40-50 kg/acre"},
            },
            "irrigation_schedules": {
                "rice": {"frequency": "Every 5-7 days", "amount": "40-50 mm"},
                "wheat": {"frequency": "Every 15-20 days", "amount": "50-60 mm"},
                "sugarcane": {"frequency": "Every 10-15 days", "amount": "50-75 mm"},
            },
            # Structured weather alerts keyed by (crop, condition) so that
            # _select_weather_alert() can return a contextually relevant
            # message instead of always returning the same static string.
            "weather_alerts": {
                "default": {
                    "heat":     "High temperature alert — increase irrigation frequency and provide shade where possible.",
                    "rain":     "Excessive rainfall expected — prepare drainage channels and watch for waterlogging.",
                    "wind":     "Strong winds forecast — secure loose farm structures and stake tall crops.",
                    "frost":    "Frost warning — protect seedlings with covers and avoid night-time irrigation.",
                    "drought":  "Dry spell ahead — conserve soil moisture and plan supplemental irrigation.",
                    "general":  "Monitor local weather closely — conditions may change rapidly this season.",
                },
                "rice": {
                    "heat":     "Heat stress risk for rice — maintain standing water in fields to cool roots.",
                    "rain":     "Heavy rain can cause blast disease in rice — ensure drainage and apply fungicide.",
                    "wind":     "Strong winds may lodge rice at heading stage — monitor and provide support.",
                    "frost":    "Rice is frost-sensitive — move nursery trays indoors and delay transplanting.",
                    "drought":  "Rice needs consistent water — prioritise irrigation at tillering and flowering stages.",
                    "general":  "Check rice fields daily during active growth — scout for pests after any rainfall.",
                },
                "wheat": {
                    "heat":     "Heat at grain-fill will shrink wheat yield — harvest early if temperatures exceed 35°C.",
                    "rain":     "Post-anthesis rain raises yellow-rust risk — spray preventive fungicide on wheat.",
                    "wind":     "Wind may cause lodging in heavy wheat crops — avoid excess nitrogen application.",
                    "frost":    "Frost at flowering stage damages wheat — apply light irrigation to reduce chill effect.",
                    "drought":  "Wheat needs water at crown-root, tillering, and grain-fill — irrigate at these stages.",
                    "general":  "Wheat is in a critical growth phase — watch for aphids and powdery mildew.",
                },
                "cotton": {
                    "heat":     "High heat causes boll shedding in cotton — increase irrigation intervals and mulch rows.",
                    "rain":     "Heavy rain promotes bollworm and fungal spread in cotton — inspect and spray as needed.",
                    "wind":     "Winds can spread whitefly and leaf-curl virus — use windbreak barriers if available.",
                    "frost":    "Cotton is frost-intolerant — trigger early harvest of open bolls before freezing nights.",
                    "drought":  "Cotton squares drop under water stress — maintain soil moisture at 50–60% field capacity.",
                    "general":  "Monitor cotton for pink bollworm and sucking pests during humid periods.",
                },
                "maize": {
                    "heat":     "Maize pollen viability drops above 35°C — irrigate during morning and evening.",
                    "rain":     "Waterlogged maize fields cause root rot — open drainage furrows immediately.",
                    "wind":     "Wind causes lodging in maize at tasseling — earthen up around stem bases.",
                    "frost":    "Maize is frost-sensitive at seedling stage — delay sowing if frost risk remains.",
                    "drought":  "Critical irrigation period for maize is tasseling to silking — do not miss this.",
                    "general":  "Scout maize for fall armyworm and apply control measures within 3 days of detection.",
                },
                "sugarcane": {
                    "heat":     "High heat increases evapotranspiration in sugarcane — irrigate every 7–10 days.",
                    "rain":     "Waterlogging stunts sugarcane — open drainage and earthen up around the crop.",
                    "wind":     "Sugarcane can lodge in strong winds — stake or tie tall stalks in exposed areas.",
                    "frost":    "Frost kills sugarcane growing points — harvest mature cane before hard frost.",
                    "drought":  "Drought reduces sugar content — maintain deficit irrigation at grand growth phase.",
                    "general":  "Monitor sugarcane for internode borer and early shoot borer during monsoon.",
                },
            },
        }
    
    def create_session(self, user_id: str, language_code: str = "hi") -> VoiceSession:
        """Create new voice session"""
        from uuid import uuid4
        session_id = str(uuid4())
        now = datetime.now().isoformat()
        session = VoiceSession(
            session_id=session_id,
            user_id=user_id,
            language_code=language_code,
            start_time=now,
            last_activity=now,
            context={},
            offline_mode=self.offline_mode,
        )
        with self._session_lock:
            self._evict_stale_sessions()
            self.sessions[session_id] = session
        self.cache_manager.save_session(session)
        return session
    
    def _validate_session(self, session: VoiceSession) -> bool:
        return (
            bool(session.session_id)
            and bool(session.user_id)
            and session.language_code in SUPPORTED_LANGUAGES
        )
    
    def _is_session_expired(self, session: VoiceSession) -> bool:
        last_activity = datetime.fromisoformat(session.last_activity)
        age = datetime.now() - last_activity
        return age.total_seconds() > self.SESSION_TIMEOUT_MINUTES * 60
    
    def process_voice_input(
        self,
        voice_input: VoiceInput,
        session_id: str,
        context: Optional[Dict] = None,
    ) -> VoiceResponse:
        """
        Process voice input and generate response
        
        Flow:
        1. Detect language (offline)
        2. Detect intent (offline)
        3. Extract entities
        4. Generate response (from cache or online)
        5. Convert to speech (if audio available)
        """
        with self._session_lock:
            if session_id not in self.sessions:
                raise ValueError(f"Invalid session: {session_id}")
            self._evict_stale_sessions()
            if session_id not in self.sessions:
                raise ValueError(f"Session expired: {session_id}")
            session = self.sessions[session_id]
            session.last_activity = datetime.now().isoformat()
        
        # Step 1: Transcribe audio (offline fallback)
        if not voice_input.transcript:
            voice_input.transcript = self._transcribe_offline(voice_input)
        
        # Step 2: Validate transcript against injection attacks
        validated_transcript = validate_voice_command(
            voice_input.transcript
        )

        # Step 3: Detect and sandbox intent
        intent, confidence = self.language_model.detect_intent(
            validated_transcript
        )

        intent = validate_voice_intent(intent)

        voice_input.intent = intent

        # Step 4: Extract entities
        entities = self.language_model.extract_entities(
            validated_transcript,
            intent,
        )
        
        # Step 4: Generate response
        response_text = self._generate_response(
            intent=intent,
            entities=entities,
            language_code=session.language_code,
            context=context or session.context,
        )
        
        # Step 5: Create response
        response = VoiceResponse(
            text=response_text,
            language_code=session.language_code,
            intent=intent,
            offline_available=self.offline_mode,
            metadata={
                "confidence": confidence,
                "entities": entities,
                "timestamp": datetime.now().isoformat(),
            },
        )
        
        # Update session context — per-session lock avoids blocking other sessions
        with session.lock:
            session.last_query = voice_input.transcript
            session.context = context or {}
        
        return response
    
    def _transcribe_offline(self, voice_input: VoiceInput) -> str:
        """Offline audio transcription (fallback)"""
        # This is a placeholder - in production, use:
        # - SpeechRecognition library
        # - Vosk (offline STT)
        # - PocketSphinx (lightweight)
        logger.warning("Using offline transcription (limited accuracy)")
        return "[offline transcription not available]"
    
    def _generate_response(
        self,
        intent: str,
        entities: Dict[str, str],
        language_code: str,
        context: Dict[str, Any],
    ) -> str:
        """Generate response based on intent and entities"""
        
        templates = RESPONSE_TEMPLATES.get(language_code, RESPONSE_TEMPLATES["en"])
        
        if intent == "crop_health":
            crop = entities.get("crop", "आपकी फसल")
            disease = entities.get("disease", "एक समस्या")
            advice = self._get_disease_advice(crop, disease, language_code)
            return templates["crop_health"].format(crop=crop, disease=disease, advice=advice)
        
        elif intent == "weather_alert":
            alert_msg, warning_text = self._select_weather_alert(
                entities=entities,
                context=context,
                language_code=language_code,
            )
            return templates["weather_alert"].format(
                condition=alert_msg,
                warning=warning_text,
            )
        
        elif intent == "fertilizer_guide":
            crop = entities.get("crop", "गेहूँ")
            fert_rec = self.offline_cache["fertilizer_recommendations"].get(
                crop.lower(), {"nitrogen": "60 kg/acre", "phosphorus": "40 kg/acre"}
            )
            return templates["fertilizer"].format(
                crop=crop,
                fertilizer="DAP और यूरिया",
                dose=fert_rec.get("nitrogen", "60 kg/acre"),
            )
        
        elif intent == "irrigation_advice":
            crop = entities.get("crop", "धान")
            irr_sched = self.offline_cache["irrigation_schedules"].get(crop.lower(), {})
            return templates["irrigation"].format(
                schedule=irr_sched.get("frequency", "हर 10 दिन में"),
                quantity=irr_sched.get("amount", "50 मिमी"),
            )
        elif intent == "yield_prediction":
            return (
                "अनुमानित उत्पादन सामान्य से अच्छा हो सकता है।"
                if language_code == "hi"
                else "Expected crop yield looks stable and healthy."
            )

        elif intent == "pest_management":
            return (
                "कीटनाशक का नियंत्रित छिड़काव करें।"
                if language_code == "hi"
                else "Use recommended pesticide spray in controlled quantity."
            )
        
        return templates.get("greeting", "नमस्ते! मैं आपके लिए यहाँ हूँ।")
    
    def _select_weather_alert(
        self,
        entities: Dict[str, str],
        context: Dict[str, Any],
        language_code: str,
    ) -> Tuple[str, str]:
        """Return a contextually relevant (condition, warning) pair from the
        offline weather-alert cache.

        Selection priority:
        1. Crop-specific entry when a crop entity is present.
        2. Season-conditioned alert when season is in context.
        3. Generic alert keyed by detected condition keywords in context.
        4. Absolute fallback to the 'general' message for the matched crop
           (or 'default' if no crop was extracted).
        """
        alerts = self.offline_cache["weather_alerts"]

        # Determine which crop bucket to use.
        crop = (entities.get("crop") or context.get("crop") or "").lower()
        crop_alerts = alerts.get(crop) if crop in alerts else None
        fallback_alerts = alerts["default"]

        # Map season to a likely weather condition so we can serve a more
        # relevant alert even when no explicit condition is in context.
        season = (context.get("season") or "").lower()
        season_condition_map: Dict[str, str] = {
            "kharif":  "rain",
            "rabi":    "frost",
            "zaid":    "heat",
            "summer":  "heat",
            "winter":  "frost",
            "monsoon": "rain",
        }

        # Detect an explicit condition from context (e.g. passed by the
        # router after calling the live weather service).
        explicit_condition = (context.get("weather_condition") or "").lower()

        # Condition priority: explicit > season-derived > general.
        condition_keys = [
            k for k in (explicit_condition, season_condition_map.get(season))
            if k
        ]

        # Pick the most specific alert message available.
        alert_msg: str = ""
        for cond in condition_keys:
            if crop_alerts and cond in crop_alerts:
                alert_msg = crop_alerts[cond]
                break
            if cond in fallback_alerts:
                alert_msg = fallback_alerts[cond]
                break

        # Absolute fallback.
        if not alert_msg:
            if crop_alerts:
                alert_msg = crop_alerts.get("general", fallback_alerts["general"])
            else:
                alert_msg = fallback_alerts["general"]

        # Build a localised warning suffix.
        warning_map = {
            "hi": "सावधान रहें और नजदीकी कृषि अधिकारी से सम्पर्क करें।",
            "bho": "सावधान रहीं और खेत पर नजर राखें।",
            "mr": "सावध रहा आणि स्थानिक कृषी सल्लागाराशी संपर्क साधा.",
            "en": "Stay alert and contact your local agricultural office if conditions worsen.",
        }
        warning_text = warning_map.get(language_code, warning_map["en"])

        logger.info(
            "Weather alert selected: crop=%r condition_keys=%r alert=%r",
            crop or "default", condition_keys, alert_msg[:60],
        )
        return alert_msg, warning_text

    def _get_disease_advice(self, crop: str, disease: str, language_code: str) -> str:
        """Get disease management advice"""
        advice_map = {
            "fungal": "कवकनाशी दवा का उपयोग करें" if language_code == "hi" else "Use fungicide spray",
            "bacterial": "बैक्टीरिया रोधी दवा लगाएं" if language_code == "hi" else "Apply bactericide",
            "viral": "संक्रमित पौधे हटाएं" if language_code == "hi" else "Remove infected plants",
            "pest": "कीटनाशक दवा का छिड़काव करें" if language_code == "hi" else "Use pesticide",
        }
        return advice_map.get(disease, "विशेषज्ञ से सलाह लें" if language_code == "hi" else "Consult expert")
    
    def text_to_speech(
        self,
        text: str,
        language_code: str,
    ) -> Optional[bytes]:
        """Convert text to speech (offline-capable)"""
        # Placeholder - in production, use:
        # - pyttsx3 (offline)
        # - gTTS (online with offline cache)
        logger.info(f"Text-to-speech: {text[:50]}... ({language_code})")
        return None  # Audio generation requires additional libraries
    
    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """Retrieve session history"""
        with self._session_lock:
            if session_id not in self.sessions:
                raise ValueError(f"Invalid session: {session_id}")
            self._evict_stale_sessions()
            if session_id not in self.sessions:
                raise ValueError(f"Session expired: {session_id}")
            session = self.sessions[session_id]
        with session.lock:
            return {
            "session_id": session_id,
            "user_id": session.user_id,
            "language": session.language_code,
            "start_time": session.start_time,
            "last_query": session.last_query,
            "offline_mode": session.offline_mode,
            "last_activity": session.last_activity,
            "conversation_history": session.conversation_history,
        }


# ============================================================================
# Language Detection
# ============================================================================

def detect_language(text: str) -> str:
    """
    Detect language from text using Unicode ranges and lexical heuristics
    Returns language code
    """
    text_lower = text.lower()
    
    # Devanagari range (Hindi, Marathi, etc.)
    if any('\u0900' <= char <= '\u097F' for char in text):
        # Lexical heuristics for language detection
        hindi_keywords = ["है", "हैं", "को", "क्या", "मेरी", "मेरा", "मेरे", "की", "का", "के", "से", "में", "करें", "बीमारी", "फसल"]
        marathi_keywords = ["आहे", "आहेत", "माझी", "माझा", "माझे", "पिके", "काय", "कधी", "करू", "करून", "रोग", "पीक"]
        bhojpuri_keywords = ["बा", "बानी", "रउआ", "तोहार", "हमरा", "कथि", "खेती"]
        
        # Check matching counts
        hi_score = sum(1 for kw in hindi_keywords if kw in text_lower)
        mr_score = sum(1 for kw in marathi_keywords if kw in text_lower)
        bho_score = sum(1 for kw in bhojpuri_keywords if kw in text_lower)
        
        # Heuristics based on character/matra occurrence
        if 'ि' in text or 'ु' in text:  # Hindi-specific marks
            hi_score += 1
            
        if mr_score > hi_score and mr_score > bho_score:
            return "mr"
        elif bho_score > hi_score and bho_score > mr_score:
            return "bho"
        else:
            return "hi"
    
    # Gujarati
    if any('\u0A80' <= char <= '\u0AFF' for char in text):
        return "gu"
    
    # Kannada
    if any('\u0C80' <= char <= '\u0CFF' for char in text):
        return "kn"
    
    # Telugu
    if any('\u0C00' <= char <= '\u0C7F' for char in text):
        return "te"
    
    # Tamil
    if any('\u0B80' <= char <= '\u0BFF' for char in text):
        return "ta"
    
    return "en"


# ============================================================================
# Voice Query Analyzer
# ============================================================================

class VoiceQueryAnalyzer:
    """Analyze voice queries for context and clarity"""
    
    @staticmethod
    def analyze(query: str, language_code: str) -> Dict[str, Any]:
        """Analyze query for quality and context"""
        return {
            "query": query,
            "length": len(query.split()),
            "language": language_code,
            "has_crop_mention": any(crop in query.lower() for crop in ["rice", "wheat", "cotton", "गेहू", "धान"]),
            "has_disease_mention": any(d in query.lower() for d in ["disease", "bimari", "problem", "issue"]),
            "clarity_score": 0.8 if len(query) > 3 else 0.4,
        }


# ============================================================================
# Offline Cache Manager
# ============================================================================

class OfflineCacheManager:
    """Manage offline knowledge cache"""
    
    def __init__(self, cache_dir: str = "./voice_assistant_cache"):
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def save_cache(self, cache_data: Dict[str, Any], key: str = "offline_data"):
        """Save cache to disk"""
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Cache saved: {cache_path}")
        except Exception as e:
            logger.error(f"Cache save error: {e}")
    
    def load_cache(self, key: str = "offline_data") -> Dict[str, Any]:
        """Load cache from disk"""
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Cache load error: {e}")
        return {}
    
    def save_session(self, session: VoiceSession):
        self.save_cache(
            asdict(session),
            key=f"session_{session.session_id}"
        )

    def load_session(self, session_id: str):
        return self.load_cache(
            key=f"session_{session_id}"
        )
# Voice assistant error handling improved
