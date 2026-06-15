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
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import re

logger = logging.getLogger(__name__)

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

class VoiceAssistant:
    """Main voice assistant for farmers"""
    
    def __init__(self, offline_mode: bool = True):
        self.offline_mode = offline_mode
        self.language_model = OfflineLanguageModel()
        self.sessions: Dict[str, VoiceSession] = {}
        self._session_lock = threading.Lock()
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
            "weather_alerts": [
                "Excessive rainfall expected - prepare for waterlogging",
                "High temperature - increase irrigation frequency",
                "Strong winds - secure loose structures",
                "Frost warning - protect seedlings",
            ],
        }
    
    def create_session(self, user_id: str, language_code: str = "hi") -> VoiceSession:
        """Create new voice session"""
        from uuid import uuid4
        session_id = str(uuid4())
        session = VoiceSession(
            session_id=session_id,
            user_id=user_id,
            language_code=language_code,
            start_time=datetime.now().isoformat(),
            context={},
            offline_mode=self.offline_mode,
        )
        with self._session_lock:
            self.sessions[session_id] = session
        return session
    
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
            session = self.sessions[session_id]
        
        # Step 1: Transcribe audio (offline fallback)
        if not voice_input.transcript:
            voice_input.transcript = self._transcribe_offline(voice_input)
        
        # Step 2: Detect intent
        intent, confidence = self.language_model.detect_intent(voice_input.transcript)
        voice_input.intent = intent
        
        # Step 3: Extract entities
        entities = self.language_model.extract_entities(voice_input.transcript, intent)
        
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
        
        # Update session context
        with self._session_lock:
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
            weather = self.offline_cache["weather_alerts"][0]
            return templates["weather_alert"].format(condition=weather, warning="सावधान रहें")
        
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
        
        return templates.get("greeting", "नमस्ते! मैं आपके लिए यहाँ हूँ।")
    
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
            session = self.sessions[session_id]
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "language": session.language_code,
            "start_time": session.start_time,
            "last_query": session.last_query,
            "offline_mode": session.offline_mode,
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
