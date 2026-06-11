import logging
"""
Quick Start - Voice Assistant Examples
Demonstrates how to use the Voice Assistant API and components
"""

# ============================================================================
# BACKEND USAGE EXAMPLES
# ============================================================================

# --- Example 1: Initialize Voice Assistant ---
from voice_assistant import VoiceAssistant, OfflineCacheManager

# Create assistant instance
va = VoiceAssistant(offline_mode=True)
cache_manager = OfflineCacheManager(cache_dir="./voice_assistant_cache")

logging.info("✓ Voice Assistant initialized in offline mode")


# --- Example 2: Create User Session ---
# Each user gets a session for managing conversations
session = va.create_session(user_id="farmer_001", language_code="hi")
logging.info(f"✓ Session created: {session.session_id}")


# --- Example 3: Process Simple Query ---
from voice_assistant import VoiceInput

query_text = "मेरी फसल को क्या समस्या है?"  # Hindi: "What problem does my crop have?"
voice_input = VoiceInput(
    audio_bytes=b"",  # In production, this would contain audio data
    language_code="hi",
    transcript=query_text,
)

response = va.process_voice_input(voice_input, session.session_id)
logging.info(f"\n📢 Query: {query_text}")
logging.info(f"🤖 Response: {response.text}")
logging.info(f"🎯 Intent: {response.intent}")


# --- Example 4: Detect Intent Automatically ---
from voice_assistant import detect_language

text = "गेहूँ को कौन सी खाद दें?"  # "Which fertilizer for wheat?"
detected_lang = detect_language(text)
intent, confidence = va.language_model.detect_intent(text)

logging.info(f"\n🔍 Language detected: {detected_lang}")
logging.info(f"💡 Intent: {intent} (confidence: {confidence})")


# --- Example 5: Analyze Query Quality ---
from voice_assistant import VoiceQueryAnalyzer

query = "मेरी धान में समस्या है"
analysis = VoiceQueryAnalyzer.analyze(query, "hi")

logging.info(f"\n📊 Query Analysis:")
logging.info(f"  - Clarity: {analysis['clarity_score']}")
logging.info(f"  - Has crop mention: {analysis['has_crop_mention']}")
logging.info(f"  - Has disease mention: {analysis['has_disease_mention']}")


# --- Example 6: Access Offline Knowledge Base ---
crops = list(va.offline_cache["crop_diseases"].keys())
logging.info(f"\n🌾 Crops in knowledge base: {crops}")

rice_diseases = va.offline_cache["crop_diseases"]["rice"]
logging.info(f"Rice diseases: {rice_diseases}")


# --- Example 7: Handle Multiple Languages ---
languages = ["hi", "en", "mr", "gu"]

queries = {
    "hi": "मेरी फसल बीमार है",  # Hindi
    "en": "My crop is sick",   # English
    "mr": "माझी पिके आजारी आहे",  # Marathi
    "gu": "મારી ફસલ બીમાર છે",  # Gujarati
}

for lang in languages:
    if lang in queries:
        session = va.create_session("user", lang)
        voice_input = VoiceInput(b"", lang, queries[lang])
        response = va.process_voice_input(voice_input, session.session_id)
        logging.info(f"\n{lang.upper()}: {response.text[:100]}...")


# --- Example 8: Extract Entities from Query ---
query = "मेरे धान में blast बीमारी है"
intent, _ = va.language_model.detect_intent(query)
entities = va.language_model.extract_entities(query, intent)

logging.info(f"\n🔑 Entities extracted:")
logging.info(f"  - Crop: {entities.get('crop', 'Not found')}")
logging.info(f"  - Disease: {entities.get('disease', 'Not found')}")


# ============================================================================
# API ENDPOINT EXAMPLES (Using curl or requests)
# ============================================================================

"""
Run these commands in terminal or Python requests library

1. CREATE SESSION
curl -X POST http://localhost:8000/api/voice/sessions/create \\
  -H "Content-Type: application/json" \\
  -d '{
    "user_id": "farmer_001",
    "language_code": "hi"
  }'

2. PROCESS VOICE QUERY
curl -X POST http://localhost:8000/api/voice/query \\
  -H "Content-Type: application/json" \\
  -d '{
    "transcript": "मेरी फसल को क्या समस्या है?",
    "language_code": "hi",
    "session_id": "SESSION_ID_FROM_ABOVE",
    "user_id": "farmer_001"
  }'

3. LIST LANGUAGES
curl http://localhost:8000/api/voice/languages

4. GET OFFLINE CACHE
curl http://localhost:8000/api/voice/offline-cache

5. ANALYZE QUERY
curl -X POST http://localhost:8000/api/voice/query-analyze \\
  -H "Content-Type: application/json" \\
  -d '{
    "transcript": "गेहूँ को खाद दें?",
    "language_code": "hi",
    "user_id": "farmer_001"
  }'

6. GET SESSION HISTORY
curl http://localhost:8000/api/voice/sessions/SESSION_ID

7. LIST SUPPORTED INTENTS
curl http://localhost:8000/api/voice/supported-intents

8. SYNC CACHE
curl -X POST http://localhost:8000/api/voice/sync-cache
"""


# ============================================================================
# REACT COMPONENT EXAMPLE
# ============================================================================

"""
Using VoiceAssistant.jsx in your React app:

import React from 'react';
import VoiceAssistant from './VoiceAssistant';

export default function FarmerDashboard() {
  return (
    <div>
      <h1>Fasal Saathi - Farmer Dashboard</h1>
      <VoiceAssistant />
    </div>
  );
}

Features:
- Real-time speech recognition
- Language selection
- Chat history
- Error handling
- Offline indicator
- Text-to-speech
"""


# ============================================================================
# TESTING EXAMPLES
# ============================================================================

"""
Run tests with pytest:

# All tests
pytest test_voice_assistant.py -v

# Specific test class
pytest test_voice_assistant.py::TestIntentDetection -v

# Specific test
pytest test_voice_assistant.py::TestIntentDetection::test_crop_health_intent_hindi -v

# With coverage
pytest test_voice_assistant.py --cov=voice_assistant
"""


# ============================================================================
# COMMON QUERIES FOR TESTING
# ============================================================================

test_queries = {
    "hindi": {
        "crop_health": [
            "मेरी फसल को क्या समस्या है?",
            "मेरे धान में बीमारी है",
            "गेहूँ पीला पड़ रहा है",
        ],
        "fertilizer": [
            "गेहूँ को कौन सी खाद दें?",
            "धान के लिए खाद का सुझाव दें",
            "पोषक तत्व कितने चाहिए?",
        ],
        "irrigation": [
            "सिंचाई कब करें?",
            "पानी कितना देना चाहिए?",
            "सिंचाई की आवृत्ति क्या होनी चाहिए?",
        ],
        "weather": [
            "मौसम कैसा रहेगा?",
            "बारिश आने वाली है?",
            "तापमान कितना बढ़ेगा?",
        ],
        "pest": [
            "कीड़ों से कैसे बचें?",
            "कीटनाशक कौन सा लगाएं?",
            "शूट फ्लाई कैसे मारें?",
        ],
    },
    "english": {
        "crop_health": [
            "What problem does my crop have?",
            "My rice has a disease",
            "Why is my wheat turning yellow?",
        ],
        "fertilizer": [
            "Which fertilizer for wheat?",
            "Suggest fertilizer for rice",
            "How much nutrient do I need?",
        ],
        "irrigation": [
            "When should I irrigate?",
            "How much water to give?",
            "What should be irrigation frequency?",
        ],
        "weather": [
            "How will the weather be?",
            "Will it rain?",
            "Will temperature increase?",
        ],
    },
}


# ============================================================================
# INTEGRATION WORKFLOW EXAMPLE
# ============================================================================

def complete_farmer_workflow():
    """Complete workflow: session creation -> query -> response"""
    
    logging.info("\n" + "="*50)
    logging.info("COMPLETE FARMER WORKFLOW")
    logging.info("="*50)
    
    # Step 1: Initialize
    logging.info("\n1️⃣  Initializing Voice Assistant...")
    assistant = VoiceAssistant(offline_mode=True)
    logging.info("   ✓ Ready in offline mode")
    
    # Step 2: Create session
    logging.info("\n2️⃣  Creating session for farmer...")
    session = assistant.create_session(user_id="farmer_001", language_code="hi")
    logging.info(f"   ✓ Session ID: {session.session_id[:8]}...")
    
    # Step 3: Process multiple queries
    logging.info("\n3️⃣  Processing queries...")
    
    queries = [
        "नमस्ते! मेरी धान में समस्या है",
        "गेहूँ के लिए खाद दें",
        "सिंचाई कब करनी चाहिए?",
    ]
    
    for i, query in enumerate(queries, 1):
        logging.info(f"\n   Query {i}: {query}")
        
        voice_input = VoiceInput(
            audio_bytes=b"",
            language_code="hi",
            transcript=query,
        )
        
        response = assistant.process_voice_input(voice_input, session.session_id)
        logging.info(f"   Intent: {response.intent}")
        logging.info(f"   Response: {response.text[:80]}...")
    
    # Step 4: Retrieve history
    logging.info("\n4️⃣  Session history:")
    history = assistant.get_session_history(session.session_id)
    logging.info(f"   User: {history['user_id']}")
    logging.info(f"   Language: {history['language']}")
    logging.info(f"   Last query: {history['last_query'][:50]}...")
    
    logging.info("\n✓ Workflow complete!")


# Run the example
if __name__ == "__main__":
    logging.info("\n" + "🎤 VOICE ASSISTANT - QUICK START EXAMPLES".center(50))
    logging.info("="*50)
    
    # Run workflow
    complete_farmer_workflow()
    
    logging.info("\n" + "="*50)
    logging.info("For more examples, check:")
    logging.info("- VOICE_ASSISTANT.md (full documentation)")
    logging.info("- test_voice_assistant.py (test cases)")
    logging.info("- backend/routers/voice_assistant.py (API endpoints)")
    logging.info("- frontend/VoiceAssistant.jsx (React component)")
    logging.info("="*50 + "\n")
