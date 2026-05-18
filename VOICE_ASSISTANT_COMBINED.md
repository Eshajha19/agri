# Voice Assistant — Complete Documentation

This single document consolidates the full voice assistant documentation into one place. It merges: `VOICE_ASSISTANT.md`, `VOICE_ASSISTANT_SETUP.md`, `VOICE_ASSISTANT_IMPLEMENTATION.md`, and `VOICE_ASSISTANT_QUICK_REFERENCE.md`.

---

## Table of Contents

1. Feature Overview
2. Architecture
3. Supported Languages & Intents
4. Installation & Setup
5. Backend & Frontend Integration
6. API Endpoints and Examples
7. Offline Mode Details & Cache
8. Usage in React
9. Performance & Optimization
10. Testing
11. Deployment
12. Troubleshooting
13. Security & Privacy
14. Extending the Assistant
15. Files Added & Implementation Summary
16. Quick Reference & Commands
17. Next Steps
18. Support & Contribution

---

## 1. Feature Overview

The Voice Assistant is a multilingual, offline-capable voice-to-text system designed to help farmers with low literacy or limited typing access interact with the Fasal Saathi platform.

Key Features:
- Multilingual Support: Hindi, Bhojpuri, Marathi, Gujarati, Kannada, Telugu, Tamil, English
- Offline Functionality: Works in low-network or no-network environments
- Intent Detection: Understands farmer queries across multiple topics
- Accessibility: Voice input and text-to-speech output

---

## 2. Architecture

Backend Components:
- `VoiceAssistant` (`voice_assistant.py`): Core class handling input processing, intent detection, entity extraction, response generation, and session management.
- `OfflineLanguageModel`: Rule-based intent detection and entity extraction.
- `OfflineCacheManager`: Persistent knowledge base for crop diseases, fertilizer recommendations, irrigation schedules, and weather alerts.
- FastAPI Router: `backend/routers/voice_assistant.py` provides REST endpoints for voice operations.

Frontend Components:
- `frontend/VoiceAssistant.jsx`: React component for real-time speech recognition, language selection, chat interface, and TTS.

---

## 3. Supported Languages & Intents

Supported Languages:
- `hi` (Hindi), `bho` (Bhojpuri), `mr` (Marathi), `gu` (Gujarati), `kn` (Kannada), `te` (Telugu), `ta` (Tamil), `en` (English)

Supported Intents:
- `crop_health` — Disease diagnosis and remedies
- `weather_alert` — Weather queries and alerts
- `fertilizer_guide` — Nutrient and fertilizer recommendations
- `irrigation_advice` — Watering schedules and amounts
- `yield_prediction` — Production forecasts
- `pest_management` — Pest control methods

---

## 4. Installation & Setup

Prerequisites:
- Python 3.8+
- Node.js 16+
- Modern browser with Web Speech API support
- Microphone access (HTTPS in production)

Install dependencies:

```bash
cd agri/
pip install -r requirements.txt
```

Frontend:

```bash
cd agri/frontend/
npm install
```

Initialize cache directory (optional):

```bash
mkdir -p voice_assistant_cache
```

---

## 5. Backend & Frontend Integration

Backend:
- Initialize in `main.py`:

```python
from voice_assistant import VoiceAssistant, OfflineCacheManager
voice_asst = VoiceAssistant(offline_mode=True)
cache_mgr = OfflineCacheManager(cache_dir="./voice_assistant_cache")
voice_assistant.init_voice_assistant(voice_asst, cache_mgr)
```

Register the router in `main.py`:

```python
app.include_router(voice_assistant.router, prefix="/api/voice", tags=["Voice Assistant"])
```

Frontend:
- Add route for `VoiceAssistant.jsx`:

```jsx
import VoiceAssistant from './VoiceAssistant';
<Route path="/voice-assistant" element={<VoiceAssistant />} />
```

---

## 6. API Endpoints and Examples

Key endpoints:
- `GET /api/voice/health` — Health check
- `GET /api/voice/languages` — List supported languages
- `POST /api/voice/sessions/create` — Create session
- `POST /api/voice/query` — Process query
- `POST /api/voice/audio-upload` — Upload audio
- `POST /api/voice/query-analyze` — Analyze query
- `GET /api/voice/offline-cache` — Get offline cache
- `POST /api/voice/sync-cache` — Sync cache
- `GET /api/voice/supported-intents` — List intents

Example — Create session:

```bash
curl -X POST http://localhost:8000/api/voice/sessions/create \
  -H "Content-Type: application/json" \
  -d '{"user_id": "farmer_001", "language_code": "hi"}'
```

Example — Process query:

```bash
curl -X POST http://localhost:8000/api/voice/query \
  -H "Content-Type: application/json" \
  -d '{"transcript": "मेरी फसल को क्या समस्या है?", "language_code": "hi", "session_id": "SESSION_ID", "user_id": "farmer_001"}'
```

---

## 7. Offline Mode Details & Cache

Offline-capable features:
- Offline speech-to-text (Vosk, PocketSphinx)
- Offline intent detection (rule-based)
- Offline TTS (pyttsx3)
- Local cache for knowledge base

Cache location: `./voice_assistant_cache/offline_data.json` (customizable)

Sync cache:

```bash
curl -X POST http://localhost:8000/api/voice/sync-cache
```

---

## 8. Usage in React

Import and mount the component:

```jsx
import VoiceAssistant from './VoiceAssistant';

export default function FarmerDashboard() {
  return (
    <div>
      <VoiceAssistant />
    </div>
  );
}
```

Features:
- Language selection
- Mic button for real-time recognition
- Chat history and TTS playback
- Offline indicator and intent badges

---

## 9. Performance & Optimization

Response times (offline):
- Speech-to-text: 1–3s
- Intent detection: <100ms
- Response generation: <50ms
- Total latency: ~2–5s

Optimization tips:
- Preload cache at startup
- Use compiled regex patterns
- Cache common responses
- Use async processing with FastAPI

---

## 10. Testing

Unit tests:

```bash
pytest test_voice_assistant.py -v
```

Manual tests: use curl examples above and the frontend component.

---

## 11. Deployment

Docker example is provided in the setup doc. For production, use HTTPS and configure reverse proxy (Nginx) to ensure microphone access.

Cloud options include AWS Lambda, Azure App Service, and Google Cloud Run.

---

## 12. Troubleshooting

Common issues and fixes are included (microphone permissions, browser support, cache validity).

---

## 13. Security & Privacy

- Do not log user audio externally in offline mode
- Use HTTPS in production
- Validate inputs and implement rate limiting

---

## 14. Extending the Assistant

Add intents, languages, and knowledge by updating `voice_assistant.py` and `OfflineCacheManager` as documented in the original docs.

---

## 15. Files Added & Implementation Summary

Summary of new files and their purpose is included (core module, router, frontend component, tests, docs).

---

## 16. Quick Reference & Commands

Quick start commands and API examples are included in the combined doc — see sections 4 and 6 above.

---

## 17. Next Steps

1. Install dependencies
2. Start backend
3. Start frontend
4. Test languages
5. Deploy to production

---

## 18. Support & Contribution

Check the repository issues and follow NSoC'26 contribution guidelines. Use this combined file as the single reference.

---

*Created by integrating the four voice-assistant docs into one combined reference.*
