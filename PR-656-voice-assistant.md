Title: feat(voice): Add Multilingual Offline Voice Assistant for Farmers (#656)

Related issue: #656

Summary
-------
This PR adds a multilingual, offline-first Voice Assistant feature for the Fasal Saathi platform. The assistant enables farmers to interact using voice (speech-to-text + text-to-speech), supports eight regional languages, includes rule-based offline intent detection, session management, and a React frontend component.

Key changes
-----------
- Add core backend module: `voice_assistant.py` (intent detection, session, offline cache manager)
- Add FastAPI router: `backend/routers/voice_assistant.py` with endpoints under `/api/voice`
- Register voice assistant router and initialize assistant in `main.py`
- Add React component and styles: `frontend/VoiceAssistant.jsx`, `frontend/VoiceAssistant.css`
- Add tests: `test_voice_assistant.py` and examples `voice_assistant_examples.py`
- Add documentation: [VOICE_ASSISTANT_COMBINED.md](VOICE_ASSISTANT_COMBINED.md) (combined docs)
- Update `requirements.txt` with optional/offline-friendly voice deps (pyttsx3, SpeechRecognition, vosk, sounddevice, librosa, soundfile, scipy) and cloud fallbacks
- Wire frontend route and navigation link in `frontend/App.jsx`

Why this change
----------------
Farmers need low-friction, accessible ways to use the platform. Voice interaction in regional languages with offline capability reduces friction in low-connectivity areas and supports low-literacy users.

Files of interest
-----------------
- [voice_assistant.py](voice_assistant.py)
- [backend/routers/voice_assistant.py](backend/routers/voice_assistant.py)
- [main.py](main.py) (initialization + router include)
- [frontend/VoiceAssistant.jsx](frontend/VoiceAssistant.jsx)
- [frontend/VoiceAssistant.css](frontend/VoiceAssistant.css)
- [frontend/App.jsx](frontend/App.jsx#L1)
- [test_voice_assistant.py](test_voice_assistant.py)
- [VOICE_ASSISTANT_COMBINED.md](VOICE_ASSISTANT_COMBINED.md)

Testing performed
-----------------
- Static verification of new files and unit tests are included. Please run locally:

```bash
# Install Python deps
cd agri
pip install -r requirements.txt

# Run backend tests
pytest test_voice_assistant.py -q

# Start backend
python main.py

# Start frontend
cd frontend
npm run dev
```

Notes
------------
- STT/TTS implementations include safe placeholders and instructions to enable `vosk` and `pyttsx3` for offline usage; see `VOICE_ASSISTANT_COMBINED.md` for setup.
- Frontend route was added to `frontend/App.jsx` and the link placed in "More" dropdown. You can add additional links (footer/banner) if desired.
- This PR does not remove the original docs; it adds `VOICE_ASSISTANT_COMBINED.md` as the consolidated reference.

How to submit the PR locally
---------------------------
Run these commands locally to create a branch, commit, push and open a PR (using `gh` or browser):

```bash
# from repo root
git checkout -b feat/voice-assistant-656
git add .
git commit -m "feat(voice): Add multilingual offline voice assistant (fixes #656)"
git push --set-upstream origin feat/voice-assistant-656

# open PR with GitHub CLI (recommended)
gh pr create --title "feat(voice): Add Multilingual Offline Voice Assistant for Farmers (#656)" --body-file PR-656-voice-assistant.md --base main --head feat/voice-assistant-656

# OR: open PR in browser
# Copy the URL below (replace USER/REPO)
# https://github.com/USER/REPO/compare/main...feat/voice-assistant-656?expand=1
```

If you'd like, I can:
- Create a `.github/pull_request_template.md` for this repo
- Open the PR for you if you give me `gh` CLI access and authentication (or run the commands here if your environment allows)

Reviewers & Labels
------------------
- Suggested reviewers: @maintainers, @ml-team, @frontend
- Suggested labels: enhancement, feature, accessibility, offline, docs

Changelog entry
---------------
- feat: add offline multilingual voice assistant with React UI and backend API (issue #656)

---

(Prepared PR body file: `PR-656-voice-assistant.md`)
