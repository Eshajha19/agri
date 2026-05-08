# Fasal Saathi

![NSoC 2026](https://img.shields.io/badge/NSoC-2026-blue)

**This project is a part of Nexus Spring of Code (NSoC) 2026**

---

## Nexus Spring of Code 2026

This repository is officially participating in **Nexus Spring of Code 2026 (NSoC'26)**.

We welcome contributors from the NSoC program to collaborate and improve this project.

### For Contributors

* Pick an issue labeled with `level1`, `level2`, or `level3` or raise an issue 
* Ask to be assigned before starting work
* Submit a Pull Request with **`NSoC'26`** in the title
* Follow proper contribution guidelines

---

## Contribution Rules (NSoC Specific)

* PR must include **NSoC'26** tag
* Issue must be assigned before PR
* PR without assignment will be closed
* Inactive contributors (7 days) may be unassigned

---

## Issue Labels

* `level1` — Beginner (level 1)
* `level2` — Intermediate (level 2)
* `level3` — Advanced (level 3)

---

## Note

This project follows all rules and guidelines defined under the **Nexus Spring of Code 2026** program.

Any misuse, spam, or low-quality contributions will not be accepted.

---

# Fasal Saathi

Fasal Saathi is a smart agriculture assistance platform built with React (frontend), Python (backend) and Firebase (database/auth). The app delivers crop recommendations, weather-based alerts, soil health analysis, and fertilizer guidance to help farmers make informed decisions.

---

## Features

- Crop recommendation based on soil profile and regional climate
- Real-time weather updates and custom farming alerts
- Soil health analysis & nutrient suggestions
- Fertilizer and pesticide guidance
- Responsive and user-friendly dashboard (React)
- Authentication & user profiles (Firebase)
- Multi-language support

---

## Tech Stack

- Frontend: React.js (Vite)
- Backend: Python (FastAPI)
- Database: Firebase (Firestore / Realtime DB)
- Auth: Firebase Authentication
- External APIs: OpenWeatherMap, Google Gemini API
- Deployment: Vercel (frontend), Render (backend)

---

## Project structure
```tree
agri/
├── main.py              # FastAPI backend entry
├── alert_rules.py       # Logic for generating alerts
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
├── frontend/            # React application
│   ├── App.jsx          # Root component & routing
│   ├── Dashboard.jsx    # Main user dashboard
│   ├── Advisor.jsx      # AI Advisor feature
│   ├── index.html       # Entry point
│   ├── package.json     # Node.js dependencies
│   └── vite.config.js   # Vite configuration
├── firebase/            # Firebase rules/config
│   └── firestore.rules
└── README.md
```

---

## Installation & Local setup

### Prerequisites
- **Node.js**: v18.0 or higher
- **Python**: v3.10 or higher
- **Firebase**: Project credentials

### 1. Clone the repository
```bash
git clone https://github.com/KGFCH2/agri.git
cd agri
```

### 2. Backend Setup (FastAPI)
```bash
# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Run the backend server
python main.py
```
*API: `http://localhost:8000`*

### 3. Frontend Setup (Vite + React)
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```
*App: `http://localhost:5173`*

### 4. Environment Configuration
Create a `.env` file in the root directory:
```env
# Backend Keys
WEATHER_API_KEY=your_openweather_key
GEMINI_API_KEY=your_google_gemini_key

# Frontend Keys (Vite prefixed)
VITE_FIREBASE_API_KEY=xxx
VITE_FIREBASE_AUTH_DOMAIN=xxx
VITE_FIREBASE_PROJECT_ID=xxx
```

## API Endpoints

- `GET /api/weather` — Weather updates + forecast
- `POST /api/soil/analyze` — Soil nutrient analysis
- `POST /api/crop/recommend` — Crop recommendations

---

## Testing

- Frontend: `npm test`
- Backend: `pytest`

---

## Objective

Provide farmers with a lightweight, region-aware digital assistant that reduces risk, improves yields, and encourages sustainable decisions through actionable insights.

---

## Future Scope

- On-device offline support / PWA for low-connectivity regions
- Satellite remote sensing for crop stress detection
- SMS / WhatsApp alerts for farmers without smartphones
- Local market price data integration
- ML models trained on local historical data
