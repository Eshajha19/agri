# 🌱 Fasal Saathi

![NSoC 2026](https://img.shields.io/badge/NSoC-2026-blue)

🚀 **This project is a part of Nexus Spring of Code (NSoC) 2026**

---

## 📘 Nexus Spring of Code 2026

This repository is officially participating in **Nexus Spring of Code 2026 (NSoC'26)**.

We welcome contributors from the NSoC program to collaborate and improve this project.

### 🧑‍💻 For Contributors

* Pick an issue labeled with `level1`, `level2`, or `level3` or raise an issue 
* Ask to be assigned before starting work
* Submit a Pull Request with **`NSoC'26`** in the title
* Follow proper contribution guidelines

---

## 📌 Contribution Rules (NSoC Specific)

* ✅ PR must include **NSoC'26** tag
* ✅ Issue must be assigned before PR
* ❌ PR without assignment will be closed
* ❌ Inactive contributors (7 days) may be unassigned

---

## 🏷️ Issue Labels

* `level1` — Beginner (level 1)
* `level2` — Intermediate (level 2)
* `level3` — Advanced (level 3)

---

## ⚠️ Note

This project follows all rules and guidelines defined under the **Nexus Spring of Code 2026** program.

Any misuse, spam, or low-quality contributions will not be accepted.

---

# 🌾 Fasal Saathi

Fasal Saathi is a smart agriculture assistance platform built with React (frontend), Python (backend) and Firebase (database/auth). The app delivers crop recommendations, weather-based alerts, soil health analysis, and fertilizer guidance to help farmers make informed decisions.

---

## 🚀 Features

- 🌱 Crop recommendation based on soil profile and regional climate
- ☁️ Real-time weather updates and custom farming alerts
- 🧪 Soil health analysis & nutrient suggestions
- 🌾 Fertilizer and pesticide guidance
- 📊 Responsive and user-friendly dashboard (React)
- 🔐 Authentication & user profiles (Firebase)
- 🌐 Multi-language support (planned / optional)

---

## 🛠️ Tech Stack

- Frontend: React.js (Vite)
- Backend: Python (FastAPI)
- Database: Firebase (Firestore / Realtime DB)
- Auth: Firebase Authentication
- External APIs: Weather API (e.g., OpenWeatherMap), Soil/Agro data APIs
- Deployment: Vercel (frontend), Render (backend - in process)

---

## 📁 Project Structure

```tree
agri/
├── frontend/                 # React application (Vite)
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   ├── main.jsx
│   ├── App.jsx
│   ├── components/
│   ├── services/
│   ├── stores/
│   ├── utils/
│   ├── hooks/
│   ├── lib/
│   ├── themes/
│   ├── weather/
│   └── public/
├── main.py                   # FastAPI backend entry
├── requirements.txt          # Python dependencies
├── ml/                       # Machine learning models
├── rag/                      # RAG advisor components
├── firebase/
│   └── firestore.rules
├── .env.example
├── README.md
└── LICENSE
```

---

## ⚙️ Installation & Local Setup

> Requirements: Node.js (v16+), npm/yarn, Python 3.9+, pip, Firebase CLI (optional).

### 1. Clone Repository

```bash
git clone https://github.com/KGFCH2/agri.git
cd agri
```

### 2. Frontend (React + Vite)

```bash
cd frontend
```

#### Install Dependencies

```bash
npm install
```

#### Start Dev Server

```bash
npm run dev
```

#### Build for Production

```bash
npm run build
```

#### Preview Production Build

```bash
npm run preview
```

### 3. Backend (Python — FastAPI)

```bash
cd ..
```

#### Create Virtual Environment (Optional)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Run FastAPI Server

```bash
python -m uvicorn main:app --reload --port 8000
```

### 4. Firebase Setup

1. Create a Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Enable Firestore (or Realtime DB) and Firebase Auth (Email/Phone)
3. Add Firebase config to frontend `.env` file (see `.env.example`)
4. (Optional) Deploy security rules found in `firebase/`

---

## 🔐 Environment Variables (.env.example)

### Backend

```env
WEATHER_API_KEY=your_weather_api_key
SOIL_API_KEY=your_soil_api_key
FIREBASE_ADMIN_CRED=/path/to/serviceAccountKey.json
BACKEND_PORT=5000
```

### Frontend

```env
REACT_APP_FIREBASE_API_KEY=xxxxxxxxxxxx
REACT_APP_FIREBASE_AUTH_DOMAIN=your-app.firebaseapp.com
REACT_APP_FIREBASE_PROJECT_ID=your-app
REACT_APP_BACKEND_URL=http://localhost:5000
```

---

## 🧩 API Endpoints (Examples)

### Backend (FastAPI)

- `GET /api/weather?lat={lat}&lon={lon}` — Returns current weather + forecast
- `POST /api/soil/analyze` — Send soil params (pH, NPK) to get recommendations
- `POST /api/crop/recommend` — Returns recommended crops for given soil & climate

(Document exact request/response schemas in docs/ or OpenAPI spec.)

---

## 🧪 Testing

- Frontend: use Vitest / React Testing Library
- Backend: pytest / unittest
- Add CI with GitHub Actions for linting + tests + deploy

---

## 🎯 Objective

Provide farmers with a lightweight, region-aware digital assistant that reduces risk, improves yields, and encourages sustainable decisions through actionable insights.

---

## 🔮 Future Scope & Ideas

- On-device offline support / PWA for low-connectivity regions
- Integrate satellite / remote sensing for crop stress detection
- SMS / WhatsApp alerts for farmers without smartphones
- Integrate local market price data for crop sale recommendations
- Train ML models using local farm historical data for precision recommendations
