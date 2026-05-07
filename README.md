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

- Frontend: React.js (Create React App / Vite)
- Backend: Python ( FastAPI)
- Database: Firebase (Firestore / Realtime DB)
- Auth: Firebase Authentication
- External APIs: Weather API (e.g., OpenWeatherMap), Soil/Agro data APIs
- Deployment: Vercel  (frontend), Render (backend- in process)

---

## 📁 Project structure
<<<<<<< HEAD

=======
```tree
>>>>>>> upstream/main
Fasal-Saathi/
├── backend/
│ ├── app.py # Flask/FastAPI app entry
│ ├── requirements.txt
│ ├── services/
│ │ ├── weather.py # Weather API wrapper
│ │ └── soil.py # Soil analysis logic / wrapper
│ └── utils/
│ └── ml_models.py # Optional: model for crop recommendation
│
├── frontend/ # React application
│ ├── package.json
│ ├── public/
│ │ └── index.html
│ └── src/
│ ├── App.jsx
│ ├── index.jsx
│ ├── api/
│ │ └── apiClient.js # communicates with backend
│ ├── components/
│ │ ├── Dashboard/
│ │ ├── CropRecommender/
│ │ └── Auth/
│ ├── pages/
│ └── assets/
│
├── firebase/ # firebase rules / config (optional)
│ └── firestore.rules
│
├── scripts/
│ └── deploy.sh
│
├── .env.example
├── README.md
└── LICENSE
<<<<<<< HEAD

=======
```
>>>>>>> upstream/main

---

## ⚙️ Installation & Local setup

> Requirements: Node.js (v16+), npm/yarn, Python 3.9+, pip, Firebase CLI (optional).

<<<<<<< HEAD
### 1) Clone repository
```bash
git clone https://github.com/your-username/fasal-saathi.git
cd fasal-saathi

2) Frontend (React)
cd frontend
# Install dependencies
npm install
# Start dev server
npm start
# Build for production
npm run build

3) Backend (Python — Flask example)
cd ../backend
# Create virtual env (optional)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Run app
export FLASK_APP=app.py
export FLASK_ENV=development
# Set environment variables (see .env.example)
flask run


# 4) Firebase

Create a Firebase project.

Enable Firestore (or Realtime DB) and Firebase Auth (Email/Phone).

Add Firebase config to frontend env (see .env.example).

(Optional) Deploy security rules found in firebase/.

🔐 Environment variables (.env.example)
# Backend
=======
## Clone repository

```bash
git clone https://github.com/your-username/fasal-saathi.git
cd fasal-saathi
```

## Frontend (React)

```bash
cd frontend
```

### 1. Install dependencies

```bash
npm install
```

### 2. Start dev server

```bash
npm start
```

### 3. Build for production

```bash
npm run build
```

## Backend (Python — Flask example)

```bash
cd ../backend
```

### 1. Create virtual env (optional)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run app

```bash
export FLASK_APP=app.py
export FLASK_ENV=development
```

### 3. Set environment variables (see .env.example)

```bash
flask run
```


## Firebase

- Create a Firebase project.

- Enable Firestore (or Realtime DB) and Firebase Auth (Email/Phone).

- Add Firebase config to frontend env (see .env.example).

- (Optional) Deploy security rules found in firebase/.

🔐 Environment variables (.env.example)

## Backend

```
>>>>>>> upstream/main
WEATHER_API_KEY=your_weather_api_key
SOIL_API_KEY=your_soil_api_key
FIREBASE_ADMIN_CRED=/path/to/serviceAccountKey.json
BACKEND_PORT=5000
<<<<<<< HEAD

# Frontend
=======
```

## Frontend

```
>>>>>>> upstream/main
REACT_APP_FIREBASE_API_KEY=xxxxxxxxxxxx
REACT_APP_FIREBASE_AUTH_DOMAIN=your-app.firebaseapp.com
REACT_APP_FIREBASE_PROJECT_ID=your-app
REACT_APP_BACKEND_URL=http://localhost:5000
<<<<<<< HEAD

🧩 API Endpoints (examples)

Backend (Flask)

GET /api/weather?lat={lat}&lon={lon} — returns current weather + forecast

POST /api/soil/analyze — send soil params (pH, NPK) to get recommendations

POST /api/crop/recommend — returns recommended crops for given soil & climate

(Document exact request/response schemas in docs/ or OpenAPI spec.)

🧪 Testing

Frontend: use React Testing Library / Jest

Backend: pytest / unittest

Add CI with GitHub Actions for linting + tests + deploy

🎯 Objective

Provide farmers with a lightweight, region-aware digital assistant that reduces risk, improves yields, and encourages sustainable decisions through actionable insights.

🔮 Future scope & ideas

On-device offline support / PWA for low-connectivity regions

Integrate satellite / remote sensing for crop stress detection

SMS / WhatsApp alerts for farmers without smartphones

Integrate local market price data for crop sale recommendations

Train ML models using local farm historical data for precision recommendations
=======
```

# ☆ API Endpoints (examples)

Backend (Flask)

- GET /api/weather?lat={lat}&lon={lon} — returns current weather + forecast

- POST /api/soil/analyze — send soil params (pH, NPK) to get recommendations

- POST /api/crop/recommend — returns recommended crops for given soil & climate

(Document exact request/response schemas in docs/ or OpenAPI spec.)

# ☆ Testing

- Frontend: use React Testing Library / Jest

- Backend: pytest / unittest

- Add CI with GitHub Actions for linting + tests + deploy

# ☆ Objective

Provide farmers with a lightweight, region-aware digital assistant that reduces risk, improves yields, and encourages sustainable decisions through actionable insights.

# ☆ Future scope & ideas

- On-device offline support / PWA for low-connectivity regions

- Integrate satellite / remote sensing for crop stress detection

- SMS / WhatsApp alerts for farmers without smartphones

- Integrate local market price data for crop sale recommendations

- Train ML models using local farm historical data for precision recommendations
>>>>>>> upstream/main
