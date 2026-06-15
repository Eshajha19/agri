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
- 🪴 AI-based crop disease detection from uploaded images
- 🌾 Fertilizer and pesticide guidance
- 🧪 A/B testing runner with traffic split, metrics pipeline, and auto-promotion
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
├── frontend/                 # React frontend application
│   ├── components/
│   ├── services/
│   ├── hooks/
│   ├── utils/
│   ├── stores/
│   ├── locales/
│   ├── weather/
│   └── public/
│
├── backend/                  # Backend APIs and routers
│   ├── routers/
│   └── schemas/
│
├── ml/                       # Machine learning pipelines and models
│   ├── adapters/
│   └── governance/
│
├── rag/                      # Retrieval-Augmented Generation modules
│
├── tests/                    # Backend test suite
├── scripts/                  # Automation and utility scripts
├── docs/                     # Project documentation
├── persistence/              # Database and migration logic
├── feature_flags/            # Feature flag and A/B testing system
├── routers/                  # Additional ML routers
├── inference/                # ONNX inference runtime
├── benchmarks/               # Benchmarking scripts
├── configs/                  # Configuration files
├── runs/                     # ML run manifests
├── runs_test/                # Test run manifests
├── .github/                  # GitHub workflows and templates
│
├── main.py
├── requirements.txt
├── package.json
├── README.md
└── CONTRIBUTING.md
```

---


### 1. Clone Repository

```bash
git clone https://github.com/Eshajha19/agri.git

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
VITE_API_BASE_URL=https://your-backend.onrender.com
# Alternative: VITE_BACKEND_URL is also supported as a fallback

For Vercel deployments, set `VITE_API_BASE_URL` to the live backend origin in the project environment variables. Without it, browser requests stay on the static frontend host and marketplace API calls will fail.

For certified/bank report generation, the backend also needs a signing key source. In production, configure either Google Cloud Secret Manager (`GOOGLE_CLOUD_PROJECT` + `REPORT_SIGNING_SECRET_NAME`) or `REPORT_SIGNING_PRIVATE_KEY_PEM` with a PEM-encoded Ed25519 private key.

---

## 🧩 API Endpoints (Examples)

### Backend (FastAPI)

- `GET /api/weather?lat={lat}&lon={lon}` — Returns current weather + forecast
- `POST /api/soil/analyze` — Send soil params (pH, NPK) to get recommendations
- `POST /api/crop/recommend` — Returns recommended crops for given soil & climate
- `POST /api/crop-disease/analyze-image` — Analyze an uploaded crop image and return the likely disease, confidence, and treatment guidance
- `POST /api/experiments/{exp_id}/traffic-split` — Update experiment traffic split for A/B testing
- `POST /api/experiments/{exp_id}/evaluate` — Evaluate experiment metrics and auto-promote a winner when the lift is clear
- `POST /api/experiments/assign` — Assign a user to a variant and emit an impression event

(Document exact request/response schemas in docs/ or OpenAPI spec.)

---

## 🚀 Render Deployment Guide

This project can be deployed easily on [Render](https://render.com) for production hosting.

### 1️⃣ Create a Render Service
- Log in to [Render Dashboard](https://render.com).
- Click **New + → Web Service**.
- Connect your GitHub repository.
- Select the branch you want to deploy (usually `main`).
- Choose environment: **Python 3.11** (or your project’s version).

---

### 2️⃣ Build Command
Render will run this to install dependencies and build the frontend:

```bash
pip install -r requirements.txt
npm install --prefix frontend && npm run build --prefix frontend

---

### 3️⃣ Start Command
Render will run this to start the backend service:

bash
uvicorn main:app --host 0.0.0.0 --port $PORT

### 4️⃣ Required Environment Variables
Set these in Render → Settings → Environment Variables:

Backend
env
WEATHER_API_KEY=your_weather_api_key
SOIL_API_KEY=your_soil_api_key
FIREBASE_ADMIN_CRED=/path/to/serviceAccountKey.json
BACKEND_PORT=5000

Frontend
env
VITE_FIREBASE_API_KEY=xxxxxxxxxxxx
VITE_FIREBASE_AUTH_DOMAIN=your-app.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your-app
VITE_FIREBASE_STORAGE_BUCKET=your-app.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=123456789
VITE_FIREBASE_APP_ID=1:123456789:web:abcdef
VITE_BACKEND_URL=https://your-render-service.onrender.com

## 🧪 Testing

- Frontend: use Vitest / React Testing Library
- Backend: pytest / unittest
- Add CI with GitHub Actions for linting + tests + deploy

## 🔒 Security CI

The repository now includes a dedicated security gate for secret scanning, dependency SCA, and policy enforcement.

Local policy check:

```bash
python scripts/security_ci.py policy --root . --policy .github/security-policy.json
```

The GitHub Actions workflow at [`.github/workflows/security-ci.yml`](.github/workflows/security-ci.yml) runs:

- secret scanning and repository policy enforcement via `scripts/security_ci.py`
- dependency SCA via `pip-audit` and `safety`

Test fixtures are excluded from the secret scan so synthetic examples in the test suite do not fail the gate.

## 🔒 Differential Privacy (DP-SGD) Proof of Concept

Yield model training now supports an optional differential privacy mode.

### What is included

- `training_mode=dp_sgd` in `train_model.py` using optional `torch + opacus`
- Configurable privacy targets via `epsilon` and `delta`
- Privacy accounting logs during training (target epsilon, spent epsilon, noise multiplier)
- Manifest/registry metadata now include privacy fields when DP mode is used
- A reproducible comparison script to evaluate baseline vs DP utility

### DP config example

```json
{
	"dataset": "Train.csv",
	"seed": 42,
	"training_mode": "dp_sgd",
	"epsilon": 3.0,
	"delta": 0.00001,
	"dp_epochs": 8,
	"dp_batch_size": 64,
	"dp_learning_rate": 0.05,
	"dp_max_grad_norm": 1.0,
	"output_model": "yield_model_dp.pt"
}
```

### Baseline config example

```json
{
	"dataset": "Train.csv",
	"seed": 42,
	"training_mode": "baseline",
	"output_model": "yield_model.joblib"
}
```

### Run reproducible utility comparison

```bash
python scripts/compare_dp_utility.py --dataset Train.csv --epsilon 3.0 --delta 1e-5 --seed 42 --output dp_utility_comparison.json
```

### Optional dependencies for DP mode

DP mode is optional and requires extra packages:

```bash
pip install torch opacus
```

## 🔁 ONNX conversion, GPU inference path, and benchmarking

We provide utilities to convert models to ONNX, run inference preferring GPU (if available) with CPU fallback, and run inference benchmarks.

Conversion script:

```bash
python scripts/convert_model_to_onnx.py --model path/to/model.joblib --n-features 39 --out model.onnx
```

Run ONNX inference benchmark (example):

```bash
python benchmarks/benchmark_inference.py --model model.onnx --input-shape 1,39 --iterations 200 --warmup 20 --output bench.json
```

The inference wrapper `inference/onnx_runtime.py` selects `CUDAExecutionProvider` when available, otherwise falls back to `CPUExecutionProvider`.

## 🧪 A/B Testing Runner

The feature-flag A/B testing stack now includes a runner that handles deterministic traffic splits, metric ingestion, and automatic winner promotion.

### What it does

- Assigns users to variants using the configured traffic split.
- Logs impression and conversion events into the experiment metrics pipeline.
- Evaluates conversion-rate lift and promotes the winning variant automatically when the threshold is met.
- After promotion, the winner receives 100% traffic and future assignments route to the promoted variant.

### Key endpoints

```bash
POST /api/experiments/{exp_id}/traffic-split
POST /api/experiments/{exp_id}/evaluate
POST /api/experiments/assign
```

### Example traffic split payload

```json
{
	"variants": [
		{"id": "control", "weight": 40},
		{"id": "treatment", "weight": 60}
	]
}
```

---

## 🚀 Next Step
1. Edit `.gitignore` → add `*.joblib` and `*.pkl`.  
2. Update `README.md` with instructions to regenerate models.  
3. Commit and push:

```bash
git checkout -b chore/gitignore-models-2983
git add .gitignore README.md
git commit -m "chore: ignore .joblib and .pkl model files, add README instructions (#2983)"
git push origin chore/gitignore-models-2983


### Tradeoffs

- Better privacy guarantees usually require stronger noise (lower utility).
- Lower epsilon means stronger privacy but can increase RMSE.
- DP training is typically slower than baseline training.
- This implementation is a research proof-of-concept and should be calibrated before production use.

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

## 🚨 New Feature: Farming Mistakes Awareness System

This in-app guide highlights common farming mistakes and practical steps to avoid them. Examples include:

- Over-fertilization — how to test soil and dose correctly.
- Wrong irrigation timing — when and how to irrigate for best results.
- Poor seed selection — choosing certified, climate-appropriate varieties.

How to access: Open the Advisor page and choose the "Farming Mistakes Awareness" card to open the modal with examples, images, and prevention tips.

Acceptance criteria:

- Common mistakes are listed with an explanation of the problem.
- Each mistake includes clear, actionable prevention steps.
- Image examples for visual recognition.
- Responsive UI and no console errors in Advisor view.

## 🖼️ New Feature: Crop Growth Stage Visual Guide

A responsive in-app visual guide that walks farmers through the crop lifecycle: Seed → Sprout → Growth → Harvest. The guide includes stage-wise care instructions, image-based examples for visual learning, and a lightweight lightbox for inspecting images.

How to access: Open the app and go to the Advisor page — the "Crop Growth Stage Visual Guide" card opens the modal with the visual walkthrough and learning images.

Acceptance criteria:

- Seed → Sprout → Growth → Harvest stages represented visually.
- Stage-wise care instructions are shown for each stage.
- Image-based learning gallery with thumbnails and enlargements.
- Responsive UI and no console errors when used in the Advisor view.

## 🌦️ New Feature: Seasonal Farming Strategy Guide

A responsive in-app strategy guide that explains how farming priorities change across the Indian crop seasons: Kharif, Rabi, and Zaid. The guide highlights season-specific crop focus, irrigation posture, key field priorities, and common risks so farmers can adapt their plan throughout the year.

How to access: Open the Advisor page and choose the "Seasonal Farming Strategy Guide" card to open the modal with the season-by-season playbook.

Acceptance criteria:

- Kharif, Rabi, and Zaid strategies are shown clearly in one guided view.
- Each season includes field priorities, irrigation guidance, and risks to watch.
- The UI stays responsive across desktop and mobile layouts.
- The guide opens cleanly in the Advisor view with no console errors.

Alternatives considered:

- Linking externally to a knowledge article or PDF (rejected — offline and discoverability concerns).
- A full LMS course module with video lessons (more content-heavy; deferred to Agri-LMS integration).
