# Render Deployment Guide for Fasal Saathi

## Architecture Overview

This deployment creates 4 services on Render:
1. **agri-backend** - FastAPI application (Docker)
2. **agri-celery-worker** - Background task worker (Docker)
3. **agri-frontend** - React static site
4. **agri-redis** - Redis key-value store for Celery

## Required Environment Variables

### Backend (agri-backend)
| Variable | Description | Required |
|----------|-------------|----------|
| GEMINI_API_KEY | Google Gemini API key | Yes |
| GOOGLE_CLOUD_PROJECT | GCP project ID | Yes (for Firebase) |
| REPORT_SIGNING_SECRET_NAME | KMS secret name | Optional |
| TWILIO_ACCOUNT_SID | Twilio account SID | Optional |
| TWILIO_AUTH_TOKEN | Twilio auth token | Optional |
| TWILIO_WHATSAPP_NUMBER | Twilio WhatsApp number | Optional |

### Frontend (agri-frontend)
| Variable | Description | Required |
|----------|-------------|----------|
| VITE_FIREBASE_API_KEY | Firebase API key | Yes |
| VITE_FIREBASE_AUTH_DOMAIN | Firebase auth domain | Yes |
| VITE_FIREBASE_PROJECT_ID | Firebase project ID | Yes |
| VITE_FIREBASE_STORAGE_BUCKET | Firebase storage bucket | Yes |
| VITE_FIREBASE_MESSAGING_SENDER_ID | Firebase sender ID | Yes |
| VITE_FIREBASE_APP_ID | Firebase app ID | Yes |

## Deployment Steps

1. Push this repository to GitHub
2. Sign in to [Render Dashboard](https://dashboard.render.com)
3. Click "New" → "Blueprint"
4. Connect your GitHub repository
5. Render will auto-detect `render.yaml`
6. Set the secret environment variables when prompted
7. Click "Deploy"

## Important Notes

### Model Files
The `yield_model.joblib` file is included in the repository and will be deployed with the Docker image.

### Firebase Configuration
Ensure Firebase Authentication and Firestore are enabled in your Firebase project. The backend uses Firebase Admin SDK for authentication.

### CORS Configuration
The `FRONTEND_URL` is automatically set from the frontend service URL. The backend's CORS allows this origin plus localhost for development.

### Redis Connection
The `REDIS_URL` is automatically configured from the Redis service. Both backend and Celery worker use this for task queue.

## Local Development

### Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
# or
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Celery Worker (optional for local)
```bash
# Requires Redis running locally
celery -A celery_worker.celery_app worker --loglevel=info
```