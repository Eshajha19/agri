# Kilo Agent Configuration

## Commands

### Development Server
- `npm run dev` - Start Vite dev server (port 5173)
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint

### Python Backend
- `python main.py` - Start FastAPI server (port 8000)

## Environment

- Frontend: Vite + React 19 + React Router 7
- Backend: FastAPI (Python)
- Database: Firebase
- Deployment: GitHub Codespaces

## Known Issues

### GitHub Codespaces PWA Fix
The Vite dev server on GitHub Codespaces requires special configuration for:
- Manifest file serving (configured via codespaceDevPlugin)
- CORS headers for cross-origin requests
- Relative path references in index.html

The `manifest.webmanifest` file must be in the frontend root directory for both dev and production builds.

## File Conventions

- Frontend source: `/workspaces/agri/frontend/`
- Public assets: `/workspaces/agri/frontend/public/` and `/workspaces/agri/frontend/Public/`
- Python source: `/workspaces/agri/`
- Build output: `/workspaces/agri/frontend/build/`

## Linting

Run `npm run lint` in the frontend directory to check code quality.

## Bug Fixes

### CORS Wildcard Origin with Credentials Violates Spec
- **File**: `main.py`
- **Issue**: `allow_origins=["*"]` combined with `allow_credentials=True` violated the CORS specification — browsers reject credentialed requests with wildcard origins, breaking authenticated cross-origin calls while leaving non-credentialed requests open to any origin.
- **Fix**: Replaced `["*"]` with an explicit allowlist built from localhost dev origins, `FRONTEND_URL` env var (production), and `ADDITIONAL_ALLOWED_ORIGINS` env var (staging/preview). The allowlist is constructed at startup before the middleware is added.
- **Impact**: Credentialed cross-origin requests now work correctly with known origins, and arbitrary origins can no longer make API requests.
