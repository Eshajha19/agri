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
### TOCTOU Race Condition in Queue Overflow (PR #996)
- **File**: `image_processing_queue.py`
- **Issue**: Queue capacity validation (`len(self._task_queue) >= self.max_queue_size`) was performed outside `_queue_lock`, allowing concurrent producers to bypass the capacity limit.
- **Fix**: Moved capacity check inside the synchronized `with self._queue_lock:` block, ensuring size validation and task insertion are atomic under concurrent workloads.
- **Impact**: Prevents uncontrolled queue growth, excessive memory consumption, and inconsistent queue state under high-throughput image processing.

### Admin/Expert Role Check Broken by Wrong Key Name (PR #1022)
- **Files**: `main.py`, `rbac.py`
- **Issue**: Consultation override logic used `token_data.get("role")` but `verify_role()` returns `{"roles": [...]}` (list under `"roles"` key). `"role"` always returned `None`, so `None not in ("admin", "expert")` was always `True`, making admins/experts unable to override consultation ownership.
- **Fix**: Migrated from Firestore `get()` role checks to Firebase custom claims (`bba24af`). The `roles` list is now correctly queried instead of the non-existent `role` key.
- **Impact**: Admins and experts can now properly override consultation ownership as intended.

### ML Pipeline Catch-All Exception Silences Startup Failures
- **File**: `main.py`
- **Issue**: `init_ml_pipeline()` wrapped its entire body in `try/except Exception`, catching corrupted model files, import errors, and other fatal conditions — then silently logged them as warnings. The server started successfully with a broken ML pipeline, causing hard-to-diagnose prediction failures at runtime.
- **Fix**: Removed the `try/except` block. The `os.path.exists()` guard handles the missing-model case gracefully (logs a warning). All other failures (corrupt file, XGBoost import error, adapter init failure) now propagate up through the lifespan, causing fail-fast on startup.
- **Impact**: ML pipeline initialization errors now crash the worker at startup instead of producing silent failures. Missing model file is still handled gracefully.
