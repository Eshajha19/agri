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
