import os
import re

from starlette.middleware.base import BaseHTTPMiddleware


def _is_vercel_origin(origin: str) -> bool:
    """
    Check if the origin is a Vercel-deployed frontend.

    Vercel URLs follow these patterns:
    - Production: https://<project>.vercel.app
    - Preview (branched): https://<branch>--<project>.vercel.app or https://<project>--<hash>-<scope>.vercel.app
    - Team preview: https://<scope>-<team>.vercel.app

    We allow all vercel.app origins to support both production and preview deployments.
    """
    if not origin:
        return False
    origin_lower = origin.lower()
    return bool(re.match(r"^https://[a-z0-9-]+\.vercel\.app$", origin_lower))


class VercelCORSMiddleware(BaseHTTPMiddleware):
    """
    CORS middleware that supports both static allowlist and dynamic Vercel preview origins.

    This middleware extends Starlette's BaseHTTPMiddleware to dynamically allow
    Vercel preview URLs while maintaining security for other origins.
    """

    def __init__(self, app, static_origins: list = None):
        super().__init__(app)
        self.static_origins = set(static_origins or [])
        self.allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        self.allow_headers = ["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"]

    def _is_allowed_origin(self, origin: str) -> bool:
        """Check if the origin is allowed."""
        if not origin:
            return False
        if origin in self.static_origins:
            return True
        if _is_vercel_origin(origin):
            return True
        return False

    async def dispatch(self, request, call_next):
        origin = request.headers.get("origin")

        response = await call_next(request)

        if origin and self._is_allowed_origin(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"

            if request.method == "OPTIONS":
                response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
                response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
                response.headers["Access-Control-Max-Age"] = "3600"

        return response


def setup_cors(app):
    cors_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "https://fasal-saathi.vercel.app",
        "https://fasal-saathi.xyz",
    ]

    frontend_url = os.getenv("FRONTEND_URL", "").strip()
    if frontend_url and frontend_url not in cors_origins:
        cors_origins.append(frontend_url)

    extra_origins = os.getenv(
        "ADDITIONAL_ALLOWED_ORIGINS",
        "",
    ).strip()

    if extra_origins:
        for origin in extra_origins.split(","):
            origin = origin.strip()
            if origin and origin not in cors_origins:
                cors_origins.append(origin)

    app.add_middleware(VercelCORSMiddleware, static_origins=cors_origins)