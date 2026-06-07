import os

from fastapi.middleware.cors import CORSMiddleware

import csrf_protection as _csrf


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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=[
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "OPTIONS",
            "PATCH",
        ],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Accept",
            "Origin",
            "X-Requested-With",
        ],
    )

    _csrf.configure(cors_origins)