"""
Centralized Content Security Policy configuration.

Policies are generated per-environment.  Firebase Authentication and
Google Sign-In domains are included in both environments so OAuth
popups, redirects, and API calls are never blocked.  Google Translate
script and frame sources are centralized here for the SPA and API.
"""

import os

# Domains Firebase Auth uses for OAuth popups, hidden iframes,
# and redirect flows.
FIREBASE_AUTH_FRAME_DOMAINS = [
    "https://*.firebaseapp.com",
    "https://*.web.app",
    "https://accounts.google.com",
    "https://apis.google.com",
]

# Endpoints Firebase Auth connects to via fetch/XHR.
FIREBASE_AUTH_CONNECT_DOMAINS = [
    "https://identitytoolkit.googleapis.com",
    "https://securetoken.googleapis.com",
    "https://www.googleapis.com",
    "https://*.googleapis.com",
]

# Google Translate widget loads scripts from translate.google.com and
# translate.googleapis.com (*.google.com does not cover googleapis.com).
GOOGLE_TRANSLATE_SCRIPT_DOMAINS = [
    "https://translate.google.com",
    "https://translate.googleapis.com",
    "https://translate-pa.googleapis.com",
    "https://*.googleapis.com",
    "https://*.google.com",
]

GOOGLE_TRANSLATE_FRAME_DOMAINS = [
    "https://translate.google.com",
]


def is_production() -> bool:
    return os.getenv("ENV", "").strip().lower() in ("production", "prod")


def _directives(env: str) -> dict:
    translate_scripts = " ".join(GOOGLE_TRANSLATE_SCRIPT_DOMAINS)
    frame_src = (
        "'self' "
        + " ".join(FIREBASE_AUTH_FRAME_DOMAINS + GOOGLE_TRANSLATE_FRAME_DOMAINS)
    )
    connect_src = "'self' " + " ".join(FIREBASE_AUTH_CONNECT_DOMAINS)

    policies = {
        "production": {
            "default-src": "'self'",
            "script-src": f"'self' https://apis.google.com {translate_scripts}",
            "style-src": "'self' 'unsafe-inline'",
            "img-src": "'self' data: https:",
            "font-src": "'self' https://fonts.gstatic.com",
            "connect-src": connect_src,
            "frame-src": frame_src,
            "frame-ancestors": "'none'",
            "object-src": "'none'",
            "base-uri": "'self'",
            "form-action": "'self'",
        },
        "development": {
            "default-src": "'self'",
            "script-src": (
                f"'self' 'unsafe-inline' 'unsafe-eval' https://apis.google.com "
                f"{translate_scripts}"
            ),
            "style-src": "'self' 'unsafe-inline'",
            "img-src": "'self' data: https:",
            "font-src": "'self' https://fonts.gstatic.com",
            "connect-src": "ws: wss: " + connect_src,
            "frame-src": frame_src,
            "frame-ancestors": "'self'",
            "object-src": "'none'",
            "base-uri": "'self'",
            "form-action": "'self'",
        },
    }
    return policies.get(env, policies["development"])


def build_csp_policy() -> str:
    env = "production" if is_production() else "development"
    dirs = _directives(env)
    return "; ".join(f"{k} {v}" for k, v in dirs.items())


def build_spa_csp_policy() -> str:
    """CSP for the SPA (frontend index.html meta tag and API middleware)."""
    env = "production" if is_production() else "development"
    dirs = _directives(env)
    return "; ".join(f"{k} {v}" for k, v in dirs.items())


REQUIRED_DIRECTIVES = {
    "default-src", "script-src", "style-src", "img-src",
    "font-src", "connect-src", "frame-src", "frame-ancestors",
    "object-src", "base-uri", "form-action",
}

RESTRICTIVE_VALUES = {
    "frame-ancestors": ("'none'", "'self'"),
    "object-src": ("'none'",),
}


def validate_csp_policy(policy: str) -> list:
    issues = []
    parts = [p.strip() for p in policy.split(";") if p.strip()]
    directives_found = set()

    for part in parts:
        tokens = part.split(None, 1)
        if not tokens:
            continue
        name = tokens[0]
        directives_found.add(name)
        value = tokens[1] if len(tokens) > 1 else ""
        if name in RESTRICTIVE_VALUES:
            allowed = RESTRICTIVE_VALUES[name]
            if not any(a in value for a in allowed):
                issues.append(
                    f"{name} should contain one of {allowed}, got: {value!r}"
                )

    missing = REQUIRED_DIRECTIVES - directives_found
    if missing:
        issues.append(f"Missing required directive(s): {', '.join(sorted(missing))}")

    return issues
