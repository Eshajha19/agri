"""
Centralized Content Security Policy configuration.

Policies are generated per-environment so development can be
more permissive while production enforces strict rules.
Firebase Authentication domains are included in both environments
so Google Sign-In popups and redirects are never blocked.
"""

import os

FIREBASE_AUTH_DOMAINS = [
    "https://*.firebaseapp.com",
    "https://*.web.app",
    "https://accounts.google.com",
    "https://apis.google.com",
]


def is_production() -> bool:
    return os.getenv("ENV", "").strip().lower() in ("production", "prod")


def _directives(env: str) -> dict:
    policies = {
        "production": {
            "default-src": "'self'",
            "script-src": "'self'",
            "style-src": "'self' 'unsafe-inline'",
            "img-src": "'self' data: https:",
            "font-src": "'self' https://fonts.gstatic.com",
            "connect-src": "'self' https://fasalsaathi.agri https://identitytoolkit.googleapis.com https://securetoken.googleapis.com",
            "frame-src": "'self' " + " ".join(FIREBASE_AUTH_DOMAINS),
            "frame-ancestors": "'none'",
            "object-src": "'none'",
            "base-uri": "'self'",
            "form-action": "'self'",
        },
        "development": {
            "default-src": "'self'",
            "script-src": "'self' 'unsafe-inline' 'unsafe-eval'",
            "style-src": "'self' 'unsafe-inline'",
            "img-src": "'self' data: https:",
            "font-src": "'self' https://fonts.gstatic.com",
            "connect-src": "'self' ws: wss: https://fasalsaathi.agri https://identitytoolkit.googleapis.com https://securetoken.googleapis.com",
            "frame-src": "'self' " + " ".join(FIREBASE_AUTH_DOMAINS),
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

    # Validate Firebase auth domains are included in frame-src
    if "frame-src" in directives_found:
        for part in parts:
            if part.startswith("frame-src"):
                vals = part.split(None, 1)[1] if len(part.split(None, 1)) > 1 else ""
                for domain in FIREBASE_AUTH_DOMAINS:
                    if domain not in vals:
                        issues.append(
                            f"frame-src missing Firebase auth domain: {domain}"
                        )
                        break
                break

    return issues
