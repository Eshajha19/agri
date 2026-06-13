"""
Centralized Content Security Policy configuration.

Policies are generated per-environment so development can be
more permissive while production enforces strict rules.
"""

import os
from typing import Dict


# ── Environment detection ─────────────────────────────────────────────────

def is_production() -> bool:
    return os.getenv("ENV", "").strip().lower() in ("production", "prod")


# ── Policy builders ────────────────────────────────────────────────────────

def _directives(env: str) -> Dict[str, str]:
    """Return directive dict for the given environment key."""
    policies = {
        "production": {
            "default-src": "'self'",
            "script-src": "'self'",
            "style-src": "'self' 'unsafe-inline'",
            "img-src": "'self' data: https:",
            "font-src": "'self' https://fonts.gstatic.com",
            "connect-src": "'self' https://fasalsaathi.agri",
            "frame-ancestors": "'none'",
            "frame-src": "'none'",
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
            "connect-src": "'self' ws: wss: https://fasalsaathi.agri",
            "frame-ancestors": "'self'",
            "frame-src": "'self'",
            "object-src": "'none'",
            "base-uri": "'self'",
            "form-action": "'self'",
        },
    }
    return policies.get(env, policies["development"])


def build_csp_policy() -> str:
    """Return a single Content-Security-Policy header value for the
    current environment."""
    env = "production" if is_production() else "development"
    dirs = _directives(env)
    return "; ".join(f"{k} {v}" for k, v in dirs.items())


# ── Validation ─────────────────────────────────────────────────────────────

REQUIRED_DIRECTIVES = {
    "default-src", "script-src", "style-src", "img-src",
    "font-src", "connect-src", "frame-ancestors", "frame-src",
    "object-src", "base-uri", "form-action",
}

RESTRICTIVE_VALUES = {
    "frame-ancestors": ("'none'", "'self'"),
    "object-src": ("'none'",),
}


def validate_csp_policy(policy: str) -> list:
    """Validate a CSP policy string. Returns a list of issue descriptions
    (empty list = valid)."""
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
