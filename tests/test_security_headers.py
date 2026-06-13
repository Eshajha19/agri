import json
import os
import re

import pytest

VERCEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "vercel.json")

REQUIRED_HEADERS = [
    "Content-Security-Policy",
    "Cross-Origin-Opener-Policy",
    "Cross-Origin-Embedder-Policy",
]

CSP_REQUIRED_DIRECTIVES = [
    "default-src",
    "script-src",
    "style-src",
    "img-src",
    "font-src",
    "connect-src",
    "frame-src",
    "frame-ancestors",
    "object-src",
    "base-uri",
    "form-action",
]

CSP_RESTRICTIVE = {
    "frame-ancestors": "'none'",
    "object-src": "'none'",
}


def _load_vercel_headers():
    with open(VERCEL_CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)
    headers_config = config.get("headers", [])
    if not headers_config:
        return {}
    # Return first source's headers as a dict
    for entry in headers_config:
        if entry.get("source") == "/(.*)":
            return {h["key"]: h["value"] for h in entry.get("headers", [])}
    return {}


class TestVercelSecurityHeaders:
    def _headers(self):
        return _load_vercel_headers()

    def test_all_required_headers_present(self):
        headers = self._headers()
        missing = [h for h in REQUIRED_HEADERS if h not in headers]
        assert not missing, f"Missing security headers in vercel.json: {missing}"

    def test_csp_has_required_directives(self):
        csp = self._headers().get("Content-Security-Policy", "")
        for directive in CSP_REQUIRED_DIRECTIVES:
            assert directive in csp, (
                f"CSP missing directive '{directive}' in: {csp}"
            )

    def test_csp_restrictive_directives(self):
        csp = self._headers().get("Content-Security-Policy", "")
        for directive, expected_value in CSP_RESTRICTIVE.items():
            # Find the directive's value segment
            match = re.search(rf"{directive}\s+([^;]+)", csp)
            assert match, f"CSP missing directive '{directive}'"
            value = match.group(1).strip()
            assert expected_value in value, (
                f"CSP directive '{directive}' should contain '{expected_value}', "
                f"got: {value}"
            )

    def test_coop_is_same_origin_allow_popups(self):
        coop = self._headers().get("Cross-Origin-Opener-Policy", "")
        assert (
            coop.strip() == "same-origin-allow-popups"
        ), f"COOP should be 'same-origin-allow-popups', got: {coop}"

    def test_coep_is_require_corp(self):
        coep = self._headers().get("Cross-Origin-Embedder-Policy", "")
        assert coep.strip() == "require-corp", (
            f"COEP should be 'require-corp', got: {coep}"
        )

    def test_x_frame_options_is_deny(self):
        headers = self._headers()
        xfo = headers.get("X-Frame-Options", "")
        if xfo:
            assert xfo.strip().upper() == "DENY", (
                f"X-Frame-Options should be 'DENY', got: {xfo}"
            )

    def test_permissions_policy_present(self):
        headers = self._headers()
        pp = headers.get("Permissions-Policy", "")
        assert pp, "Permissions-Policy header is missing in vercel.json"
