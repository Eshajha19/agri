"""Tests for centralized CSP configuration and validation."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from csp import build_csp_policy, validate_csp_policy, REQUIRED_DIRECTIVES


class TestCSPPolicy:
    def test_development_policy_has_all_required_directives(self):
        os.environ.pop("ENV", None)  # default → development
        policy = build_csp_policy()
        assert validate_csp_policy(policy) == []

    def test_production_policy_is_restrictive(self):
        os.environ["ENV"] = "production"
        policy = build_csp_policy()
        issues = validate_csp_policy(policy)
        assert issues == [], f"Production CSP validation failed: {issues}"
        assert "'unsafe-inline'" not in policy.split("script-src")[-1].split(";")[0]

    def test_validation_detects_missing_directive(self):
        issues = validate_csp_policy("default-src 'self'")
        missing = [d for d in REQUIRED_DIRECTIVES if d != "default-src"]
        assert any("Missing required" in i for i in issues)

    def test_validation_detects_permissive_frame_ancestors(self):
        policy = "default-src 'self'; frame-ancestors https://evil.com"
        issues = validate_csp_policy(policy)
        assert any("frame-ancestors" in i for i in issues)
