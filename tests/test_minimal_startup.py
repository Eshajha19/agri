"""
Verify the FastAPI application starts successfully when optional
dependencies are absent.  Optional imports (e.g. cloud-specific SDKs)
are protected by try/except ImportError guards — this test ensures those
guards work and the app's core API is still functional.

This test runs in an isolated subprocess so it can manipulate
sys.modules without affecting other tests.
"""
import subprocess
import sys

TEST_CODE = r"""
# Block optional packages before any other import.
import sys
import unittest.mock

OPTIONAL_PACKAGES = [
    "google.cloud.secretmanager",
    "google.cloud",
]

for pkg in OPTIONAL_PACKAGES:
    sys.modules[pkg] = unittest.mock.MagicMock(side_effect=ImportError)

# Now import the main app.
import main
from fastapi.testclient import TestClient

with TestClient(main.app) as client:
    # Health endpoint must respond without optional deps.
    resp = client.get("/health")
    assert resp.status_code == 200, f"/health returned {resp.status_code}"
    data = resp.json()
    assert data.get("status") == "healthy", f"Unexpected health status: {data}"

print("OK: App started and /health returned 200 with minimal dependencies")
sys.exit(0)
"""


import os


def test_startup_without_optional_deps():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, "-c", TEST_CODE],
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Minimal startup test failed (exit={result.returncode})\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )
