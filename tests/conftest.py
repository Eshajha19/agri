import pytest
import os
import sys
from fastapi.testclient import TestClient

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from main import app
except Exception:
    app = None


@pytest.fixture
def client():
    """
    Fixture for the FastAPI TestClient. If the app failed to import (missing optional
    dependencies such as firebase_admin), tests depending on `client` will be skipped.
    """
    if app is None:
        pytest.skip("FastAPI app unavailable in this environment")

    with TestClient(app) as c:
        yield c
