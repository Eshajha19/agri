import pytest
import os
import sys
from fastapi.testclient import TestClient

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

@pytest.fixture
def client():
    """
    Fixture for the FastAPI TestClient.
    """
    with TestClient(app) as c:
        yield c
