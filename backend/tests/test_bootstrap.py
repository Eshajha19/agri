import importlib
import pytest

def test_import_has_no_side_effects(monkeypatch):
    # Patch external clients to detect calls
    monkeypatch.setenv("DB_URL", "fake")
    called = {}

    def fake_db_client(url):
        called["db"] = True
        return object()

    monkeypatch.setattr("backend.db.DatabaseClient", fake_db_client)

    # Import main.py
    importlib.reload(importlib.import_module("backend.main"))

    # ✅ No DB calls should have happened at import
    assert "db" not in called
