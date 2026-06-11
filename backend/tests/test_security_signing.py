import os, pytest
from ml import security

def test_production_disallows_unsigned(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("ALLOW_UNSIGNED_MODELS", "true")
    with pytest.raises(RuntimeError):
        importlib.reload(security)  # reload to trigger guardrail

def test_dev_allows_unsigned(monkeypatch, tmp_path):
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("ALLOW_UNSIGNED_MODELS", "true")
    model_file = tmp_path / "m.joblib"
    model_file.write_text("dummy")
    # Should load without signature
    obj = security.verify_and_load_joblib(str(model_file))
    assert obj is not None
