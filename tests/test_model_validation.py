import os
from ml.validator import validate_model_file
from ml.model_manifest import create_manifest
from ml_model_registry import get_model_registry


def test_validate_yield_joblib():
    path = os.path.abspath("yield_model.joblib")
    assert os.path.exists(path), "yield_model.joblib must exist for this test"

    ok, details = validate_model_file(path)
    assert ok, f"Validation failed: {details}"

    manifest = create_manifest(path, "yield_prediction", "1.0", created_by="test")
    assert "sha256" in manifest

    registry = get_model_registry()
    model = registry.register_model(
        model_name=manifest["model_name"],
        version=manifest["version"],
        model_path=path,
        created_by=manifest["created_by"],
        description="Test register"
    )

    assert registry.get_model_version("yield_prediction", "1.0") == model
