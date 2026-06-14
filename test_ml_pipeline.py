"""
Tests for Secure ML Model Signature Verification (Issue #4).

Verifies that verify_and_load_joblib() correctly enforces HMAC-SHA256
signature checking before deserializing joblib model files, preventing
arbitrary code execution through tampered model artifacts.
"""
import os
import io
import hmac
import hashlib
import tempfile
import joblib
from unittest.mock import patch

_TEST_KEY = "test-signing-key-do-not-use-in-prod"


def _make_signed_model(directory: str, filename: str = "model.joblib"):
    """Create a tiny model file and its HMAC-SHA256 .sig file for tests."""
    model_path = os.path.join(directory, filename)
    sig_path = model_path + ".sig"
    model_obj = {"type": "mock_model", "version": "1.0", "weights": [0.1, 0.2, 0.3]}
    joblib.dump(model_obj, model_path)
    with open(model_path, "rb") as f:
        data = f.read()
    sig = hmac.new(_TEST_KEY.encode("utf-8"), data, hashlib.sha256).hexdigest()
    with open(sig_path, "w", encoding="utf-8") as sf:
        sf.write(sig)
    return model_path, sig_path, sig


def test_valid_signature_loads_model():
    """Model with a correct HMAC-SHA256 signature should load successfully."""
    print("Testing valid signature loads model...")
    with tempfile.TemporaryDirectory() as tmp:
        model_path, _, _ = _make_signed_model(tmp)
        env = {"MODEL_SIGNING_KEY": _TEST_KEY, "TESTING": "false",
               "ALLOW_UNSIGNED_MODELS": "false", "LOCAL_TEST_MODE": "false"}
        with patch.dict(os.environ, env):
            import importlib
            import ml.security as ml_sec
            importlib.reload(ml_sec)
            result = ml_sec.verify_and_load_joblib(model_path)
        assert result["type"] == "mock_model"
        assert result["version"] == "1.0"
    print("  [OK] Valid signature: model loaded correctly")
    return True


def test_tampered_model_raises_signature_error():
    """A model file modified after signing must be rejected with ModelSignatureError."""
    print("Testing tampered model is rejected...")
    with tempfile.TemporaryDirectory() as tmp:
        model_path, _, _ = _make_signed_model(tmp)
        # Tamper: append garbage bytes to simulate a corrupted/replaced model
        with open(model_path, "ab") as f:
            f.write(b"\x00\xFF\xDE\xAD\xBE\xEF")
        env = {"MODEL_SIGNING_KEY": _TEST_KEY, "TESTING": "false",
               "ALLOW_UNSIGNED_MODELS": "false", "LOCAL_TEST_MODE": "false"}
        with patch.dict(os.environ, env):
            import importlib
            import ml.security as ml_sec
            importlib.reload(ml_sec)
            try:
                ml_sec.verify_and_load_joblib(model_path)
                assert False, "Expected ModelSignatureError was not raised"
            except ml_sec.ModelSignatureError as e:
                assert "tampered" in str(e) or "failed" in str(e)
    print("  [OK] Tampered model correctly rejected with ModelSignatureError")
    return True


def test_missing_signature_file_raises():
    """A model with no .sig file must be rejected when signing is enforced."""
    print("Testing missing .sig file is rejected...")
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "unsigned_model.joblib")
        joblib.dump({"type": "unsigned"}, model_path)
        env = {"MODEL_SIGNING_KEY": _TEST_KEY, "TESTING": "false",
               "ALLOW_UNSIGNED_MODELS": "false", "LOCAL_TEST_MODE": "false"}
        with patch.dict(os.environ, env):
            import importlib
            import ml.security as ml_sec
            importlib.reload(ml_sec)
            try:
                ml_sec.verify_and_load_joblib(model_path)
                assert False, "Expected RuntimeError was not raised"
            except RuntimeError as e:
                assert "missing" in str(e).lower() or "signature" in str(e).lower()
    print("  [OK] Missing .sig file correctly raises RuntimeError")
    return True


def test_wrong_key_rejected():
    """A .sig file generated with a different key must be rejected."""
    print("Testing wrong signing key is rejected...")
    with tempfile.TemporaryDirectory() as tmp:
        model_path, sig_path, _ = _make_signed_model(tmp)
        # Overwrite sig with one computed using a *different* key
        with open(model_path, "rb") as f:
            data = f.read()
        wrong_sig = hmac.new(b"completely-wrong-key", data, hashlib.sha256).hexdigest()
        with open(sig_path, "w") as sf:
            sf.write(wrong_sig)
        env = {"MODEL_SIGNING_KEY": _TEST_KEY, "TESTING": "false",
               "ALLOW_UNSIGNED_MODELS": "false", "LOCAL_TEST_MODE": "false"}
        with patch.dict(os.environ, env):
            import importlib
            import ml.security as ml_sec
            importlib.reload(ml_sec)
            try:
                ml_sec.verify_and_load_joblib(model_path)
                assert False, "Expected ModelSignatureError was not raised"
            except ml_sec.ModelSignatureError:
                pass
    print("  [OK] Wrong signing key correctly rejected")
    return True


def test_allow_unsigned_bypasses_for_dev():
    """ALLOW_UNSIGNED_MODELS=true should load model without key or .sig file."""
    print("Testing ALLOW_UNSIGNED_MODELS dev bypass...")
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "local_model.joblib")
        joblib.dump({"type": "local_dev_model"}, model_path)
        # No .sig file — simulates local dev environment
        with patch.dict(os.environ, {"ALLOW_UNSIGNED_MODELS": "true", "MODEL_SIGNING_KEY": ""}):
            import importlib
            import ml.security as ml_sec
            importlib.reload(ml_sec)
            result = ml_sec.verify_and_load_joblib(model_path)
        assert result["type"] == "local_dev_model"
    print("  [OK] ALLOW_UNSIGNED_MODELS=true bypasses verification (dev mode only)")
    return True


def test_sign_model_creates_sig_file():
    """sign_model() should create a valid .sig file beside the model."""
    print("Testing sign_model() creates .sig file...")
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.joblib")
        joblib.dump({"v": 1}, model_path)
        with patch.dict(os.environ, {"MODEL_SIGNING_KEY": _TEST_KEY}):
            from ml.security import sign_model
            sig = sign_model(model_path)
        sig_path = model_path + ".sig"
        assert os.path.exists(sig_path), ".sig file was not created"
        with open(sig_path) as f:
            written = f.read().strip()
        assert written == sig
        assert len(sig) == 64  # SHA256 hex digest is always 64 chars
    print("  [OK] sign_model() creates correct .sig file with 64-char hex digest")
    return True


def test_sign_then_verify_roundtrip():
    """Signing a model then verifying it should succeed end-to-end."""
    print("Testing sign -> verify roundtrip...")
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "roundtrip.joblib")
        original = {"weights": [1.5, 2.5], "bias": 0.1}
        joblib.dump(original, model_path)
        env = {"MODEL_SIGNING_KEY": _TEST_KEY, "TESTING": "false",
               "ALLOW_UNSIGNED_MODELS": "false", "LOCAL_TEST_MODE": "false"}
        with patch.dict(os.environ, env):
            import importlib
            import ml.security as ml_sec
            importlib.reload(ml_sec)
            ml_sec.sign_model(model_path)
            loaded = ml_sec.verify_and_load_joblib(model_path)
        assert loaded["weights"] == original["weights"]
        assert loaded["bias"] == original["bias"]
    print("  [OK] sign -> verify roundtrip works correctly end-to-end")
    return True


def test_drift_triggers_model_rollback():
    """Test that high prediction drift triggers model rollback in the registry."""
    print("Testing drift triggers model rollback...")
    
    class DummyModel:
        def __init__(self, name):
            self.name = name
        def predict(self, data):
            return [1.0]

    stable_model = DummyModel("stable")
    active_model = DummyModel("active")

    from ml.registry import ModelRegistry
    ModelRegistry.register("xgboost", active_model)
    assert ModelRegistry.get("xgboost") == active_model

    from backend.routers import ml as ml_router
    async def mock_verify_role(req):
        return {"uid": "test_user"}
    
    class DummyModelRouter:
        def predict(self, data, context=None):
            model = ModelRegistry.get("xgboost")
            return model.predict(data)
            
    ml_router.init_router(DummyModelRouter(), stable_model, verify_role=mock_verify_role)

    from ml.governance.drift_detector import DriftDetector
    from backend.routers import governance
    detector = DriftDetector(window_size=10, prediction_drift_threshold=0.1)
    governance.drift_detector = detector
    
    ml_router.init_router(DummyModelRouter(), stable_model, verify_role=mock_verify_role)
    
    detector.set_baseline("xgboost", [1.0] * 20)
    
    drift_detected = False
    for i in range(15):
        detected, alert = detector.check_prediction_drift("xgboost", 5.0)
        if detected:
            drift_detected = True
            
    assert drift_detected, "Drift should have been detected"
    current_model = ModelRegistry.get("xgboost")
    assert current_model == stable_model, "Active model should have rolled back to the stable model"
    print("  [OK] Model rollback triggered successfully by drift detection")
    return True


def test_shadow_evaluator_rejection_triggers_rollback():
    """Test that a rejected candidate model in shadow evaluation triggers rollback (safety alert)."""
    print("Testing shadow evaluator rejection triggers rollback...")
    
    class DummyModel:
        def __init__(self, name):
            self.name = name
    
    stable_model = DummyModel("stable")
    active_model = DummyModel("active")
    
    from ml.registry import ModelRegistry
    ModelRegistry.register("xgboost", active_model)
    
    from backend.routers import ml as ml_router
    async def mock_verify_role(req):
        return {"uid": "test_user"}
        
    class DummyModelRouter:
        def predict(self, data, context=None):
            return [1.0]
            
    ml_router.init_router(DummyModelRouter(), stable_model, verify_role=mock_verify_role)
    
    from ml.governance.shadow_evaluator import ShadowEvaluator
    from backend.routers import governance
    evaluator = ShadowEvaluator(min_samples=5, error_improvement_threshold=0.05)
    governance.shadow_evaluator = evaluator
    
    ml_router.init_router(DummyModelRouter(), stable_model, verify_role=mock_verify_role)
    
    eval_id = evaluator.start_shadow_evaluation("xgboost", "candidate_xgboost")
    
    for _ in range(10):
        evaluator.record_predictions(eval_id, 1.0, 10.0, 1.0)
        
    result = evaluator.evaluate_candidate(eval_id)
    assert result.recommendation == "reject"
    
    current_model = ModelRegistry.get("xgboost")
    assert current_model == stable_model, "Active model should have rolled back to the stable model on shadow rejection"
    print("  [OK] Model rollback triggered successfully by shadow rejection")
    return True


def test_celery_worker_model_loaders():
    """Test caching and locking behavior of model loaders in celery_worker."""
    print("Testing celery_worker model loaders caching and lock...")
    env = {
        "ALLOW_INSECURE_REDIS": "true",
        "MODEL_SIGNING_KEY": _TEST_KEY,
        "TESTING": "true",
        "ALLOW_UNSIGNED_MODELS": "false",
        "LOCAL_TEST_MODE": "false",
    }
    with patch.dict(os.environ, env):
        import celery_worker
        
        # Reset cached global variables to ensure fresh test state
        celery_worker._model_lag = None
        celery_worker._model_trend = None
        
        with patch("celery_worker.verify_and_load_joblib") as mock_verify_load:
            mock_verify_load.return_value = "mock_lag_model"
            
            # Call multiple times to verify caching and lock
            m1 = celery_worker._get_lag_model()
            m2 = celery_worker._get_lag_model()
            
            assert m1 == "mock_lag_model"
            assert m2 == "mock_lag_model"
            # verify_and_load_joblib should be called exactly once because of caching
            mock_verify_load.assert_called_once_with("sklearn_yield_model.joblib")
            
        with patch("celery_worker.verify_and_load_joblib") as mock_verify_load, \
             patch("os.path.exists", return_value=True):
            mock_verify_load.return_value = "mock_trend_model"
            
            m1 = celery_worker._get_trend_model()
            m2 = celery_worker._get_trend_model()
            
            assert m1 == "mock_trend_model"
            assert m2 == "mock_trend_model"
            # verify_and_load_joblib should be called exactly once because of caching
            mock_verify_load.assert_called_once_with("trend_forecast_model.joblib")
            
    print("  [OK] celery_worker model loaders double-checked locking & caching verified")
    return True


def main():
    print("=" * 60)
    print("ML Model Security Test Suite (Issue #4)")
    print("=" * 60)
    tests = [
        test_valid_signature_loads_model,
        test_tampered_model_raises_signature_error,
        test_missing_signature_file_raises,
        test_wrong_key_rejected,
        test_allow_unsigned_bypasses_for_dev,
        test_sign_model_creates_sig_file,
        test_sign_then_verify_roundtrip,
        test_drift_triggers_model_rollback,
        test_shadow_evaluator_rejection_triggers_rollback,
        test_celery_worker_model_loaders,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
    print("=" * 60)
    if failed == 0:
        print(f"[OK] ALL {passed} TESTS PASSED")
    else:
        print(f"[FAIL] {failed} FAILED, {passed} PASSED")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
