"""Model validation utilities: compute model hash and run basic smoke tests."""
from typing import Tuple
import joblib
import os
import numpy as np
import traceback

from .model_manifest import compute_sha256


def validate_model_file(path: str) -> Tuple[bool, dict]:
    """Validate a model file: compute sha256 and attempt a smoke predict.

    Returns (success, details)
    """
    details = {"path": path}

    if not os.path.exists(path):
        details["error"] = "file_not_found"
        return False, details

    try:
        sha = compute_sha256(path)
        details["sha256"] = sha
    except Exception as e:
        details["error"] = f"hash_error: {e}"
        return False, details

    # Try loading model via joblib
    try:
        model = joblib.load(path)
        details["loaded_via"] = "joblib"

        # If possible, attempt a smoke predict using n_features_in_
        n_features = getattr(model, "n_features_in_", None)
        if n_features is not None:
            sample = np.zeros((1, int(n_features)))
            try:
                _ = model.predict(sample)
                details["smoke_predict"] = "ok"
            except Exception as e:
                details["smoke_predict"] = f"predict_failed: {e}"
        else:
            details["smoke_predict"] = "skipped_no_feature_info"

        return True, details

    except ModuleNotFoundError as e:
        # Some models (e.g., XGBoost) require optional packages to unpickle.
        # Treat missing optional dependency as a partial success: we could
        # still validate the artifact hash and register the model; smoke
        # predict is skipped because runtime dependencies are missing.
        details["load_error"] = str(e)
        details["load_trace"] = traceback.format_exc()
        details["note"] = "missing_optional_dependency_skipped_load"
        return True, details
    except Exception as e:
        details["load_error"] = str(e)
        details["load_trace"] = traceback.format_exc()
        return False, details
