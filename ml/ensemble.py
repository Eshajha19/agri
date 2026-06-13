"""
Ensemble Stacking Engine
========================
Combines XGBoost, LSTM, and Random Forest predictions with learned weights,
bootstrap confidence intervals, and inter-model disagreement detection.
"""

import concurrent.futures
import json
import logging
import math
import os
from datetime import datetime as _dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from ml.security import verify_and_load_joblib
import pandas as pd

_DEFAULT_MODEL_TIMEOUT = 10.0  # seconds per model

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================

_WEIGHTS_PATH = Path("ensemble_weights.json")
_PREDICTIONS_LOG_PATH = Path("ensemble_predictions.jsonl")
_DISAGREEMENT_THRESHOLD = 0.25  # CV threshold for flagging high disagreement
_BOOTSTRAP_SAMPLES = 1000


# =============================================================================
# MODEL LOADERS
# =============================================================================

def _load_model_with_timeout(loader_fn, timeout: float, model_name: str):
    """
    Execute a model loader in a thread with a timeout.
    Returns the loaded model on success, None on timeout/failure.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(loader_fn)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning("%s model loading timed out after %.1fs", model_name, timeout)
            return None
        except Exception as exc:
            logger.warning("%s model loading failed: %s", model_name, exc)
            return None


def _load_xgboost_model():
    from ml.adapters.xgboost_adapter import XGBoostAdapter

    adapter = XGBoostAdapter()
    path = "yield_model.joblib"
    if not os.path.exists(path):
        raise FileNotFoundError(f"XGBoost model not found: {path}")
    adapter.load(path)
    return adapter


def _load_lstm_model():
    from tensorflow.keras.models import load_model

    model_path = "lstm_yield_model.h5"
    scaler_path = "lstm_scaler.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model not found: {model_path}")
    model = load_model(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler


def _load_rf_model():
    path = "sklearn_yield_model.joblib"
    # Fallback to .pkl if .joblib missing (migration safety)
    if not os.path.exists(path):
        path = "sklearn_yield_model.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"RF model not found: sklearn_yield_model.joblib or .pkl")
    return verify_and_load_joblib(path)


# =============================================================================
# ENSEMBLE STACKER
# =============================================================================

class _ModelStatus:
    """Tracks health and metadata for a single model slot."""

    def __init__(self, name: str):
        self.name = name
        self.available = False
        self.load_error: Optional[str] = None
        self.load_duration_ms: Optional[float] = None
        self.last_loaded: Optional[str] = None


class EnsembleStacker:
    """
    Stacked ensemble with learned weights, bootstrap CIs, and disagreement detection.
    Supports lazy per-model loading with timeouts and graceful degradation.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        model_timeout: float = _DEFAULT_MODEL_TIMEOUT,
    ):
        self.xgb_adapter = None
        self.lstm_model = None
        self.lstm_scaler = None
        self.rf_model = None
        self.weights = weights or self._load_or_default_weights()
        self._models_loaded = False
        self._model_timeout = model_timeout
        self._status = {
            "xgboost": _ModelStatus("xgboost"),
            "lstm": _ModelStatus("lstm"),
            "random_forest": _ModelStatus("random_forest"),
        }

    # -------------------------------------------------------------------------
    # HEALTH
    # -------------------------------------------------------------------------

    def health(self) -> Dict[str, any]:
        """
        Return per-model health status for monitoring.
        """
        return {
            "ensemble_ready": any(s.available for s in self._status.values()),
            "models": {
                name: {
                    "available": status.available,
                    "load_error": status.load_error,
                    "load_duration_ms": status.load_duration_ms,
                    "last_loaded": status.last_loaded,
                }
                for name, status in self._status.items()
            },
            "weights": self.weights,
            "timestamp": _dt.utcnow().isoformat(),
        }

    # -------------------------------------------------------------------------
    # WEIGHTS
    # -------------------------------------------------------------------------

    @staticmethod
    def _default_weights() -> Dict[str, float]:
        return {"xgboost": 0.5, "lstm": 0.25, "random_forest": 0.25}

    def _load_or_default_weights(self) -> Dict[str, float]:
        if _WEIGHTS_PATH.exists():
            try:
                data = json.loads(_WEIGHTS_PATH.read_text(encoding="utf-8"))
                return data.get("weights", self._default_weights())
            except Exception:
                logger.warning("Failed loading ensemble weights, using defaults")
        return self._default_weights()

    def _save_weights(self):
        record = {
            "weights": self.weights,
            "updated_at": _dt.utcnow().isoformat(),
        }
        tmp = _WEIGHTS_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        os.replace(tmp, _WEIGHTS_PATH)

    # -------------------------------------------------------------------------
    # MODEL LOADING
    # -------------------------------------------------------------------------

    def _load_single(self, name: str, loader_fn):
        """Load one model with timeout, updating status metadata."""
        status = self._status[name]
        start = _dt.utcnow()
        try:
            result = _load_model_with_timeout(loader_fn, self._model_timeout, name)
            if result is not None:
                status.available = True
                status.load_error = None
                status.load_duration_ms = round(
                    (_dt.utcnow() - start).total_seconds() * 1000, 2
                )
                status.last_loaded = _dt.utcnow().isoformat()
                return result
            else:
                status.available = False
                status.load_error = "timeout_or_failure"
        except Exception as exc:
            status.available = False
            status.load_error = str(exc)
        return None

    def load_models(self):
        """Lazy-load all three models with per-model timeouts. Safe to call multiple times."""
        if self._models_loaded:
            return

        self.xgb_adapter = self._load_single("xgboost", _load_xgboost_model)

        lstm_result = self._load_single("lstm", _load_lstm_model)
        if lstm_result is not None:
            self.lstm_model, self.lstm_scaler = lstm_result
        else:
            self.lstm_model = None
            self.lstm_scaler = None

        self.rf_model = self._load_single("random_forest", _load_rf_model)

        self._models_loaded = True

    # -------------------------------------------------------------------------
    # INDIVIDUAL PREDICTIONS
    # -------------------------------------------------------------------------

    def _predict_xgboost(self, input_data: dict) -> Optional[float]:
        if self.xgb_adapter is None:
            return None
        try:
            return float(self.xgb_adapter.predict(input_data))
        except Exception as exc:
            logger.warning("XGBoost prediction failed: %s", exc)
            return None

    def _predict_lstm(self, input_data: dict) -> Optional[float]:
        if self.lstm_model is None or self.lstm_scaler is None:
            return None
        try:
            # LSTM expects sequential lag features; extract from input if present
            # Fallback: use numeric features reshaped as a single-step sequence
            features = []
            for i in range(1, 6):
                key = f"lag_{i}"
                if key in input_data:
                    features.append(float(input_data[key]))
                else:
                    # Use available numeric fields as proxy
                    features.append(float(input_data.get("CropCoveredArea", 0)))
            arr = np.array(features).reshape(1, 5, 1)
            scaled = self.lstm_scaler.transform(arr.reshape(-1, 1)).reshape(1, 5, 1)
            pred = self.lstm_model.predict(scaled, verbose=0).squeeze()
            # Inverse transform
            pred_2d = np.array([[float(pred)]])
            unscaled = self.lstm_scaler.inverse_transform(pred_2d)[0][0]
            return float(unscaled)
        except Exception as exc:
            logger.warning("LSTM prediction failed: %s", exc)
            return None

    def _predict_rf(self, input_data: dict) -> Optional[float]:
        if self.rf_model is None:
            return None
        try:
            # RF expects lag_1..lag_5 features
            features = []
            for i in range(1, 6):
                key = f"lag_{i}"
                if key in input_data:
                    features.append(float(input_data[key]))
                else:
                    features.append(float(input_data.get("CropCoveredArea", 0)))
            arr = np.array(features).reshape(1, -1)
            return float(self.rf_model.predict(arr)[0])
        except Exception as exc:
            logger.warning("RF prediction failed: %s", exc)
            return None

    # -------------------------------------------------------------------------
    # ENSEMBLE PREDICTION
    # -------------------------------------------------------------------------

    def predict(self, input_data: dict) -> Dict[str, any]:
        """
        Single prediction with confidence interval and model contributions.
        """
        self.load_models()

        preds = {
            "xgboost": self._predict_xgboost(input_data),
            "lstm": self._predict_lstm(input_data),
            "random_forest": self._predict_rf(input_data),
        }

        available = {k: v for k, v in preds.items() if v is not None}
        if not available:
            raise RuntimeError("No models available for ensemble prediction")

        # Weighted ensemble point estimate
        total_weight = sum(self.weights.get(k, 0) for k in available)
        if total_weight == 0:
            total_weight = len(available)
            weights = {k: 1.0 / total_weight for k in available}
        else:
            weights = {k: self.weights.get(k, 0) / total_weight for k in available}

        point_estimate = sum(weights[k] * v for k, v in available.items())

        # Bootstrap confidence interval
        samples = []
        for _ in range(_BOOTSTRAP_SAMPLES):
            # Resample models with replacement
            boot_models = np.random.choice(list(available.keys()), size=len(available), replace=True)
            boot_preds = [available[m] for m in boot_models]
            boot_weights = [weights[m] for m in boot_models]
            boot_total = sum(boot_weights)
            if boot_total == 0:
                continue
            boot_estimate = sum((w / boot_total) * p for w, p in zip(boot_weights, boot_preds))
            samples.append(boot_estimate)

        samples = sorted(samples)
        lower = np.percentile(samples, 5)
        upper = np.percentile(samples, 95)

        # Disagreement detection
        values = list(available.values())
        mean_val = sum(values) / len(values)
        cv = (np.std(values) / mean_val) if mean_val != 0 else 0
        high_disagreement = cv > _DISAGREEMENT_THRESHOLD

        result = {
            "point_estimate": round(point_estimate, 2),
            "confidence_interval": {
                "lower": round(lower, 2),
                "upper": round(upper, 2),
            },
            "confidence_level": 0.90,
            "model_predictions": {k: round(v, 2) for k, v in available.items()},
            "model_weights": {k: round(v, 4) for k, v in weights.items()},
            "disagreement": {
                "coefficient_of_variation": round(cv, 4),
                "high_disagreement": high_disagreement,
                "threshold": _DISAGREEMENT_THRESHOLD,
            },
            "models_used": list(available.keys()),
            "models_missing": [k for k in preds if preds[k] is None],
            "timestamp": _dt.utcnow().isoformat(),
        }

        self._log_prediction(result)
        return result

    # -------------------------------------------------------------------------
    # MULTI-STEP FORECAST
    # -------------------------------------------------------------------------

    def multi_step_forecast(
        self,
        input_data: dict,
        steps: int = 3,
    ) -> List[Dict[str, any]]:
        """
        Recursive multi-step forecast. Each step feeds the previous prediction
        back as lag features.
        """
        self.load_models()

        forecasts = []
        current = dict(input_data)

        for step in range(1, steps + 1):
            result = self.predict(current)
            result["step"] = step
            forecasts.append(result)

            # Update lags for next step
            prev_lags = {f"lag_{i}": current.get(f"lag_{i}", 0) for i in range(1, 5)}
            for i in range(5, 1, -1):
                current[f"lag_{i}"] = prev_lags.get(f"lag_{i - 1}", 0)
            current["lag_1"] = result["point_estimate"]

        return forecasts

    # -------------------------------------------------------------------------
    # WEIGHT LEARNING
    # -------------------------------------------------------------------------

    def learn_weights(
        self,
        validation_data: List[Tuple[dict, float]],
    ) -> Dict[str, float]:
        """
        Optimize stacking weights using validation RMSE.
        """
        self.load_models()

        model_errors = {k: [] for k in ["xgboost", "lstm", "random_forest"]}

        for features, actual in validation_data:
            preds = {
                "xgboost": self._predict_xgboost(features),
                "lstm": self._predict_lstm(features),
                "random_forest": self._predict_rf(features),
            }
            for k, p in preds.items():
                if p is not None:
                    model_errors[k].append((p - actual) ** 2)

        # Inverse-RMSE weighting
        inv_rmse = {}
        for k, errors in model_errors.items():
            if errors:
                rmse = math.sqrt(sum(errors) / len(errors))
                inv_rmse[k] = 1.0 / max(rmse, 1e-6)
            else:
                inv_rmse[k] = 0.0

        total = sum(inv_rmse.values())
        if total == 0:
            self.weights = self._default_weights()
        else:
            self.weights = {k: v / total for k, v in inv_rmse.items()}

        self._save_weights()
        return self.weights

    # -------------------------------------------------------------------------
    # LOGGING
    # -------------------------------------------------------------------------

    def _log_prediction(self, result: dict):
        try:
            line = json.dumps(result, default=str) + "\n"
            with open(_PREDICTIONS_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as exc:
            logger.warning("Failed logging ensemble prediction: %s", exc)

    # -------------------------------------------------------------------------
    # DISAGREEMENT HISTORY
    # -------------------------------------------------------------------------

    def recent_disagreements(self, limit: int = 50) -> List[dict]:
        """
        Return recent predictions where models disagreed significantly.
        """
        if not _PREDICTIONS_LOG_PATH.exists():
            return []

        try:
            lines = _PREDICTIONS_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
            records = [json.loads(line) for line in lines if line.strip()]
            flagged = [r for r in records if r.get("disagreement", {}).get("high_disagreement")]
            return flagged[-limit:]
        except Exception as exc:
            logger.warning("Failed reading disagreement history: %s", exc)
            return []


# =============================================================================
# SINGLETON
# =============================================================================

_ensemble_stacker: Optional[EnsembleStacker] = None


def get_ensemble_stacker() -> EnsembleStacker:
    global _ensemble_stacker
    if _ensemble_stacker is None:
        _ensemble_stacker = EnsembleStacker()
    return _ensemble_stacker