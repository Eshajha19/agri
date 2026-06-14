import json
import logging
import os
import threading
from datetime import datetime as _dt
from pathlib import Path
from typing import Optional

import numpy as np
from celery import Celery
from ml.security import verify_and_load_joblib

logger = logging.getLogger(__name__)

# =============================================================================
# CELERY CONFIG
# =============================================================================

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "agri_ml_tasks",
    broker=redis_url,
    backend=redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
    result_expires=3600,
)

# =============================================================================
# GLOBAL CACHED MODELS
# =============================================================================

_model_lag = None
_model_trend = None
_ml_router = None
_ensemble_stacker = None
_model_lag_lock = threading.Lock()
_model_trend_lock = threading.Lock()
_ml_router_lock = threading.Lock()
_ensemble_stacker_lock = threading.Lock()


# =============================================================================
# MODEL LOADERS
# =============================================================================

def _get_lag_model():
    global _model_lag

    if _model_lag is None:
        try:
            model_path = "sklearn_yield_model.joblib"

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"{model_path} not found")

            _model_lag = verify_and_load_joblib(model_path)

            logger.info("Lag model loaded successfully")

        except Exception:
            logger.exception("Failed to load lag model")
            raise

    return _model_lag


def _get_trend_model():
    global _model_trend

    if _model_trend is None:
        try:
            model_path = "trend_forecast_model.joblib"

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"{model_path} not found")

            _model_trend = verify_and_load_joblib(model_path)

            logger.info("Trend model loaded successfully")

        except Exception:
            logger.exception("Failed to load trend model")
            raise

    return _model_trend


def _get_ml_router():
    global _ml_router

    if _ml_router is None:
        try:
            from ml.adapters.xgboost_adapter import XGBoostAdapter
            from ml.registry import ModelRegistry
            from ml.router import ModelRouter

            model_path = "yield_model.joblib"

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"{model_path} not found")

            xgb_adapter = XGBoostAdapter()
            xgb_adapter.load(model_path)

            ModelRegistry.register("xgboost", xgb_adapter)

            _ml_router = ModelRouter(default_model="xgboost")

            logger.info("ML router initialized successfully")

        except Exception:
            logger.exception("Failed to initialize ML router")
            raise

    return _ml_router

@celery_app.task(
    bind=True,
    name="predict_yield_batch_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 2},
    soft_time_limit=30,
    time_limit=45,
)
def predict_yield_batch_task(
    self,
    inputs: list[dict],
    context: Optional[dict] = None,
):
    """
    Batch yield prediction using ML router.
    """

    try:
        router = _get_ml_router()

        predictions = router.predict_batch(
            inputs,
            context,
        )

        return {
            "predictions": predictions,
            "count": len(predictions),
            "model": router.default_model,
        }

    except Exception:
        logger.exception(
            "Batch yield prediction task failed"
        )
        raise


@celery_app.task(
    bind=True,
    name="predict_ensemble_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 2},
    soft_time_limit=30,
    time_limit=45,
)
def predict_ensemble_task(
    self,
    input_data: dict,
):
    """
    Ensemble prediction.
    """

    try:
        stacker = _get_ensemble_stacker()

        result = stacker.predict(input_data)

        return result

    except RuntimeError:
        logger.exception(
            "Ensemble prediction failed: no models available"
        )
        raise

    except Exception:
        logger.exception(
            "Ensemble prediction task failed"
        )
        raise
def _get_ensemble_stacker():
    global _ensemble_stacker

    if _ensemble_stacker is None:
        with _ensemble_stacker_lock:
            if _ensemble_stacker is None:
                try:
                    from ml.ensemble import EnsembleStacker

                    _ensemble_stacker = EnsembleStacker()
                    logger.info("Ensemble stacker initialized successfully")
                except Exception:
                    logger.exception("Failed to initialize ensemble stacker")
                    raise

    return _ensemble_stacker


# =============================================================================
# HELPERS
# =============================================================================

def _validate_numeric_list(data, expected_length=5):
    if not isinstance(data, list):
        raise ValueError("Input must be a list")

    if len(data) != expected_length:
        raise ValueError(f"Exactly {expected_length} values are required")

    validated = []

    for value in data:
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("All values must be numeric")

        if not np.isfinite(value):
            raise ValueError("Invalid numeric value")

        validated.append(value)

    return validated


# =============================================================================
# TASKS
# =============================================================================

@celery_app.task(
    bind=True,
    name="predict_yield_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
    soft_time_limit=25,
    time_limit=30,
)
def predict_yield_task(self, input_data: dict, context: dict):
    """
    Yield prediction using ML router.
    """

    try:
        router = _get_ml_router()

        prediction = router.predict(input_data, context)

        return {
            "predicted_ExpYield": round(float(prediction), 2)
        }

    except Exception:
        logger.exception("Yield prediction task failed")
        raise


@celery_app.task(
    bind=True,
    name="predict_yield_lag_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
    soft_time_limit=25,
    time_limit=30,
)
def predict_yield_lag_task(self, data: list):
    """
    Time-series lag prediction.
    """

    try:
        validated = _validate_numeric_list(data)

        model = _get_lag_model()

        data_arr = np.array(validated).reshape(1, -1)

        prediction = model.predict(data_arr)

        return {
            "prediction": round(float(prediction[0]), 2),
            "model": "RandomForest Time Series (Lag Features)",
        }

    except Exception:
        logger.exception("Lag prediction task failed")
        raise


@celery_app.task(
    bind=True,
    name="predict_yield_trend_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
    soft_time_limit=25,
    time_limit=30,
)
def predict_yield_trend_task(self, data: list):
    """
    Multi-step trend forecasting.
    """

    try:
        validated = _validate_numeric_list(data)

        model = _get_trend_model()

        temp = list(validated)

        trend = []

        for _ in range(5):
            features = temp[-5:]

            pred = model.predict([features])[0]

            pred_value = round(float(pred), 2)

            trend.append(pred_value)

            temp.append(pred_value)

        return {
            "trend": trend,
            "prediction": trend[-1],
            "model": "RandomForest Trend Forecast (Lag Features)"
        }

    except Exception:
        logger.exception("Trend prediction task failed")
        raise


@celery_app.task(
    bind=True,
    name="predict_yield_batch_task",
    name="predict_ensemble_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 2},
    soft_time_limit=30,
    time_limit=45,
)
def predict_yield_batch_task(self, inputs: list[dict], context: Optional[dict] = None):
    """
    Batch yield prediction using ML router.

    Accepts a list of input dicts and returns aligned predictions.
    """
    try:
        router = _get_ml_router()

        predictions = router.predict_batch(inputs, context)

        return {
            "predictions": predictions,
            "count": len(predictions),
            "model": router.default_model,
        }

    except Exception:
        logger.exception("Batch yield prediction task failed")
def predict_ensemble_task(self, input_data: dict):
    """
    Ensemble prediction with lazy per-model loading and graceful degradation.
    """

    try:
        stacker = _get_ensemble_stacker()

        result = stacker.predict(input_data)

        return result

    except RuntimeError as exc:
        logger.error("Ensemble prediction failed: no models available")
        raise

    except Exception:
        logger.exception("Ensemble prediction task failed")
        raise


@celery_app.task(
    bind=True,
    name="process_whatsapp_webhook_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
    soft_time_limit=20,
    time_limit=25,
)
def process_whatsapp_webhook_task(self, body: str, sender_number: str):
    """
    Async WhatsApp processing task.
    """

    try:
        from whatsapp_service import process_webhook_message

        if not isinstance(body, str):
            raise ValueError("body must be string")

        if not isinstance(sender_number, str):
            raise ValueError("sender_number must be string")

        body = body.strip()[:2000]
        sender_number = sender_number.strip()[:30]

        result = process_webhook_message(body, sender_number)

        return {
            "status": "processed",
            "sender": sender_number,
            "result": result,
        }

    except Exception:
        logger.exception("WhatsApp webhook task failed")
        raise


# =============================================================================
# MODEL RETRAINING
# =============================================================================

@celery_app.task(
    bind=True,
    name="refresh_farmer_segments_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 2},
    soft_time_limit=120,
    time_limit=180,
)
def refresh_farmer_segments_task(self, force_full: bool = False):
    """
    Background incremental refresh of farmer segmentation clusters.
    """
    try:
        from ml.farmer_segmentation import get_segmentation

        segmentation = get_segmentation()

        import firebase_admin
        from firebase_admin import firestore

        db = None
        if firebase_admin._apps:
            db = firestore.client()
        else:
            try:
                firebase_admin.initialize_app()
                db = firestore.client()
            except Exception:
                logger.warning("Firebase not available for segment refresh")
                return {"status": "firebase_unavailable"}

        if db is None:
            return {"status": "firebase_unavailable"}

        result = segmentation.fit(db, force_full=force_full)

        return {
            "status": result.get("status"),
            "farmers_count": result.get("farmers_count"),
            "incremental": result.get("incremental", False),
            "duration_ms": result.get("duration_ms"),
            "refreshed_at": result.get("refreshed_at"),
        }

    except Exception:
        logger.exception("Farmer segment refresh task failed")
        raise


@celery_app.task(
    bind=True,
    name="retrain_yield_model_task",
    time_limit=1800,
    soft_time_limit=1500,
)
def retrain_yield_model_task(
    self,
    csv_path="Train.csv",
    model_output="yield_model.joblib",
):
    """
    Retrain and promote model safely.
    """

    history_path = Path("retraining_history.json")

    def _append_history(record):
        try:
            data = (
                json.loads(history_path.read_text())
                if history_path.exists()
                else {"runs": []}
            )

            data["runs"].append(record)

            data["runs"] = data["runs"][-100:]

            tmp_path = str(history_path) + ".tmp"

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            os.replace(tmp_path, str(history_path))

        except Exception:
            logger.exception("Failed writing retraining history")

    try:
        current_rmse = None

        if history_path.exists():
            try:
                data = json.loads(history_path.read_text())

                promoted = [
                    r for r in data.get("runs", [])
                    if r.get("outcome") == "promoted"
                ]

                if promoted:
                    current_rmse = promoted[-1].get("rmse")

            except Exception:
                logger.exception("Failed reading retraining history")

        self.update_state(
            state="PROGRESS",
            meta={"step": "training"},
        )

        from train_model import train_yield_model

        result = train_yield_model(
            csv_path=csv_path,
            model_output=model_output + ".candidate",
            baseline_output="feature_baseline.candidate.json",
        )

        candidate_rmse = result["rmse"]

        self.update_state(
            state="PROGRESS",
            meta={
                "step": "validating",
                "candidate_rmse": candidate_rmse,
            },
        )

        if current_rmse is None or candidate_rmse <= current_rmse:

            if os.path.exists(model_output):
                os.replace(model_output, model_output + ".prev")

            os.replace(
                model_output + ".candidate",
                model_output,
            )

            if os.path.exists("feature_baseline.json"):
                os.replace(
                    "feature_baseline.json",
                    "feature_baseline.prev.json",
                )

            if os.path.exists("feature_baseline.candidate.json"):
                os.replace(
                    "feature_baseline.candidate.json",
                    "feature_baseline.json",
                )

            outcome = "promoted"

        else:
            outcome = "rejected"

            for f in [
                model_output + ".candidate",
                "feature_baseline.candidate.json",
            ]:
                try:
                    os.remove(f)
                except OSError:
                    pass

        record = {
            "triggered_at": result["trained_at"],
            "completed_at": _dt.utcnow().isoformat(),
            "rmse": candidate_rmse,
            "previous_rmse": current_rmse,
            "outcome": outcome,
            "csv_path": csv_path,
        }

        _append_history(record)

        return {
            "outcome": outcome,
            "candidate_rmse": round(candidate_rmse, 4),
            "previous_rmse": (
                round(current_rmse, 4)
                if current_rmse is not None
                else None
            ),
            "promoted": outcome == "promoted",
        }

    except Exception as exc:
        logger.exception("Model retraining failed")

        _append_history({
            "triggered_at": _dt.utcnow().isoformat(),
            "completed_at": _dt.utcnow().isoformat(),
            "outcome": "failed",
            "error": str(exc),
        })

        return {
            "error": str(exc),
            "type": type(exc).__name__,
        }


@celery_app.task(
    bind=True,
    name="run_hyperparameter_optimization_task",
    time_limit=3600,
    soft_time_limit=3000,
)
def run_hyperparameter_optimization_task(
    self,
    csv_path="Train.csv",
    n_trials=50,
    cv_folds=5,
    study_name="yield_xgb_optimization",
):
    """
    Bayesian hyperparameter optimization for XGBoost yield model.
    Uses Optuna with k-fold cross-validation per trial.
    Tracks RMSE, MAE, R²; persists best params to hyperparameter_config.json.
    """
    try:
        import optuna
        import pandas as pd
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import xgboost as xgb
        import joblib
        import math
    except ImportError as exc:
        raise RuntimeError("Hyperopt requires: optuna, scikit-learn, xgboost, joblib") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    self.update_state(state="PROGRESS", meta={"step": "loading_data"})

    df = pd.read_csv(csv_path)
    df["SDate"] = pd.to_datetime(df["SDate"], errors="coerce")
    df = df.dropna(subset=["SDate"]).sort_values("SDate")

    _CAT_COLS = ["Crop", "CNext", "CLast", "CTransp", "IrriType", "IrriSource", "Season"]
    _DROP_COLS = ["FarmID", "category", "State", "District", "Sub-District",
                  "SDate", "HDate", "ExpYield", "geometry"]

    X = df.drop(columns=[c for c in _DROP_COLS if c in df.columns], errors="ignore")
    y = df["ExpYield"]

    X = pd.get_dummies(X, columns=[c for c in _CAT_COLS if c in X.columns], drop_first=True)

    def _objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBRegressor(
                **params,
                random_state=42,
                n_jobs=1,
            )
            model.fit(X_train_fold, y_train_fold)

            preds = model.predict(X_val_fold)
            rmse = math.sqrt(mean_squared_error(y_val_fold, preds))
            mae = mean_absolute_error(y_val_fold, preds)
            r2 = r2_score(y_val_fold, preds)

            fold_scores.append({
                "fold": fold_idx + 1,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            })

        mean_rmse = sum(f["rmse"] for f in fold_scores) / len(fold_scores)
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("mean_mae", sum(f["mae"] for f in fold_scores) / len(fold_scores))
        trial.set_user_attr("mean_r2", sum(f["r2"] for f in fold_scores) / len(fold_scores))
        return mean_rmse

    self.update_state(state="PROGRESS", meta={"step": "optimizing", "trials_total": n_trials})

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def _callback(study, trial):
        self.update_state(
            state="PROGRESS",
            meta={
                "step": "optimizing",
                "trials_completed": len(study.trials),
                "trials_total": n_trials,
                "best_rmse": study.best_value if study.best_trial else None,
                "best_params": study.best_params if study.best_trial else None,
            },
        )

    study.optimize(_objective, n_trials=n_trials, callbacks=[_callback], show_progress_bar=False)

    best_params = study.best_params
    best_trial = study.best_trial

    # Persist best config
    config_path = Path("hyperparameter_config.json")
    config_record = {
        "study_name": study_name,
        "n_trials": n_trials,
        "cv_folds": cv_folds,
        "best_params": best_params,
        "best_rmse": best_trial.value,
        "best_mean_mae": best_trial.user_attrs.get("mean_mae"),
        "best_mean_r2": best_trial.user_attrs.get("mean_r2"),
        "optimized_at": _dt.utcnow().isoformat(),
        "csv_path": csv_path,
    }
    tmp = config_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)
    os.replace(tmp, config_path)

    # Build trial history
    trial_history = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        trial_history.append({
            "trial_number": t.number,
            "params": t.params,
            "rmse": t.value,
            "mean_mae": t.user_attrs.get("mean_mae"),
            "mean_r2": t.user_attrs.get("mean_r2"),
            "fold_scores": t.user_attrs.get("fold_scores", []),
            "duration_ms": int(t.duration.total_seconds() * 1000) if t.duration else None,
        })

    # Benchmark vs production model
    self.update_state(state="PROGRESS", meta={"step": "benchmarking"})

    benchmark = {"production_model_exists": False, "improved": None, "production_rmse": None}

    if os.path.exists("yield_model.joblib"):
        from sklearn.metrics import mean_squared_error as _mse
        from sklearn.model_selection import train_test_split

        prod_model = joblib.load("yield_model.joblib")
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        prod_preds = prod_model.predict(X_test_b)
        prod_rmse = math.sqrt(_mse(y_test_b, prod_preds))

        opt_model = xgb.XGBRegressor(**best_params, random_state=42)
        opt_model.fit(X_train_b, y_train_b)
        opt_preds = opt_model.predict(X_test_b)
        opt_rmse = math.sqrt(_mse(y_test_b, opt_preds))

        benchmark = {
            "production_model_exists": True,
            "production_rmse": prod_rmse,
            "optimized_rmse": opt_rmse,
            "improved": opt_rmse < prod_rmse,
            "improvement_pct": ((prod_rmse - opt_rmse) / prod_rmse * 100) if prod_rmse > 0 else 0,
        }

    return {
        "study_name": study_name,
        "best_params": best_params,
        "best_rmse": best_trial.value,
        "best_mean_mae": best_trial.user_attrs.get("mean_mae"),
        "best_mean_r2": best_trial.user_attrs.get("mean_r2"),
        "n_trials": n_trials,
        "cv_folds": cv_folds,
        "optimized_at": config_record["optimized_at"],
        "trial_history": trial_history,
        "benchmark": benchmark,
        "config_path": str(config_path),
    }


@celery_app.task(
    bind=True,
    name="ensemble_health_task",
    soft_time_limit=10,
    time_limit=15,
)
def ensemble_health_task(self):
    """
    Async ensemble health check for monitoring.
    """
    try:
        stacker = _get_ensemble_stacker()

        health = stacker.health()

        return health

    except Exception:
        logger.exception("Ensemble health check failed")
        raise


@celery_app.task(
    bind=True,
    name="generate_regional_benchmark_task",
    time_limit=600,
    soft_time_limit=500,
)
def generate_regional_benchmark_task(self, farmer_uid: str, farmer_yield: float, region: str, crop_type: str):
    """
    Async generation of regional benchmark report with statistical analysis.
    """
    try:
        self.update_state(state="PROGRESS", meta={"step": "fetching_data"})

        import firebase_admin
        from firebase_admin import firestore

        db = None
        if firebase_admin._apps:
            db = firestore.client()
        else:
            try:
                firebase_admin.initialize_app()
                db = firestore.client()
            except Exception:
                logger.warning("Firebase not available for benchmark report")
                return {"status": "firebase_unavailable"}

        if db is None:
            return {"status": "firebase_unavailable"}

        from ml.regional_analytics import get_regional_analytics

        analytics = get_regional_analytics()

        self.update_state(state="PROGRESS", meta={"step": "computing_statistics"})

        report = analytics.generate_report(db, farmer_uid, farmer_yield, region, crop_type)

        self.update_state(state="PROGRESS", meta={"step": "persisting"})

        return {
            "status": "success",
            "report_id": report["report_id"] if report else None,
            "report": report,
        }

    except Exception:
        logger.exception("Regional benchmark report generation failed")
        raise


if __name__ == "__main__":
    celery_app.start()
