import json
import logging
import os
from datetime import datetime as _dt
from pathlib import Path

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
_model_lag_lock = threading.Lock()
_model_trend_lock = threading.Lock()
_ml_router_lock = threading.Lock()


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


if __name__ == "__main__":
    celery_app.start()