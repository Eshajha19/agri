import json
import logging
import os
import threading
import joblib
import numpy as np
from celery import Celery
from ml.security import verify_and_load_joblib

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
_model_lock = threading.Lock()

def _get_lag_model():
    global _model_lag

    if _model_lag is None:
        with _model_lock:
            if _model_lag is None:
                try:
                    _model_lag = joblib.load("sklearn_yield_model.joblib")
                except Exception as e:
                    print(f"Failed to load lag model: {e}")
        try:
            _model_lag = verify_and_load_joblib("sklearn_yield_model.joblib")
        except Exception as e:
            import logging
            logging.error(f"Celery worker error: {e}")
            logger.exception("Failed to load lag model")
            raise

    return _model_lag


def _get_trend_model():
    global _model_trend

    if _model_trend is None:
        with _model_lock:
            if _model_trend is None:
                try:
                    if os.path.exists("trend_forecast_model.joblib"):
                        _model_trend = joblib.load("trend_forecast_model.joblib")
                except Exception as e:
                    print(f"Failed to load trend model: {e}")
        try:
            if os.path.exists("trend_forecast_model.joblib"):
                _model_trend = verify_and_load_joblib("trend_forecast_model.joblib")
        except Exception as e:
            print(f"Failed to load trend model: {e}")
    return _model_trend


def _get_ml_router():
    global _ml_router

    if _ml_router is None:
        with _model_lock:
            if _ml_router is None:
                try:
                    from ml.router import ModelRouter
                    from ml.registry import ModelRegistry
                    from ml.adapters.xgboost_adapter import XGBoostAdapter
                    
                    xgb_adapter = XGBoostAdapter()
                    if os.path.exists("yield_model.joblib"):
                        xgb_adapter.load("yield_model.joblib")
                        ModelRegistry.register("xgboost", xgb_adapter)
                    
                    _ml_router = ModelRouter(default_model="xgboost")
                except Exception as e:
                    print(f"Failed to initialize ML router: {e}")
    return _ml_router


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
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}

@celery_app.task(bind=True, name="predict_ensemble_task")
def predict_ensemble_task(self, data: list):
    """Celery task for yield prediction using ensemble of models."""
    lag_model = _get_lag_model()
    trend_model = _get_trend_model()
    
    if not lag_model and not trend_model:
        raise RuntimeError("No models loaded in worker")
    
    try:
        if len(data) != 5:
            raise ValueError("Exactly 5 values are required")
        
        predictions = []
        
        if lag_model:
            data_arr = np.array(data).reshape(1, -1)
            lag_pred = lag_model.predict(data_arr)[0]
            predictions.append(float(lag_pred))
        
        if trend_model:
            temp = list(data)
            for _ in range(5):
                features = temp[:5]
                trend_pred = trend_model.predict([features])[0]
                predictions.append(float(trend_pred))
                temp = temp[1:] + [trend_pred]
        
        if not predictions:
            raise RuntimeError("No predictions generated")
        
        ensemble_pred = sum(predictions) / len(predictions)
        
        return {
            "prediction": round(ensemble_pred, 2),
            "model": "Ensemble (Lag + Trend)",
            "individual_predictions": [round(p, 2) for p in predictions],
        }
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}

if __name__ == "__main__":
    celery_app.start()

    except Exception:
        logger.exception("Trend prediction task failed")
        raise


@celery_app.task(bind=True, name="process_whatsapp_webhook_task")
def process_whatsapp_webhook_task(self, body: str, sender_number: str):
    """Celery task for processing incoming WhatsApp messages asynchronously."""
    from webhook_validator import validate_and_parse, WebhookValidationError
    from whatsapp_service import process_webhook_message

    try:
        message = validate_and_parse(body, sender_number)
    except WebhookValidationError as exc:
        logger.warning("Discarding invalid webhook payload from %r: %s", sender_number, exc)
        return {"status": "discarded", "reason": str(exc)}

    result = process_webhook_message(message.text or body, message.sender_number)
    return {"status": "processed", "sender": message.sender_number, "result": result}

if __name__ == "__main__":
    celery_app.start()
