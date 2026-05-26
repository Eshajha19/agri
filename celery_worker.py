import os
import joblib
import numpy as np
from celery import Celery
import logging

logger = logging.getLogger(__name__)

# Initialize Celery app
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
)

# Global model variables for the worker
_model_lag = None
_model_trend = None
_ml_router = None


def _get_lag_model():
    global _model_lag
    if _model_lag is None:
        try:
            _model_lag = joblib.load("sklearn_yield_model.joblib")
        except Exception:
            logger.exception("Failed to load lag model")
    return _model_lag

def _get_trend_model():
    global _model_trend
    if _model_trend is None:
        try:
            if os.path.exists("trend_forecast_model.joblib"):
                _model_trend = joblib.load("trend_forecast_model.joblib")
        except Exception:
            logger.exception("Failed to load trend model")
    return _model_trend

def _get_ml_router():
    global _ml_router
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
        except Exception:
            logger.exception("Failed to initialize ML router")
    return _ml_router

@celery_app.task(
    bind=True,
    name="predict_yield_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def predict_yield_task(self, input_data: dict, context: dict):
    """Celery task for yield prediction using ML router."""
    router = _get_ml_router()
    if not router:
        raise RuntimeError("ML Router not initialized in worker")
    try:
        predicted_yield = router.predict(input_data, context)
        return {"predicted_ExpYield": float(predicted_yield)}
    except Exception:
        logger.exception("Yield prediction task failed")
        raise

@celery_app.task(
    bind=True,
    name="predict_yield_lag_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def predict_yield_lag_task(self, data: list):
    """Celery task for yield prediction using time-series lag model."""
    model = _get_lag_model()
    if not model:
        raise RuntimeError("Lag model not loaded in worker")

    try:
        if len(data) != 5:
            raise ValueError("Exactly 5 values are required")
        data_arr = np.array(data).reshape(1, -1)
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
    retry_kwargs={"max_retries": 3},
)
def predict_yield_trend_task(self, data: list):
    """Celery task for yield trend forecasting."""
    model = _get_trend_model()  # Use dedicated trend model, not the lag model
    if not model:
        raise RuntimeError("Trend model not loaded in worker")

    try:
        if len(data) != 5:
            raise ValueError("Exactly 5 values are required")
        temp = list(data)
        trend = []
        for _ in range(5):
            features = temp[:5]
            pred = model.predict([features])[0]
            pred_value = round(float(pred), 2)
            trend.append(pred_value)
            temp = temp[1:] + [pred_value]

        return {
            "trend": trend,
            "prediction": trend[-1],
            "model": "RandomForest Trend Forecast (Lag Features)"
        }
    except Exception:
        logger.exception("Trend prediction task failed")
        raise

if __name__ == "__main__":
    celery_app.start()

@celery_app.task(
    bind=True,
    name="process_whatsapp_webhook_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def process_whatsapp_webhook_task(self, body: str, sender_number: str):
    """Celery task for processing incoming WhatsApp messages asynchronously."""
    from whatsapp_service import process_webhook_message

    result = process_webhook_message(body, sender_number)
    return {"status": "processed", "sender": sender_number, "result": result}
