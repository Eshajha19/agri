import os
import logging
import joblib
import numpy as np
from celery import Celery

logger = logging.getLogger(__name__)

# Initialize Celery app — Redis authentication is required.
# Set REDIS_URL for a full connection string, or REDIS_PASSWORD to use
# default redis://:{password}@localhost:6379/0.  Set ALLOW_INSECURE_REDIS=1
# (NOT RECOMMENDED) to allow an unauthenticated redis://localhost:6379/0
# fallback for local development only.
redis_url = os.getenv("REDIS_URL")
redis_password = os.getenv("REDIS_PASSWORD")
allow_insecure = os.getenv("ALLOW_INSECURE_REDIS", "").lower() in ("1", "true", "yes")

if not redis_url:
    if redis_password:
        redis_url = f"redis://:{redis_password}@localhost:6379/0"
        logger.info("Celery: using Redis with password authentication")
    elif allow_insecure:
        redis_url = "redis://localhost:6379/0"
        logger.warning(
            "CELERY INSECURE REDIS: ALLOW_INSECURE_REDIS is set — connecting "
            "without authentication. This is DANGEROUS if Redis is exposed to "
            "the network."
        )
    else:
        logger.critical(
            "CELERY REDIS AUTH REQUIRED: Neither REDIS_URL nor REDIS_PASSWORD "
            "is set. Set REDIS_PASSWORD for a password-authenticated connection "
            "to localhost:6379/0, or set REDIS_URL for a full connection string. "
            "To allow an unauthenticated connection (NOT RECOMMENDED), set "
            "ALLOW_INSECURE_REDIS=1."
        )
        raise ValueError("REDIS_URL or REDIS_PASSWORD must be set")

celery_app = Celery(
    "agri_ml_tasks",
    broker=redis_url,
    backend=redis_url
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
        except Exception as e:
            print(f"Failed to load lag model: {e}")
    return _model_lag

def _get_trend_model():
    global _model_trend
    if _model_trend is None:
        try:
            if os.path.exists("trend_forecast_model.joblib"):
                _model_trend = joblib.load("trend_forecast_model.joblib")
        except Exception as e:
            print(f"Failed to load trend model: {e}")
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
        except Exception as e:
            print(f"Failed to initialize ML router: {e}")
    return _ml_router

@celery_app.task(bind=True, name="predict_yield_task")
def predict_yield_task(self, input_data: dict, context: dict):
    """Celery task for yield prediction using ML router."""
    router = _get_ml_router()
    if not router:
        raise RuntimeError("ML Router not initialized in worker")
    
    try:
        predicted_yield = router.predict(input_data, context)
        return {"predicted_ExpYield": float(predicted_yield)}
    except Exception as e:
        # Wrap exception info to be serializable
        return {"error": str(e), "type": type(e).__name__}

@celery_app.task(bind=True, name="predict_yield_lag_task")
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
            "model": "RandomForest Time Series (Lag Features)"
        }
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}

@celery_app.task(bind=True, name="predict_yield_trend_task")
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
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}

if __name__ == "__main__":
    celery_app.start()

@celery_app.task(bind=True, name="process_whatsapp_webhook_task")
def process_whatsapp_webhook_task(self, body: str, sender_number: str):
    """Celery task for processing incoming WhatsApp messages asynchronously."""
    from whatsapp_service import process_webhook_message
    
    result = process_webhook_message(body, sender_number)
    return {"status": "processed", "sender": sender_number, "result": result}
