import os
from ml.security import verify_and_load_joblib
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
            _model_lag = verify_and_load_joblib("sklearn_yield_model.joblib")
        except Exception:
            logger.exception("Failed to load lag model")
    return _model_lag

def _get_trend_model():
    global _model_trend
    if _model_trend is None:
        try:
            if os.path.exists("trend_forecast_model.joblib"):
                _model_trend = verify_and_load_joblib("trend_forecast_model.joblib")
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

@celery_app.task(bind=True, name="retrain_yield_model_task", time_limit=1800, soft_time_limit=1500)
def retrain_yield_model_task(self, csv_path="Train.csv", model_output="yield_model.joblib"):
    """
    Retrain the XGBoost yield model, compare RMSE against last promoted run,
    promote if improved (lower RMSE), discard otherwise.
    Writes every run to retraining_history.json.
    """
    import json
    from datetime import datetime as _dt
    from pathlib import Path

    history_path = Path("retraining_history.json")

    def _append_history(record):
        try:
            data = json.loads(history_path.read_text()) if history_path.exists() else {"runs": []}
            data["runs"].append(record)
            data["runs"] = data["runs"][-100:]
            tmp = str(history_path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, str(history_path))
        except Exception:
            pass

    try:
        # Read last known RMSE from history
        current_rmse = None
        if history_path.exists():
            try:
                data = json.loads(history_path.read_text())
                promoted = [r for r in data.get("runs", []) if r.get("outcome") == "promoted"]
                if promoted:
                    current_rmse = promoted[-1].get("rmse")
            except Exception:
                pass

        self.update_state(state="PROGRESS", meta={"step": "training"})

        from train_model import train_yield_model
        result = train_yield_model(
            csv_path=csv_path,
            model_output=model_output + ".candidate",
            baseline_output="feature_baseline.candidate.json",
        )
        candidate_rmse = result["rmse"]

        self.update_state(state="PROGRESS", meta={"step": "validating", "candidate_rmse": candidate_rmse})

        # Promote if no previous run or candidate is equal/better
        if current_rmse is None or candidate_rmse <= current_rmse:
            # Atomic model promotion
            if os.path.exists(model_output):
                os.replace(model_output, model_output + ".prev")
            os.replace(model_output + ".candidate", model_output)

            # Atomic baseline promotion
            if os.path.exists("feature_baseline.json"):
                os.replace("feature_baseline.json", "feature_baseline.prev.json")
            if os.path.exists("feature_baseline.candidate.json"):
                os.replace("feature_baseline.candidate.json", "feature_baseline.json")

            # Hot-reload into ML registry so live requests use new model immediately
            try:
                from ml.adapters.xgboost_adapter import XGBoostAdapter
                from ml.registry import ModelRegistry
                xgb_adapter = XGBoostAdapter()
                xgb_adapter.load(model_output)
                ModelRegistry.register("xgboost", xgb_adapter)
            except Exception:
                pass  # Non-fatal — next cold start will reload

            outcome = "promoted"
        else:
            # Candidate is worse — discard cleanly
            for f in [model_output + ".candidate", "feature_baseline.candidate.json"]:
                try:
                    os.remove(f)
                except OSError:
                    pass
            outcome = "rejected"

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
            "previous_rmse": round(current_rmse, 4) if current_rmse is not None else None,
            "promoted": outcome == "promoted",
        }

    except Exception as exc:
        _append_history({
            "triggered_at": _dt.utcnow().isoformat(),
            "completed_at": _dt.utcnow().isoformat(),
            "outcome": "failed",
            "error": str(exc),
        })
        return {"error": str(exc), "type": type(exc).__name__}


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
