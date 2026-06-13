import logging
from typing import Optional, Dict, Any

from ml.registry import ModelRegistry


logger = logging.getLogger(__name__)

# Per-process governance instances — initialised by calling
# ``init_governance_router()`` from the FastAPI lifespan and from the
# Celery worker init so that every process that runs predictions also
# runs governance checks.
_drift_detector = None
_shadow_evaluator = None
_version_manager = None


def init_governance_router(drift_detector=None, shadow_evaluator=None, version_manager=None):
    """Inject governance instances into this process's ML router module.

    Must be called from every worker process that runs predictions
    (both the FastAPI lifespan and the Celery worker initialiser).
    """
    global _drift_detector, _shadow_evaluator, _version_manager
    _drift_detector = drift_detector
    _shadow_evaluator = shadow_evaluator
    _version_manager = version_manager
    logger.info(
        "Governance router initialised: drift=%s shadow=%s versions=%s",
        drift_detector is not None,
        shadow_evaluator is not None,
        version_manager is not None,
    )


class ModelRouter:
    """
    Central ML inference router.

    Responsibilities:
    - model selection
    - safe prediction execution
    - fallback handling
    - registry validation
    """

    def __init__(self, default_model: str = "xgboost"):
        self.default_model = default_model

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _resolve_model_name(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Resolve model name using request context.
        """

        if context is None:
            context = {}

        requested_model = context.get("model")

        if requested_model:
            try:
                ModelRegistry.get(requested_model)
                return requested_model

            except Exception:
                logger.warning(
                    "Requested model '%s' unavailable. Falling back.",
                    requested_model,
                )

        return self.default_model

    def _get_model(self, model_name: str):
        """
        Safely retrieve model from registry.
        """

        try:
            model = ModelRegistry.get(model_name)

        except Exception as exc:
            logger.exception(
                "Failed retrieving model '%s'",
                model_name,
            )

            raise RuntimeError(
                f"Model '{model_name}' unavailable"
            ) from exc

        if model is None:
            raise RuntimeError(
                f"Model '{model_name}' not found"
            )

        return model

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def predict(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Run single-sample prediction using resolved model.
        """

        if not isinstance(input_data, dict):
            raise ValueError("input_data must be dictionary")

        model_name = self._resolve_model_name(context)

        model = self._get_model(model_name)

        logger.info(
            "Running single prediction using model='%s'",
            model_name,
        )

        try:
            prediction = model.predict(input_data)

        except Exception as exc:
            logger.exception(
                "Prediction failed for model='%s'",
                model_name,
            )

            raise RuntimeError(
                f"Inference failed for model '{model_name}'"
            ) from exc

        return prediction

    def predict_batch(
        self,
        inputs: list[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> list[Any]:
        """
        Run batch prediction using resolved model.

        Falls back to individual predict() calls if the model adapter
        does not implement predict_batch().
        """
        if not inputs:
            return []

        model_name = self._resolve_model_name(context)
        model = self._get_model(model_name)

        # Governance: check for input drift against this model's baseline
        if _drift_detector is not None:
            try:
                _drift_detector.check_input_drift(active_name, input_data)
            except Exception as exc:
                logger.warning("Input drift check failed: %s", exc)

        logger.info(
            "Running batch prediction using model='%s', n=%d",
            model_name,
            len(inputs),
        )

        result = model.predict(processed_df)

        # Governance: check prediction drift after inference
        if _drift_detector is not None:
            try:
                _drift_detector.check_prediction_drift(active_name, float(result))
            except Exception as exc:
                logger.warning("Prediction drift check failed: %s", exc)

        return result
