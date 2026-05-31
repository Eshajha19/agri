import logging
from typing import Any, Dict, Optional

from ml.registry import ModelRegistry


logger = logging.getLogger(__name__)


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
        Run prediction using resolved model.
        """

        if not isinstance(input_data, dict):
            raise ValueError("input_data must be dictionary")

        model_name = self._resolve_model_name(context)

        model = self._get_model(model_name)

        logger.info(
            "Running prediction using model='%s'",
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

    def available_models(self):
        """
        Return list of registered models.
        """

        try:
            return ModelRegistry.list_models()

        except Exception:
            logger.exception(
                "Failed listing models"
            )

            return []

    def health(self):
        """
        Simple router health snapshot.
        """

        models = self.available_models()

        return {
            "status": "healthy",
            "default_model": self.default_model,
            "registered_models": models,
            "total_models": len(models),
        }