import logging
from typing import Optional, Dict, Any, Tuple

from ml.registry import ModelRegistry
from ml.preprocessing import FeaturePreprocessor

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Routes prediction requests to the appropriate registered model.

    The router selects a model based on context (location, crop), then
    delegates preprocessing and prediction to that model's adapter.

    If the selected model is unavailable the router falls back to the
    default model and logs a warning — it does NOT silently switch to a
    model with a different feature schema without telling the caller.
    """

    def __init__(self, default_model: str = "xgboost"):
        self.default_model = default_model
        self.preprocessor = FeaturePreprocessor()

    def route(self, context: Dict[str, Any]) -> str:
        """Return the name of the model to use for this request."""
        location = context.get("location", "").lower()
        crop = context.get("crop", "").lower()

        if "punjab" in location or "haryana" in location:
            return "xgboost"
        elif "karnataka" in location and crop == "rice":
            return "lstm"

        return self.default_model

    def predict(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> float:
        """
        Preprocess input and run prediction through the selected model.

        Parameters
        ----------
        input_data : dict
            Raw feature dictionary from the API request.
        context : dict, optional
            Routing hints (e.g. ``location``, ``crop``).

        Returns
        -------
        float
            Predicted yield value.

        Raises
        ------
        ml.preprocessing.UnknownCategoryError
            If a categorical input value was not seen during training.
        ml.preprocessing.MissingFeatureError
            If required feature columns are absent after encoding.
        ValueError
            If no model is available (neither selected nor default).
        """
        if context is None:
            context = {}

        selected_name = self.route(context)
        model = ModelRegistry.get_model(selected_name)
        active_name = selected_name

        if model is None and selected_name != self.default_model:
            logger.warning(
                "Model '%s' is not registered. Falling back to default model '%s'. "
                "Note: the default model may have been trained on a different feature "
                "schema — ensure inputs are compatible.",
                selected_name,
                self.default_model,
            )
            model = ModelRegistry.get_model(self.default_model)
            active_name = self.default_model

        if model is None:
            raise ValueError(
                f"No model available: '{selected_name}' was selected but is not "
                f"registered, and the default model '{self.default_model}' is also "
                "not registered. Check that init_ml_pipeline() completed successfully."
            )

        # Bind the preprocessor to this model's expected feature schema.
        # Each model adapter exposes the exact column list it was trained on.
        if hasattr(model, "feature_names") and model.feature_names:
            self.preprocessor.feature_cols = model.feature_names
        else:
            logger.warning(
                "Model '%s' does not expose feature_names. "
                "Preprocessing will not validate column alignment.",
                active_name,
            )

        # Raises UnknownCategoryError or MissingFeatureError on bad input —
        # never silently fills missing columns with 0.
        processed_df = self.preprocessor.preprocess(input_data)

        logger.info(
            "[ML Router] Routing to model: %s (%s)", active_name, model.model_type
        )

        return model.predict(processed_df)
