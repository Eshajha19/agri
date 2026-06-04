import logging
import os

import joblib
import numpy as np
import pandas as pd

from ml.preprocessing import (
    MissingFeatureError,
    preprocess_prediction_input,
)


logger = logging.getLogger(__name__)


class XGBoostAdapter:
    """
    Safe XGBoost model adapter.
    """

    def __init__(self):
        self.model = None
        self.feature_columns = []

        self.numeric_columns = [
            "CropCoveredArea",
            "CHeight",
            "IrriCount",
            "WaterCov",
        ]

        self.categorical_vocab = {
            "Crop": [],
            "CNext": [],
            "CLast": [],
            "CTransp": [],
            "IrriType": [],
            "IrriSource": [],
            "Season": [],
        }

    # =========================================================================
    # LOAD
    # =========================================================================

    def load(self, model_path: str):
        """
        Load serialized model safely.
        """

        if not isinstance(model_path, str):
            raise ValueError(
                "model_path must be string"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        try:
            self.model = joblib.load(model_path)

        except Exception as exc:
            logger.exception(
                "Failed loading model"
            )

            raise RuntimeError(
                "Could not load model"
            ) from exc

        # Try extracting feature names
        try:
            if hasattr(self.model, "feature_names_in_"):
                self.feature_columns = list(
                    self.model.feature_names_in_
                )

        except Exception:
            logger.warning(
                "Could not extract feature names"
            )

        logger.info(
            "XGBoost model loaded successfully"
        )

    # =========================================================================
    # PREPROCESSING
    # =========================================================================

    def _prepare_dataframe(
        self,
        input_data: dict,
    ):
        """
        Prepare dataframe for inference.
        """

        required_columns = [
            "Crop",
            "CropCoveredArea",
            "CHeight",
            "CNext",
            "CLast",
            "CTransp",
            "IrriType",
            "IrriSource",
            "IrriCount",
            "WaterCov",
            "Season",
        ]

        dataframe = preprocess_prediction_input(
            input_data=input_data,
            required_columns=required_columns,
            numeric_columns=self.numeric_columns,
            categorical_vocab=self.categorical_vocab,
        )

        dataframe = pd.get_dummies(dataframe)

        # Align missing columns
        if self.feature_columns:

            missing = [
                col
                for col in self.feature_columns
                if col not in dataframe.columns
            ]

            for column in missing:
                dataframe[column] = 0

            dataframe = dataframe[
                self.feature_columns
            ]

        return dataframe

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict(self, input_data: dict):
        """
        Run single-sample prediction safely.
        """

        if self.model is None:
            raise RuntimeError("Model not loaded")

        if not isinstance(input_data, dict):
            raise ValueError("input_data must be a dictionary")

        if len(input_data) == 0:
            raise ValueError("input_data is empty — cannot run XGBoost inference on zero samples")
        try:
            dataframe = self._prepare_dataframe(
                input_data
            )

            prediction = self.model.predict(
                dataframe
            )

            if prediction is None:
                raise RuntimeError(
                    "Empty prediction returned"
                )

            value = float(prediction[0])

            if not np.isfinite(value):
                raise RuntimeError(
                    "Invalid prediction value"
                )

            logger.info(
                "Prediction completed successfully"
            )

            return value

        except MissingFeatureError:
            raise

        except Exception as exc:
            logger.exception(
                "Prediction failed"
            )

            raise RuntimeError(
                "Model inference failed"
            ) from exc

    def predict_batch(self, inputs: list[dict]) -> list[float]:
        """
        Run batch prediction on multiple samples in a single DataFrame.

        Returns a list of prediction floats aligned with input order.
        Empty list returns empty list. Individual failures log warning
        and return None at that index so callers can degrade gracefully.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if not inputs:
            return []

        dataframes = []
        valid_indices = []

        for idx, input_data in enumerate(inputs):
            if not isinstance(input_data, dict) or len(input_data) == 0:
                logger.warning("Batch item %d skipped: invalid or empty input", idx)
                continue
            try:
                df = self._prepare_dataframe(input_data)
                dataframes.append(df)
                valid_indices.append(idx)
            except MissingFeatureError:
                logger.warning("Batch item %d skipped: missing features", idx)
            except Exception as exc:
                logger.warning("Batch item %d skipped: preprocessing failed (%s)", idx, exc)

        if not dataframes:
            return [None] * len(inputs)

        try:
            combined = pd.concat(dataframes, ignore_index=True)
            raw_preds = self.model.predict(combined)

            results = [None] * len(inputs)
            for vi, pred in zip(valid_indices, raw_preds):
                value = float(pred)
                if np.isfinite(value):
                    results[vi] = value
                else:
                    logger.warning("Batch item %d returned non-finite prediction", vi)

            logger.info("Batch prediction completed: %d/%d successful", len(valid_indices), len(inputs))
            return results

        except Exception as exc:
            logger.exception("Batch prediction failed")
            raise RuntimeError("Batch model inference failed") from exc

    # =========================================================================
    # HEALTH
    # =========================================================================

    def health(self):
        """
        Adapter health snapshot.
        """

        return {
            "loaded": self.model is not None,
            "feature_count": len(
                self.feature_columns
            ),
        }