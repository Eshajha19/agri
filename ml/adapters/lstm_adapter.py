import os
import logging
import pandas as pd
import numpy as np
import joblib
from ml.base import YieldModel

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available - LSTMAdapter will not work")

class LSTMAdapter(YieldModel):
    """
    Adapter for LSTM yield prediction model.
    
    Loads both model and scaler for production deployment.
    Handles path resolution similar to lstm_yield_model.py for consistency.
    """

    def __init__(self, time_steps: int = 1, feature_names: list[str] | None = None, scaler_path: str | None = None):
        self.model = None
        self.scaler = None
        self.time_steps = time_steps
        self._feature_names = feature_names or []
        self.scaler_path = scaler_path

    def load(self, model_path: str, scaler_path: str | None = None):
        """Load LSTM model and optional scaler with production path resolution."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTMAdapter")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM model not found at {model_path}")

        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"LSTM model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading LSTM model from {model_path}: {e}")
            raise
        
        # Load scaler if path provided or stored
        scaler_to_load = scaler_path or self.scaler_path
        if scaler_to_load and os.path.exists(scaler_to_load):
            try:
                self.scaler = joblib.load(scaler_to_load)
                logger.info(f"Scaler loaded from {scaler_to_load}")
            except Exception as e:
                logger.warning(f"Could not load scaler from {scaler_to_load}: {e}")
                logger.warning("Predictions will use unscaled model output")
        elif scaler_to_load:
            logger.warning(f"Scaler path specified but file not found: {scaler_to_load}")
        else:
            logger.debug("No scaler path provided for LSTMAdapter")

    def predict(self, input_data: pd.DataFrame) -> float:
        """Run LSTM inference with proper shape validation."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("input_data must be a pandas DataFrame")

        if len(input_data) == 0:
            raise ValueError("input_data is empty — cannot run LSTM inference on zero samples")

        # Capture feature names from first call if not already set
        if not self._feature_names:
            self._feature_names = list(input_data.columns)
        else:
            missing = [c for c in self._feature_names if c not in input_data.columns]
            if missing:
                raise ValueError(
                    f"LSTMAdapter.predict() received a DataFrame missing "
                    f"{len(missing)} expected column(s): {missing}. "
                    "Ensure FeaturePreprocessor.preprocess() is called first."
                )
            input_data = input_data[self._feature_names]

        # LSTM models require 3D input: (samples, time_steps, features_per_step)
        # Use stored time_steps metadata to preserve temporal structure
        data_array = input_data.values
        num_samples = data_array.shape[0]
        total_features = data_array.shape[1]

        if total_features % self.time_steps != 0:
            raise ValueError(
                f"Total features ({total_features}) must be divisible by "
                f"time_steps ({self.time_steps}) to preserve temporal structure."
            )

        features_per_step = total_features // self.time_steps
        reshaped_data = data_array.reshape((num_samples, self.time_steps, features_per_step))

        prediction = self.model.predict(reshaped_data)
        pred_value = float(prediction[0][0])
        
        # Inverse-transform to original yield unit if scaler available
        if self.scaler is not None:
            try:
                pred_value = float(self.scaler.inverse_transform([[pred_value]])[0][0])
            except Exception as e:
                logger.warning(f"Failed to inverse-transform prediction: {e}. Using raw value.")
        
        return pred_value

    @property
    def model_type(self) -> str:
        return "LSTM"

    @property
    def feature_names(self):
        return self._feature_names
