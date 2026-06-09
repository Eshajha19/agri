import pandas as pd
import numpy as np
import joblib
from ml.base import YieldModel

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class LSTMAdapter(YieldModel):
    """
    Adapter for LSTM yield prediction model.
    """

    def __init__(self, time_steps: int = 1, feature_names: list[str] | None = None):
        self.model = None
        self.scaler = None
        self.time_steps = time_steps
        self._feature_names = feature_names or []

    def load(self, model_path: str):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTMAdapter")

        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"LSTM model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            raise
        
        # Load scaler if path provided
        if self.scaler_path:
            try:
                self.scaler = joblib.load(self.scaler_path)
                print(f"Scaler loaded from {self.scaler_path}")
            except Exception as e:
                print(f"Warning: Could not load scaler from {self.scaler_path}: {e}")

    def predict(self, input_data: pd.DataFrame) -> float:
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        # Capture feature names from first call if not already set
        if not self._feature_names:
            self._feature_names = list(input_data.columns)
        else:
            missing = [c for c in self._feature_names if c not in input_data.columns]
            if missing:
                raise ValueError(
                    f"LSTMAdapter.predict() received a DataFrame that is "
                    f"missing {len(missing)} expected column(s): {missing}. "
                    "Ensure FeaturePreprocessor.preprocess() is called first."
                )
            input_data = input_data[self._feature_names]

        # LSTM models require 3D input: (samples, time_steps, features_per_step)
        # We use the stored time_steps metadata to preserve temporal structure.
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data must be a pandas DataFrame")
    data_array = input_data.values

    if len(input_data) == 0:
            raise ValueError("input_data is empty — cannot run LSTM inference on zero samples")
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
        
        # Inverse-transform to original yield unit
        if self.scaler is not None:
            pred_value = float(self.scaler.inverse_transform([[pred_value]])[0][0])
        
        return pred_value

    @property
    def model_type(self) -> str:
        return "LSTM"

    @property
    def feature_names(self):
        return self._feature_names
