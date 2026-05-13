import pandas as pd
import numpy as np
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
    
    def __init__(self, time_steps: int = 1):
        self.model = None
        self.time_steps = time_steps

    def load(self, model_path: str):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTMAdapter")
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"LSTM model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            raise

    def predict(self, input_data: pd.DataFrame) -> float:
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # LSTM models require 3D input: (samples, time_steps, features_per_step)
        # We use the stored time_steps metadata to preserve temporal structure.
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
        return float(prediction[0][0])

    @property
    def model_type(self) -> str:
        return "LSTM"
