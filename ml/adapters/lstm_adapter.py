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
    
    def __init__(self):
        self.model = None

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
        
        # LSTM models often require 3D input: (samples, time_steps, features)
        # For simplicity, we assume single-step prediction or handles reshaping here
        # This is a placeholder for the actual reshaping logic needed for LSTM
        
        # Example: Reshape to (1, 1, num_features)
        data_array = input_data.values
        reshaped_data = data_array.reshape((data_array.shape[0], 1, data_array.shape[1]))
        
        prediction = self.model.predict(reshaped_data)
        return float(prediction[0][0])

    @property
    def model_type(self) -> str:
        return "LSTM"
