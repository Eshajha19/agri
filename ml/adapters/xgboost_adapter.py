import joblib
import pandas as pd
from ml.base import YieldModel

class XGBoostAdapter(YieldModel):
    """
    Adapter for XGBoost yield prediction model.
    """
    
    def __init__(self):
        self.model = None
        self._feature_names = []

    def load(self, model_path: str):
        try:
            self.model = joblib.load(model_path)
            # Try to extract feature names if it's an XGBoost model
            if hasattr(self.model, 'get_booster'):
                self._feature_names = list(self.model.get_booster().feature_names)
            elif hasattr(self.model, 'feature_names_in_'):
                self._feature_names = list(self.model.feature_names_in_)
            print(f"XGBoost model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading XGBoost model: {e}")
            raise

    def predict(self, input_data: pd.DataFrame) -> float:
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Ensure columns match what the model expects
        if self._feature_names:
            for col in self._feature_names:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[self._feature_names]
            
        prediction = self.model.predict(input_data)
        return float(prediction[0])

    @property
    def model_type(self) -> str:
        return "XGBoost"
    
    @property
    def feature_names(self):
        return self._feature_names
