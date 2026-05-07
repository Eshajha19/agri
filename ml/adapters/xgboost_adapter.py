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

        # The FeaturePreprocessor is responsible for validating and aligning
        # columns before this point.  We do a final shape check here as a
        # safety net, but we do NOT silently fill missing columns with 0 —
        # that would mask corrupt inputs and produce meaningless predictions.
        if self._feature_names:
            missing = [c for c in self._feature_names if c not in input_data.columns]
            if missing:
                raise ValueError(
                    f"XGBoostAdapter.predict() received a DataFrame that is "
                    f"missing {len(missing)} expected column(s): {missing}. "
                    "Ensure FeaturePreprocessor.preprocess() is called first."
                )
            input_data = input_data[self._feature_names]

        prediction = self.model.predict(input_data)
        return float(prediction[0])

    @property
    def model_type(self) -> str:
        return "XGBoost"
    
    @property
    def feature_names(self):
        return self._feature_names
