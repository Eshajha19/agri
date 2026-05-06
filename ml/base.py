from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class YieldModel(ABC):
    """
    Abstract base class for all yield prediction models.
    Ensures a standardized interface for prediction.
    """
    
    @abstractmethod
    def load(self, model_path: str):
        """Load the model from the specified path."""
        pass
    
    @abstractmethod
    def predict(self, input_data: pd.DataFrame) -> float:
        """
        Make a prediction based on the input data.
        Returns a float representing the expected yield.
        """
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the type of the model (e.g., 'XGBoost', 'LSTM')."""
        pass
