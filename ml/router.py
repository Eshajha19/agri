from typing import Optional, Dict, Any
from ml.registry import ModelRegistry
from ml.preprocessing import FeaturePreprocessor

class ModelRouter:
    """
    Handles dynamic routing to the appropriate model based on context.
    """
    
    def __init__(self, default_model: str = "xgboost"):
        self.default_model = default_model
        self.preprocessor = FeaturePreprocessor()

    def route(self, context: Dict[str, Any]) -> str:
        """
        Logic to decide which model to use.
        - Location-based selection
        - Data availability based selection
        - Random A/B testing
        """
        location = context.get("location", "").lower()
        crop = context.get("crop", "").lower()
        
        # Example dynamic selection logic
        if "punjab" in location or "haryana" in location:
            return "xgboost"  # XGBoost might be better for these regions
        elif "karnataka" in location and crop == "rice":
            return "lstm"     # LSTM might be better for specific crops in specific regions
        
        return self.default_model

    def predict(self, input_data: Dict[str, Any], context: Dict[str, Any] = None) -> float:
        """
        The main entry point for predictions.
        """
        if context is None:
            context = {}
            
        model_name = self.route(context)
        model = ModelRegistry.get_model(model_name)
        
        if not model:
            # Fallback to default if selected model is not available
            model = ModelRegistry.get_model(self.default_model)
            
        if not model:
            raise ValueError(f"No model found for routing: {model_name} or default: {self.default_model}")

        # Standardize features using preprocessor
        # If model adapter provides feature names, use them for preprocessing
        if hasattr(model, 'feature_names'):
            self.preprocessor.feature_cols = model.feature_names
            
        processed_df = self.preprocessor.preprocess(input_data)
        
        # Log which model was used (Monitoring)
        print(f"[ML Router] Routing to model: {model_name} ({model.model_type})")
        
        return model.predict(processed_df)
