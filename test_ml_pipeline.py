import sys
import os
import pandas as pd

# Add the parent directory to sys.path to allow importing from agri.ml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.registry import ModelRegistry
from ml.adapters.xgboost_adapter import XGBoostAdapter
from ml.router import ModelRouter

def test_pipeline():
    print("--- Starting ML Pipeline Test ---")
    
    # 1. Initialize Adapters
    xgb_adapter = XGBoostAdapter()
    
    # Load the existing model
    model_path = os.path.join(os.path.dirname(__file__), "yield_model.joblib")
    if os.path.exists(model_path):
        xgb_adapter.load(model_path)
    else:
        print(f"Warning: {model_path} not found. Test might fail on prediction.")
        return

    # 2. Register Models
    ModelRegistry.register("xgboost", xgb_adapter)
    
    # 3. Test Registry
    models = ModelRegistry.list_models()
    print(f"Registered Models: {models}")
    assert "xgboost" in models

    # 4. Test Router
    router = ModelRouter(default_model="xgboost")
    
    # Mock Input Data
    sample_input = {
        "Crop": "rice",
        "CropCoveredArea": 10.5,
        "CHeight": 50,
        "CNext": "wheat",
        "CLast": "maize",
        "CTransp": "none",
        "IrriType": "drip",
        "IrriSource": "well",
        "IrriCount": 3,
        "WaterCov": 80,
        "Season": "kharif"
    }
    
    # Test routing logic
    context_punjab = {"location": "Punjab"}
    context_karnataka = {"location": "Karnataka", "crop": "rice"}
    
    print(f"Routing for Punjab: {router.route(context_punjab)}")
    print(f"Routing for Karnataka/Rice: {router.route(context_karnataka)}")
    
    # 5. Test Prediction
    try:
        prediction = router.predict(sample_input, context_punjab)
        print(f"Prediction Result: {prediction}")
        assert isinstance(prediction, float)
        print("SUCCESS: Pipeline Test Successful!")
    except Exception as e:
        print(f"FAILURE: Prediction failed: {e}")

if __name__ == "__main__":
    test_pipeline()
