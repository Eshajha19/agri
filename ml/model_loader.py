import joblib
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@lru_cache(maxsize=8)
def get_crop_recommendation_model():
    model_path = BASE_DIR / "models" / "crop_recommendation.pkl"
    return joblib.load(model_path)

@lru_cache(maxsize=8)
def get_fertilizer_model():
    model_path = BASE_DIR / "models" / "fertilizer.pkl"
    return joblib.load(model_path)