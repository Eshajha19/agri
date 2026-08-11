import os
import logging
import joblib
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Get base directory - either from environment or file location
_BASE_DIR = Path(os.getenv("ML_MODELS_DIR", "")).resolve() if os.getenv("ML_MODELS_DIR") else Path(__file__).resolve().parent.parent
_MODELS_DIR = _BASE_DIR / "models"

def _resolve_model_path(model_name: str, fallback_locations: list[str] | None = None) -> Path:
    """
    Resolve model path with support for production deployment.
    
    Priority order:
    1. Environment variable: ML_MODELS_DIR/<model_name>
    2. Production location: /app/models/<model_name>
    3. Repo location: <repo_root>/models/<model_name>
    4. Provided fallback locations
    5. Current directory
    """
    # Environment override
    if models_dir := os.getenv("ML_MODELS_DIR"):
        env_path = Path(models_dir) / model_name
        if env_path.exists():
            logger.info(f"Found {model_name} at ML_MODELS_DIR: {env_path}")
            return env_path
        logger.debug(f"ML_MODELS_DIR set but {model_name} not found at: {env_path}")
    
    # Production location
    prod_path = Path("/app/models") / model_name
    if prod_path.exists():
        logger.info(f"Found {model_name} at production location: {prod_path}")
        return prod_path
    
    # Repo models directory
    repo_path = _MODELS_DIR / model_name
    if repo_path.exists():
        logger.info(f"Found {model_name} at repo models dir: {repo_path}")
        return repo_path
    
    # Fallback locations
    if fallback_locations:
        for fallback in fallback_locations:
            fb_path = Path(fallback)
            if fb_path.exists():
                logger.info(f"Found {model_name} at fallback location: {fb_path}")
                return fb_path
            logger.debug(f"Fallback location not found: {fb_path}")
    
    # Current directory
    cwd_path = Path.cwd() / model_name
    logger.warning(f"Falling back to current directory for {model_name}: {cwd_path}")
    return cwd_path


@lru_cache(maxsize=8)
def get_crop_recommendation_model():
    """Load crop recommendation model with production path resolution."""
    model_name = "crop_recommendation.pkl"
    try:
        model_path = _resolve_model_path(model_name)
        if not model_path.exists():
            raise FileNotFoundError(f"Crop recommendation model not found at {model_path}")
        logger.info(f"Loading crop recommendation model from {model_path}")
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load crop recommendation model: {e}")
        raise


@lru_cache(maxsize=8)
def get_fertilizer_model():
    """Load fertilizer recommendation model with production path resolution."""
    model_name = "fertilizer.pkl"
    try:
        model_path = _resolve_model_path(model_name)
        if not model_path.exists():
            raise FileNotFoundError(f"Fertilizer model not found at {model_path}")
        logger.info(f"Loading fertilizer model from {model_path}")
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load fertilizer model: {e}")
        raise


def get_model_by_name(model_name: str, fallback_locations: list[str] | None = None):
    """
    Generic model loader with caching and production path resolution.
    
    Args:
        model_name: Name of the model file (e.g., "my_model.joblib")
        fallback_locations: List of alternative paths to check
    
    Returns:
        Loaded model object
    
    Raises:
        FileNotFoundError: If model cannot be found at any location
    """
    try:
        model_path = _resolve_model_path(model_name, fallback_locations)
        if not model_path.exists():
            raise FileNotFoundError(f"Model '{model_name}' not found at {model_path}")
        logger.info(f"Loading model '{model_name}' from {model_path}")
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        raise