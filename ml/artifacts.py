"""
Graceful ML artifact loading utilities.

Prevents startup failures when model files are missing or corrupted,
and provides meaningful error messages for debugging.
"""
import os
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


def load_joblib_model(
    path: str,
    model_name: str,
    optional: bool = True,
) -> Optional[Any]:
    """
    Load a joblib model file with graceful fallback.

    Parameters
    ----------
    path : str
        File path to the .joblib model.
    model_name : str
        Human-readable name for logging (e.g. "lag yield model").
    optional : bool
        If True, returns None on failure instead of raising.

    Returns
    -------
    The loaded model, or None if optional=True and loading failed.
    """
    if not os.path.exists(path):
        msg = f"ML artifact not found: {path} ({model_name}) — running without it"
        if optional:
            logger.warning(msg)
            return None
        raise FileNotFoundError(msg)

    try:
        import joblib
        model = joblib.load(path)
        logger.info("Loaded %s from %s", model_name, path)
        return model
    except Exception as exc:
        msg = f"Failed to load {model_name} from {path}: {exc}"
        if optional:
            logger.warning("%s — running without it", msg)
            return None
        raise RuntimeError(msg) from exc


def load_keras_model(
    path: str,
    model_name: str,
    optional: bool = True,
) -> Optional[Any]:
    """
    Load a Keras/TensorFlow model with graceful fallback.

    Parameters
    ----------
    path : str
        File path to the .h5 or SavedModel directory.
    model_name : str
        Human-readable name for logging.
    optional : bool
        If True, returns None on failure instead of raising.

    Returns
    -------
    The loaded Keras model, or None if optional=True and loading failed.
    """
    if not os.path.exists(path):
        msg = f"ML artifact not found: {path} ({model_name}) — running without it"
        if optional:
            logger.warning(msg)
            return None
        raise FileNotFoundError(msg)

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        logger.info("Loaded %s from %s", model_name, path)
        return model
    except Exception as exc:
        msg = f"Failed to load {model_name} from {path}: {exc}"
        if optional:
            logger.warning("%s — running without it", msg)
            return None
        raise RuntimeError(msg) from exc


def load_model_file(
    path: str,
    model_name: str,
    optional: bool = True,
) -> Optional[Any]:
    """
    Auto-detect format and load a model file.

    Supports .joblib, .pkl, .h5, and SavedModel directories.
    """
    if not os.path.exists(path):
        msg = f"ML artifact not found: {path} ({model_name}) — running without it"
        if optional:
            logger.warning(msg)
            return None
        raise FileNotFoundError(msg)

    ext = os.path.splitext(path)[1].lower()
    if ext in (".h5", ".keras"):
        return load_keras_model(path, model_name, optional)
    elif ext in (".joblib", ".pkl"):
        return load_joblib_model(path, model_name, optional)
    elif os.path.isdir(path):
        return load_keras_model(path, model_name, optional)
    else:
        msg = f"Unknown model format for {path} ({model_name})"
        if optional:
            logger.warning(msg)
            return None
        raise ValueError(msg)
