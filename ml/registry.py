import logging
import threading
from typing import Dict, List


logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Thread-safe in-memory ML model registry.
    """

    _models: Dict[str, object] = {}
    _lock = threading.Lock()

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    @classmethod
    def register(cls, model_name: str, model) -> None:
        """
        Register model instance.
        """

        if not isinstance(model_name, str):
            raise ValueError("model_name must be string")

        model_name = model_name.strip()

        if not model_name:
            raise ValueError("model_name cannot be empty")

        if model is None:
            raise ValueError("model cannot be None")

        with cls._lock:
            cls._models[model_name] = model

        logger.info(
            "Registered model '%s'",
            model_name,
        )

    # =========================================================================
    # LOOKUP
    # =========================================================================

    @classmethod
    def get(cls, model_name: str):
        """
        Retrieve model by name.
        """

        if not isinstance(model_name, str):
            raise ValueError("model_name must be string")

        with cls._lock:
            model = cls._models.get(model_name)

        if model is None:
            raise KeyError(
                f"Model '{model_name}' not registered"
            )

        return model

    @classmethod
    def exists(cls, model_name: str) -> bool:
        """
        Check if model exists.
        """

        with cls._lock:
            return model_name in cls._models

    # =========================================================================
    # REMOVAL
    # =========================================================================

    @classmethod
    def unregister(cls, model_name: str) -> bool:
        """
        Remove model from registry.
        """

        with cls._lock:
            if model_name not in cls._models:
                return False

            del cls._models[model_name]

        logger.info(
            "Unregistered model '%s'",
            model_name,
        )

        return True

    # =========================================================================
    # LISTING
    # =========================================================================

    @classmethod
    def list_models(cls) -> List[str]:
        """
        Return registered model names.
        """

        with cls._lock:
            return sorted(cls._models.keys())

    @classmethod
    def clear(cls) -> None:
        """
        Clear registry.
        """

        with cls._lock:
            cls._models.clear()

        logger.warning(
            "Model registry cleared"
        )

    # =========================================================================
    # DEBUG / HEALTH
    # =========================================================================

    @classmethod
    def stats(cls):
        """
        Registry health snapshot.
        """

        models = cls.list_models()

        return {
            "registered_models": models,
            "total_models": len(models),
        }