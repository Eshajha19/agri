import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Dict, Generic, List, Optional, TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class ModelEntry(Generic[T]):
    """
    Registry record.
    """

    name: str
    model: T
    registered_at: datetime


class ModelRegistry(Generic[T]):
    """
    Thread-safe in-memory model registry.

    Example:
        ModelRegistry.register("bert", model)
        model = ModelRegistry.get("bert")
    """

    _models: Dict[str, ModelEntry[Any]] = {}
    _lock = RLock()

    # =========================================================================
    # INTERNALS
    # =========================================================================

    @classmethod
    def _normalize_name(cls, model_name: str) -> str:
        """
        Validate and normalize model name.
        """

        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string")

        model_name = model_name.strip()

        if not model_name:
            raise ValueError("model_name cannot be empty")

        return model_name

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    @classmethod
    def register(
        cls,
        model_name: str,
        model: T,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register a model.

        Args:
            model_name: Unique model name.
            model: Model instance.
            overwrite: Allow replacing existing model.
        """

        model_name = cls._normalize_name(model_name)

        if model is None:
            raise ValueError("model cannot be None")

        with cls._lock:
            exists = model_name in cls._models

            if exists and not overwrite:
                raise ValueError(
                    f"Model '{model_name}' is already registered"
                )

            cls._models[model_name] = ModelEntry(
                name=model_name,
                model=model,
                registered_at=datetime.now(timezone.utc),
            )

        logger.info(
            "Registered model '%s'%s",
            model_name,
            " (overwritten)" if exists else "",
        )

    # =========================================================================
    # LOOKUP
    # =========================================================================

    @classmethod
    def get(cls, model_name: str) -> T:
        """
        Retrieve model instance.
        """

        model_name = cls._normalize_name(model_name)

        with cls._lock:
            entry = cls._models.get(model_name)

        if entry is None:
            raise KeyError(
                f"Model '{model_name}' is not registered"
            )

        return entry.model

    @classmethod
    def get_entry(
        cls,
        model_name: str,
    ) -> ModelEntry[T]:
        """
        Retrieve full registry entry.
        """

        model_name = cls._normalize_name(model_name)

        with cls._lock:
            entry = cls._models.get(model_name)

        if entry is None:
            raise KeyError(
                f"Model '{model_name}' is not registered"
            )

        return entry

    @classmethod
    def exists(cls, model_name: str) -> bool:
        """
        Check whether a model exists.
        """

        model_name = cls._normalize_name(model_name)

        with cls._lock:
            return model_name in cls._models

    # =========================================================================
    # REMOVAL
    # =========================================================================

    @classmethod
    def unregister(cls, model_name: str) -> bool:
        """
        Remove a model from the registry.

        Returns:
            True if removed.
            False if model was not present.
        """

        model_name = cls._normalize_name(model_name)

        with cls._lock:
            removed = cls._models.pop(model_name, None)

        if removed is not None:
            logger.info(
                "Unregistered model '%s'",
                model_name,
            )

        return removed is not None

    @classmethod
    def clear(cls) -> None:
        """
        Remove all models.
        """

        with cls._lock:
            removed_count = len(cls._models)
            cls._models.clear()

        logger.warning(
            "Cleared model registry (%d models removed)",
            removed_count,
        )

    # =========================================================================
    # LISTING
    # =========================================================================

    @classmethod
    def list_models(cls) -> List[str]:
        """
        Return sorted model names.
        """

        with cls._lock:
            return sorted(cls._models.keys())

    @classmethod
    def count(cls) -> int:
        """
        Return total number of registered models.
        """

        with cls._lock:
            return len(cls._models)

    @classmethod
    def snapshot(cls) -> Dict[str, ModelEntry[Any]]:
        """
        Return shallow copy of registry.
        """

        with cls._lock:
            return dict(cls._models)

    # =========================================================================
    # STATS / HEALTH
    # =========================================================================

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        """
        Registry health snapshot.
        """

        with cls._lock:
            names = sorted(cls._models.keys())

        return {
            "total_models": len(names),
            "registered_models": names,
        }

    # =========================================================================
    # MAGIC METHODS
    # =========================================================================

    @classmethod
    def contains(cls, model_name: str) -> bool:
        """
        Alias for exists().
        """

        return cls.exists(model_name)
