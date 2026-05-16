import logging
import threading
from typing import Dict, Optional

from ml.base import YieldModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Per-process registry that maps model names to loaded model instances.

    Multi-worker safety
    -------------------
    In a multi-worker Uvicorn/Gunicorn deployment each worker is a separate
    OS process with its own memory space.  Class-level state is NOT shared
    across processes — every worker must initialise its own registry.

    The registry is designed to be initialised inside a FastAPI ``lifespan``
    context manager (or an ``@app.on_event("startup")`` handler) so that
    initialisation runs in **every** worker process, not just the main one.

    Thread safety
    -------------
    ``_lock`` serialises ``register`` calls so that concurrent startup
    coroutines (if any) cannot corrupt ``_models``.  ``get_model`` and
    ``list_models`` are read-only and safe without locking because CPython's
    GIL makes dict reads atomic for simple key lookups.
    """

    _models: Dict[str, YieldModel] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def register(cls, model_name: str, model_instance: YieldModel) -> None:
        """Register a model instance under *model_name*.

        Thread-safe: acquires ``_lock`` before mutating ``_models``.
        """
        with cls._lock:
            cls._models[model_name] = model_instance
        logger.info(
            "ModelRegistry: registered '%s' (%s) in worker pid=%d",
            model_name,
            model_instance.model_type,
            __import__("os").getpid(),
        )

    @classmethod
    def get_model(cls, model_name: str) -> Optional[YieldModel]:
        """Return the model registered under *model_name*, or ``None``."""
        return cls._models.get(model_name)

    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """Return a ``{name: model_type}`` snapshot of all registered models."""
        return {name: model.model_type for name, model in cls._models.items()}

    @classmethod
    def is_empty(cls) -> bool:
        """Return ``True`` when no models have been registered yet."""
        return len(cls._models) == 0
