from typing import Dict, Type, Optional
from ml.base import YieldModel

class ModelRegistry:
    """
    Registry system to manage available yield prediction models.
    """
    _models: Dict[str, YieldModel] = {}

    @classmethod
    def register(cls, model_name: str, model_instance: YieldModel):
        """Register a new model instance."""
        cls._models[model_name] = model_instance
        print(f"Model '{model_name}' ({model_instance.model_type}) registered.")

    @classmethod
    def get_model(cls, model_name: str) -> Optional[YieldModel]:
        """Retrieve a model instance by name."""
        return cls._models.get(model_name)

    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """List all registered models and their types."""
        return {name: model.model_type for name, model in cls._models.items()}
