import threading
import pytest
from backend.ml_model_registry import ModelRegistry

def test_concurrent_register_and_get():
    registry = ModelRegistry()

    def register_model():
        registry.register("m1", object())

    def get_model():
        try:
            registry.get_model("m1")
        except KeyError:
            pass

    threads = []
    for _ in range(5):
        threads.append(threading.Thread(target=register_model))
        threads.append(threading.Thread(target=get_model))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # registry should contain model once
    assert "m1" in registry.list_models()
    assert isinstance(registry.get_model("m1"), object)

def test_missing_model_error():
    registry = ModelRegistry()
    with pytest.raises(KeyError):
        registry.get_model("unknown")
