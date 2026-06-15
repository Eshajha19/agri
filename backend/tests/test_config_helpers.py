import pytest
from backend.train_yield_model import resolve_config, DEFAULT_XGB_CONFIG, DEFAULT_DP_CONFIG

def test_resolve_config_merges_defaults():
    user_config = {"n_estimators": 50, "max_depth": 3}
    merged = resolve_config(user_config, DEFAULT_XGB_CONFIG)
    assert merged["n_estimators"] == 50
    assert merged["max_depth"] == 3
    # untouched defaults remain
    assert merged["learning_rate"] == DEFAULT_XGB_CONFIG["learning_rate"]

def test_resolve_config_with_none():
    merged = resolve_config(None, DEFAULT_XGB_CONFIG)
    assert merged == DEFAULT_XGB_CONFIG
