import os
import tempfile
import pandas as pd
import pytest
from backend.train_yield_model import train_yield_model, train_from_config_dict

def make_dummy_csv(path: str):
    df = pd.DataFrame({
        "SDate": pd.date_range("2020-01-01", periods=10, freq="D"),
        "ExpYield": range(10),
        "Crop": ["Wheat"] * 10,
    })
    df.to_csv(path, index=False)

def test_train_yield_model_runs(tmp_path):
    csv_path = tmp_path / "Train.csv"
    make_dummy_csv(csv_path)

    result = train_yield_model(csv_path=str(csv_path), model_output=str(tmp_path / "model.joblib"))
    assert "rmse" in result
    assert os.path.exists(result["model_path"])

def test_train_from_config_dict_runs(tmp_path):
    csv_path = tmp_path / "Train.csv"
    make_dummy_csv(csv_path)

    config = {
        "dataset": str(csv_path),
        "output_model": str(tmp_path / "model.joblib"),
        "training_mode": "baseline",
        "categorical_cols": ["Crop"],
    }
    result = train_from_config_dict(config)
    assert "training_mode" in result
    assert os.path.exists(result["model_path"])
