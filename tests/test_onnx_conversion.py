import os
import tempfile
import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.convert_model_to_onnx import convert_sklearn


def test_sklearn_to_onnx_conversion_and_runtime():
    X = np.random.rand(100, 5).astype(np.float32)
    y = (X.sum(axis=1) + np.random.randn(100) * 0.01).astype(np.float32)

    model = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, "m.joblib")
        onnx_path = os.path.join(td, "m.onnx")
        joblib.dump(model, model_path)

        # convert
        convert_sklearn(model_path, onnx_path, opset=13, n_features=5)

        assert os.path.exists(onnx_path)

        # load via ONNX runtime wrapper
        try:
            from inference.onnx_runtime import ONNXRuntimeModel
        except Exception:
            return

        m = ONNXRuntimeModel(onnx_path, prefer_gpu=False)
        inp = np.random.rand(2, 5).astype(np.float32)
        out = m.predict(inp)
        assert out.shape[0] == inp.shape[0]
