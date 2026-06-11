"""Convert common model formats to ONNX.

Supports:
- scikit-learn / XGBoost saved with joblib (uses skl2onnx if available)
- Keras (.h5) models (uses tf2onnx if available)

This is a best-effort conversion tool; installation of optional
dependencies is required for each path.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

def convert_sklearn(model_path: str, out_path: str, opset: int = 14, n_features: int | None = None):
    try:
        import joblib
        from skl2onnx import convert_sklearn as skl2_convert
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as exc:
        raise RuntimeError("scikit-learn/XGBoost conversion requires 'skl2onnx' and 'joblib' installed") from exc

    model = joblib.load(model_path)
    if n_features is None:
        n_features = getattr(model, "n_features_in_", None)
        if n_features is None:
            raise RuntimeError("Could not infer n_features; provide --n-features")

    initial_types = [("input", FloatTensorType([None, int(n_features)]))]
    onnx_model = skl2_convert(model, initial_types=initial_types, target_opset=opset)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def convert_keras(model_path: str, out_path: str, opset: int = 14):
    try:
        import tensorflow as tf
        import tf2onnx
    except Exception as exc:
        raise RuntimeError("Keras conversion requires 'tf2onnx' and 'tensorflow' installed") from exc

    model = tf.keras.models.load_model(model_path)
    spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset)
    with open(out_path, "wb") as f:
        f.write(model_proto.SerializeToString())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to input model file")
    p.add_argument("--out", required=False, help="ONNX output path (defaults same name with .onnx)")
    p.add_argument("--opset", type=int, default=14)
    p.add_argument("--n-features", type=int, default=None, help="Number of input features (for sklearn models)")
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model path not found: {model_path}")
        sys.exit(2)

    out_path = Path(args.out) if args.out else model_path.with_suffix(".onnx")

    try:
        if model_path.suffix in (".joblib", ".pkl"):
            convert_sklearn(str(model_path), str(out_path), opset=args.opset, n_features=args.n_features)
        elif model_path.suffix in (".h5", ".keras", ".keras2"):
            convert_keras(str(model_path), str(out_path), opset=args.opset)
        else:
            raise RuntimeError("Unsupported model extension for automatic conversion")
    except Exception as exc:
        print(f"Conversion failed: {exc}")
        sys.exit(1)

    print(f"Wrote ONNX model: {out_path}")


if __name__ == "__main__":
    main()
