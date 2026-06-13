"""ONNX runtime wrapper with GPU-first provider selection and CPU fallback."""

from __future__ import annotations

import numpy as np
import os
from typing import List


class ONNXRuntimeModel:
    def __init__(self, model_path: str, prefer_gpu: bool = True):
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError("onnxruntime is required for ONNX inference") from exc

        self.ort = ort
        self.model_path = model_path

        providers = self.ort.get_available_providers()
        selected_providers: List[str] = []
        if prefer_gpu and "CUDAExecutionProvider" in providers:
            selected_providers.append("CUDAExecutionProvider")
        # Always include CPU provider as fallback
        if "CPUExecutionProvider" in providers:
            selected_providers.append("CPUExecutionProvider")

        if not selected_providers:
            # If no providers discovered, let InferenceSession choose defaults
            self.sess = self.ort.InferenceSession(self.model_path)
        else:
            try:
                self.sess = self.ort.InferenceSession(self.model_path, providers=selected_providers)
            except Exception:
                # Fallback to default
                self.sess = self.ort.InferenceSession(self.model_path)

        # Determine input name(s)
        self.input_names = [inp.name for inp in self.sess.get_inputs()]
        self.output_names = [out.name for out in self.sess.get_outputs()]

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.input_names:
            raise RuntimeError(
                f"ONNX model '{self.model_path}' has no inputs — model may be malformed"
            )
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)

        # If model expects 2D, ensure shape compatibility
        feed = {self.input_names[0]: x.astype(np.float32)}
        res = self.sess.run(self.output_names, feed)
        # If single output, return flattened array
        return np.asarray(res[0])

    def providers(self):
        return self.sess.get_providers()
