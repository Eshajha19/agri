"""Benchmark ONNX model inference latency with GPU fallback.

Writes JSON summary including mean latency, std, provider used, and environment details.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
import numpy as np

from inference.onnx_runtime import ONNXRuntimeModel


def run_benchmark(model_path: str, input_shape: tuple, iterations: int = 200, warmup: int = 20):
    m = ONNXRuntimeModel(model_path)
    active_provider = getattr(m, "get_active_provider", lambda: "Unknown")()

    # Create synthetic input
    batch = max(1, input_shape[0])
    sample = np.random.rand(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        _ = m.predict(sample)

    latencies = []
    for _ in range(iterations):
    # Ensure GPU sync before timing
        try:
            import torch
            if torch.cuda.is_available():
            torch.cuda.synchronize()
        except Exception:
            pass

        t0 = time.perf_counter()
        _ = m.predict(sample)

    # Ensure GPU sync after inference
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)


    arr = np.array(latencies)
    result = {
        "model_path": str(model_path),
        "providers": providers,
        "active_provider": active_provider,
        "iterations": int(iterations),
        "warmup": int(warmup),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "throughput_fps": float(1000.0 / arr.mean()),
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input-shape", required=False, help="Comma-separated input shape (e.g. 1,39)")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--output", default="benchmark_result.json")
    args = p.parse_args()

    if args.input_shape:
        shape = tuple(int(x) for x in args.input_shape.split(","))
    else:
        shape = (1, 39)

    res = run_benchmark(args.model, shape, iterations=args.iterations, warmup=args.warmup)
    Path(args.output).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote benchmark result to {args.output}")


if __name__ == "__main__":
    main()
