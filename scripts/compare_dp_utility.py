"""Short experiment script: baseline utility vs DP-SGD utility.

This script creates two reproducible configs and runs training via train_model.py:
1) baseline (XGBoost)
2) dp_sgd (torch + opacus, optional)

Results are written as JSON for quick reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_model import train_from_config


def _write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _build_config(base: Dict[str, Any], mode: str, out_model: str, epsilon: float, delta: float) -> Dict[str, Any]:
    cfg = dict(base)
    cfg["training_mode"] = mode
    cfg["output_model"] = out_model
    if mode == "dp_sgd":
        cfg["epsilon"] = epsilon
        cfg["delta"] = delta
        cfg.setdefault("dp_epochs", 4)
        cfg.setdefault("dp_batch_size", 64)
    return cfg


def run_experiment(args) -> Dict[str, Any]:
    base = {
        "dataset": args.dataset,
        "seed": args.seed,
        "test_size": args.test_size,
        "model_name": args.model_name,
        "created_by": "dp_experiment",
    }

    summary: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset": args.dataset,
        "seed": args.seed,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "results": {},
    }

    # Use a permanent output directory for model artifacts so paths remain
    # valid after the temporary config/scratch directory is cleaned up.
    output_dir = Path(args.output).parent / f"dp_models_{args.model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_model_path = str(output_dir / "baseline_model.joblib")
    dp_model_path = str(output_dir / "dp_model.pt")

    with tempfile.TemporaryDirectory(prefix="dp-compare-") as tmpdir:
        baseline_cfg = _build_config(
            base=base,
            mode="baseline",
            out_model=baseline_model_path,
            epsilon=args.epsilon,
            delta=args.delta,
        )
        baseline_cfg_path = os.path.join(tmpdir, "baseline_config.json")
        _write_json(baseline_cfg_path, baseline_cfg)
        summary["results"]["baseline"] = train_from_config(baseline_cfg_path)
        summary["results"]["baseline_model_path"] = baseline_model_path

        dp_cfg = _build_config(
            base=base,
            mode="dp_sgd",
            out_model=dp_model_path,
            epsilon=args.epsilon,
            delta=args.delta,
        )
        dp_cfg_path = os.path.join(tmpdir, "dp_config.json")
        _write_json(dp_cfg_path, dp_cfg)

        try:
            summary["results"]["dp_sgd"] = train_from_config(dp_cfg_path)
            summary["results"]["dp_model_path"] = dp_model_path
            summary["dp_status"] = "ok"
        except Exception as exc:  # pragma: no cover - dependency/runtime sensitive path
            summary["dp_status"] = "failed"
            summary["dp_error"] = str(exc)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and DP utility for yield prediction.")
    parser.add_argument("--dataset", default="Train.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model-name", default="yield_model")
    parser.add_argument("--epsilon", type=float, default=3.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument(
        "--output",
        default="dp_utility_comparison.json",
        help="Output JSON path for experiment summary",
    )
    args = parser.parse_args()

    summary = run_experiment(args)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote experiment summary to: {output_path}")


if __name__ == "__main__":
    main()