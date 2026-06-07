"""Reproducible training run utilities: create run manifests and seed RNGs."""
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List
import subprocess

from .model_manifest import compute_sha256


def get_versions() -> Dict[str, str]:
    """Collect versions of key packages for provenance."""
    versions = {}
    try:
        import numpy as np
        versions['numpy'] = getattr(np, '__version__', 'unknown')
    except Exception:
        versions['numpy'] = 'not_installed'
    try:
        import pandas as pd
        versions['pandas'] = getattr(pd, '__version__', 'unknown')
    except Exception:
        versions['pandas'] = 'not_installed'
    try:
        import xgboost as xgb
        versions['xgboost'] = getattr(xgb, '__version__', 'unknown')
    except Exception:
        versions['xgboost'] = 'not_installed'
    try:
        import sklearn
        versions['scikit-learn'] = getattr(sklearn, '__version__', 'unknown')
    except Exception:
        versions['scikit-learn'] = 'not_installed'
    try:
        import joblib
        versions['joblib'] = getattr(joblib, '__version__', 'unknown')
    except Exception:
        versions['joblib'] = 'not_installed'

    # git commit (if available)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        versions['git_commit'] = commit
    except Exception:
        versions['git_commit'] = 'unknown'

    return versions


def create_run_manifest(
    dataset_paths: List[str],
    config: Dict,
    out_dir: str = 'runs'
) -> Dict:
    """Create a run manifest containing dataset hashes, config, versions, and seed."""
    run_id = str(uuid.uuid4())
    run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    datasets = {}
    for p in dataset_paths:
        if os.path.exists(p):
            datasets[os.path.basename(p)] = compute_sha256(p)
        else:
            datasets[os.path.basename(p)] = None

    manifest = {
        'run_id': run_id,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'config': config,
        'datasets': datasets,
        'versions': get_versions(),
    }

    manifest_path = os.path.join(run_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    return manifest
