"""Utilities for creating model manifests (hash, metadata)."""
import hashlib
import json
from datetime import datetime
from typing import Dict


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def create_manifest(path: str, model_name: str, version: str, created_by: str = "system", extra: Dict = None) -> Dict:
    """Create a manifest dict for a model file and return it.

    The manifest includes hash, file name, created_at, and any extra metadata.
    """
    manifest = {
        "model_name": model_name,
        "version": version,
        "file_name": path,
        "sha256": compute_sha256(path),
        "created_by": created_by,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    if extra:
        manifest.update(extra)

    return manifest


def write_manifest(manifest: Dict, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
