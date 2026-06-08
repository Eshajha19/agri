"""CLI to validate a model file and register it into the in-memory registry.

Usage:
  python scripts/validate_and_register.py <model_path> <model_name> <version>
"""
import sys
import os
from ml.validator import validate_model_file
from ml.model_manifest import create_manifest, write_manifest
from ml_model_registry import get_model_registry


def main(argv):
    if len(argv) < 4:
        print("Usage: validate_and_register.py <model_path> <model_name> <version>")
        return 2

    _, model_path, model_name, version = argv

    ok, details = validate_model_file(model_path)
    if not ok:
        print("Validation failed:", details)
        return 1

    manifest = create_manifest(model_path, model_name, version, created_by="cli")

    # ensure models dir exists
    out_dir = os.path.join("models", model_name, version)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    write_manifest(manifest, manifest_path)

    # Register in global registry — if this fails, clean up the orphaned manifest
    try:
        registry = get_model_registry()
        registry.register_model(
            model_name=model_name,
            version=version,
            model_path=os.path.abspath(model_path),
            created_by="cli",
            description=f"Registered via validate_and_register: {manifest_path}",
            metrics={}
        )
    except Exception as exc:
        # Remove orphaned manifest so it cannot be mistaken for a registered model
        try:
            os.remove(manifest_path)
        except OSError as remove_err:
            print("Warning: failed to remove orphaned manifest:", remove_err)
        print("Registry error, manifest rolled back:", exc)
        return 1

    print("Validation succeeded, manifest written to:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))