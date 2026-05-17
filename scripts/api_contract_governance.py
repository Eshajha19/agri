"""API Contract Governance
- Generates OpenAPI spec from `main.app`
- Stores baseline in `scripts/.contracts/openapi_prev.json`
- Compares current spec against baseline and enforces compatibility gates:
  - Path or operation removals are breaking
  - New required properties in request bodies are breaking
  - Removed response properties are breaking
- Exits with non-zero code on breaking changes
"""
import os
import json
import sys
from deepdiff import DeepDiff

CONTRACTS_DIR = os.path.join(os.path.dirname(__file__), ".contracts")
PREV_FILE = os.path.join(CONTRACTS_DIR, "openapi_prev.json")
CUR_FILE = os.path.join(CONTRACTS_DIR, "openapi_current.json")


def load_openapi():
    # Import the FastAPI app and generate spec using file-based import to avoid
    # module resolution issues.
    try:
        import importlib.util
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        main_path = os.path.join(repo_root, "main.py")
        spec = importlib.util.spec_from_file_location("_main_for_openapi", main_path)
        mod = importlib.util.module_from_spec(spec)
        # Ensure repository root is on sys.path so local packages like `backend` can be imported
        import sys
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        spec.loader.exec_module(mod)
        app = getattr(mod, "app")
    except Exception as e:
        print(f"ERROR: failed to load main.app from {main_path}: {e}")
        raise
    spec = app.openapi()
    return spec


def ensure_dir():
    if not os.path.exists(CONTRACTS_DIR):
        os.makedirs(CONTRACTS_DIR)


def save_spec(path, spec):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)


def load_spec(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_breaking(prev, curr):
    breaking = []
    # 1) Paths or operations removed
    prev_paths = set(prev.get("paths", {}).keys())
    curr_paths = set(curr.get("paths", {}).keys())
    removed_paths = prev_paths - curr_paths
    if removed_paths:
        breaking.append(f"Removed paths: {sorted(list(removed_paths))}")
    # 2) For shared paths, operations removed
    for path in prev_paths & curr_paths:
        prev_ops = set(prev["paths"][path].keys())
        curr_ops = set(curr["paths"][path].keys())
        removed_ops = prev_ops - curr_ops
        if removed_ops:
            breaking.append(f"Removed operations for {path}: {sorted(list(removed_ops))}")
    # 3) New required properties in request bodies
    for path in curr.get("paths", {}):
        for method, op in curr["paths"][path].items():
            try:
                curr_req = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {})
                prev_req = prev.get("paths", {}).get(path, {}).get(method, {}).get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {})
                curr_required = set(curr_req.get("required", [])) if isinstance(curr_req, dict) else set()
                prev_required = set(prev_req.get("required", [])) if isinstance(prev_req, dict) else set()
                added_required = curr_required - prev_required
                if added_required:
                    breaking.append(f"New required request properties on {method.upper()} {path}: {sorted(list(added_required))}")
            except Exception:
                pass
    # 4) Removed top-level response properties (approximate check for 200/201)
    for path in prev.get("paths", {}):
        for method, op in prev["paths"][path].items():
            try:
                prev_resp = op.get("responses", {}).get("200", {}).get("content", {}).get("application/json", {}).get("schema", {})
                curr_resp = curr.get("paths", {}).get(path, {}).get(method, {}).get("responses", {}).get("200", {}).get("content", {}).get("application/json", {}).get("schema", {})
                if prev_resp and curr_resp:
                    prev_props = set(prev_resp.get("properties", {}).keys())
                    curr_props = set(curr_resp.get("properties", {}).keys())
                    removed_props = prev_props - curr_props
                    if removed_props:
                        breaking.append(f"Removed response properties on {method.upper()} {path}: {sorted(list(removed_props))}")
            except Exception:
                pass
    return breaking


def main():
    ensure_dir()
    spec = load_openapi()
    save_spec(CUR_FILE, spec)
    if not os.path.exists(PREV_FILE):
        save_spec(PREV_FILE, spec)
        print("Baseline OpenAPI spec created at", PREV_FILE)
        print("No breaking changes (baseline created).")
        return 0
    prev = load_spec(PREV_FILE)
    curr = spec
    breaking = check_breaking(prev, curr)
    if breaking:
        print("BREAKING CHANGES DETECTED:")
        for b in breaking:
            print(" -", b)
        # Also dump deep diff for debugging
        diff = DeepDiff(prev.get("paths", {}), curr.get("paths", {}), ignore_order=True)
        print("DeepDiff summary:", diff)
        return 2
    else:
        # Optionally, update baseline if no breaking changes (policy: auto-bump non-breaking)
        # But we will not auto-update to preserve history; just report success
        print("No breaking changes detected. API contract compatible.")
        return 0


if __name__ == "__main__":
    code = main()
    sys.exit(code)
