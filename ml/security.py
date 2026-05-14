import os
import io
import hmac
import hashlib
import joblib
from typing import Optional


def verify_and_load_joblib(model_path: str, sig_path: Optional[str] = None, key_env: str = "MODEL_SIGNING_KEY"):
    """
    Verify an HMAC-SHA256 signature for a joblib model file and load it safely.

    - The signing key is read from the environment variable named by `key_env`.
    - The expected hex signature is read from `sig_path` (defaults to model_path + '.sig').
    - If verification succeeds the model is loaded from memory using joblib.load on
      a BytesIO buffer to avoid re-reading the file after verification.

    Raises RuntimeError on missing key, missing signature, or verification failure.
    """
    if sig_path is None:
        sig_path = model_path + ".sig"

    key = os.getenv(key_env)
    if not key:
        raise RuntimeError(f"Model signing key environment variable '{key_env}' is not set")

    # Read model bytes once
    with open(model_path, "rb") as f:
        data = f.read()

    # Read expected signature
    try:
        with open(sig_path, "r", encoding="utf-8") as sf:
            expected = sf.read().strip()
    except FileNotFoundError:
        raise RuntimeError(f"Signature file not found for model: {sig_path}")

    # Compute HMAC-SHA256 and compare in constant time
    mac = hmac.new(key.encode("utf-8"), data, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(mac, expected):
        raise RuntimeError("Model signature verification failed - refusing to load model")

    # Load model from verified bytes
    return joblib.load(io.BytesIO(data))
