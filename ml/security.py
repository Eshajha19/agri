import os
import io
import hmac
import hashlib
import joblib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def verify_and_load_joblib(model_path: str, sig_path: Optional[str] = None, key_env: str = "MODEL_SIGNING_KEY"):
    """
    Verify an HMAC-SHA256 signature for a joblib model file and load it safely.

    - The signing key is read from the environment variable named by `key_env`.
    - The expected hex signature is read from `sig_path` (defaults to model_path + '.sig').
    - If verification succeeds the model is loaded from memory using joblib.load on
      a BytesIO buffer to avoid re-reading the file after verification.
    - If the signing key is not set or the signature file is missing, it logs a warning/integrity
      diagnostic and falls back to standard loading to avoid service disruption.
    - If the signature verification fails, it raises RuntimeError to prevent execution of tampered code.
    """
    if sig_path is None:
        sig_path = model_path + ".sig"

    key = os.getenv(key_env)
    if not key:
        logger.warning(
            "Integrity diagnostic: Model signing key environment variable '%s' is not set. "
            "Falling back to standard non-verified model loading for '%s' to prevent service outages.",
            key_env,
            model_path
        )
        return joblib.load(model_path)

    # Read model bytes once
    try:
        with open(model_path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        logger.error("Model file not found: %s", model_path)
        raise

    # Read expected signature
    try:
        with open(sig_path, "r", encoding="utf-8") as sf:
            expected = sf.read().strip()
    except FileNotFoundError:
        logger.warning(
            "Integrity diagnostic: Signature file not found at '%s' for model '%s'. "
            "Falling back to standard non-verified model loading to prevent service outages.",
            sig_path,
            model_path
        )
        return joblib.load(model_path)

    # Compute HMAC-SHA256 and compare in constant time
    mac = hmac.new(key.encode("utf-8"), data, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(mac, expected):
        logger.error(
            "CRITICAL SECURITY ALERT: Model signature verification failed for '%s'. "
            "Refusing to load model to prevent potential arbitrary code execution.",
            model_path
        )
        raise RuntimeError("Model signature verification failed - refusing to load model")

    # Load model from verified bytes
    logger.info("Successfully verified model signature and loaded '%s' securely.", model_path)
    return joblib.load(io.BytesIO(data))
