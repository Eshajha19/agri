import os
import io
import hmac
import hashlib
import joblib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

ALLOW_UNSIGNED_MODELS = (
    os.getenv("ALLOW_UNSIGNED_MODELS", "false").lower() == "true"
)


def verify_and_load_joblib(
    model_path: str,
    sig_path: Optional[str] = None,
    key_env: str = "MODEL_SIGNING_KEY",
):
    """
    Verify a joblib model signature before loading.

    Models are loaded only after successful HMAC-SHA256 verification unless
    ALLOW_UNSIGNED_MODELS=true is explicitly enabled for development.
    """
    if sig_path is None:
        sig_path = model_path + ".sig"

    key = os.getenv(key_env)
    if not key:
        logger.warning(
            "Model signing key '%s' is not configured for '%s'",
            key_env,
            model_path
        )
        if ALLOW_UNSIGNED_MODELS:
            logger.warning(
                "Loading unsigned model '%s' because ALLOW_UNSIGNED_MODELS=true",
                model_path,
            )
            return joblib.load(model_path)

        raise RuntimeError(
            f"Model signing key '{key_env}' is not configured for '{model_path}'"
        )

    # Read model bytes once
    try:
        with open(model_path, "rb") as f:
            data = f.read()
    except FileNotFoundError as e:
        logger.error("Model file not found: %s", model_path)
        raise

    # Read expected signature
    try:
        with open(sig_path, "r", encoding="utf-8") as sf:
            expected = sf.read().strip()
    except FileNotFoundError as e:
        logger.warning(
            "Signature file '%s' not found for model '%s'",
            sig_path,
            model_path
        )
        if ALLOW_UNSIGNED_MODELS:
            logger.warning(
                "Loading unsigned model '%s' because ALLOW_UNSIGNED_MODELS=true",
                model_path,
            )
            return joblib.load(model_path)

        raise RuntimeError(
            f"Signature file missing for model '{model_path}'"
        ) from e

    # Compute HMAC-SHA256 and compare in constant time
    mac = hmac.new(key.encode("utf-8"), data, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(mac, expected):
        logger.error(
            "CRITICAL SECURITY ALERT: Model signature verification failed for '%s'. "
            "Refusing to load model to prevent potential arbitrary code execution.",
            model_path
        )
        raise RuntimeError(
            "Model signature verification failed - refusing to load model"
        )

    # Load model from verified bytes
    logger.info(
        "Successfully verified and securely loaded model '%s'",
        model_path,
    )
    return joblib.load(io.BytesIO(data))
