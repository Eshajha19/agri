import os
import io
import hmac
import hashlib
import joblib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Allow unsigned models in dev/test environments via env vars.
# NEVER set ALLOW_UNSIGNED_MODELS=true in production.
ALLOW_UNSIGNED_MODELS = (
    os.getenv("ALLOW_UNSIGNED_MODELS", "false").lower() == "true"
    or os.getenv("LOCAL_TEST_MODE", "false").lower() == "true"
    or os.getenv("TESTING", "false").lower() == "true"
)


class ModelSignatureError(RuntimeError):
    """Raised when a model file fails HMAC-SHA256 signature verification.

    This indicates the model file may have been tampered with or corrupted.
    Loading a tampered joblib file can result in arbitrary code execution
    via pickle deserialization. Always treat this as a critical security event.
    """


def sign_model(model_path: str, key_env: str = "MODEL_SIGNING_KEY") -> str:
    """Compute and persist an HMAC-SHA256 signature for a joblib model file.

    The signature is written to ``<model_path>.sig`` as a hex string.
    Run this once during model training/publishing before deploying to production.

    Args:
        model_path: Path to the ``.joblib`` model file to sign.
        key_env: Name of the environment variable holding the signing secret key.

    Returns:
        The hex-encoded HMAC-SHA256 signature string.

    Raises:
        RuntimeError: If the signing key env var is not configured.
        FileNotFoundError: If the model file does not exist.
    """
    key = os.getenv(key_env)
    if not key:
        raise RuntimeError(
            f"Model signing key '{key_env}' is not configured. "
            "Set this environment variable before signing models."
        )

    with open(model_path, "rb") as f:
        data = f.read()

    sig = hmac.new(key.encode("utf-8"), data, hashlib.sha256).hexdigest()
    sig_path = model_path + ".sig"

    with open(sig_path, "w", encoding="utf-8") as sf:
        sf.write(sig)

    logger.info(
        "Model '%s' signed successfully. Signature written to '%s'",
        model_path,
        sig_path,
    )
    return sig


def verify_and_load_joblib(
    model_path: str,
    sig_path: Optional[str] = None,
    key_env: str = "MODEL_SIGNING_KEY",
):
    """Verify a joblib model's HMAC-SHA256 signature before loading.

    Replaces direct ``joblib.load()`` calls to prevent arbitrary code execution
    attacks via tampered or corrupted model files. Models are deserialized only
    after the signature is verified in constant time.

    Args:
        model_path: Path to the ``.joblib`` model file.
        sig_path: Optional path to the signature file. Defaults to
                  ``<model_path>.sig``.
        key_env: Name of the environment variable holding the signing key.

    Returns:
        The deserialized model object.

    Raises:
        ModelSignatureError: If HMAC verification fails (file tampered/corrupted).
        RuntimeError: If signing key is missing and unsigned models are not allowed,
                      or if the signature file is missing.
        FileNotFoundError: If the model file itself does not exist.
    """
    if sig_path is None:
        sig_path = model_path + ".sig"

    key = os.getenv(key_env)
    if not key:
        logger.warning(
            "Model signing key '%s' is not configured for '%s'",
            key_env,
            model_path,
        )
        if ALLOW_UNSIGNED_MODELS:
            logger.warning(
                "Loading unsigned model '%s' because ALLOW_UNSIGNED_MODELS=true. "
                "This MUST NOT be used in production.",
                model_path,
            )
            return joblib.load(model_path)

        raise RuntimeError(
            f"Model signing key '{key_env}' is not configured for '{model_path}'. "
            "Set MODEL_SIGNING_KEY or enable ALLOW_UNSIGNED_MODELS for local dev."
        )

    # Read model bytes once to avoid TOCTOU issues
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
            "Signature file '%s' not found for model '%s'",
            sig_path,
            model_path,
        )
        if ALLOW_UNSIGNED_MODELS:
            logger.warning(
                "Loading unsigned model '%s' because ALLOW_UNSIGNED_MODELS=true. "
                "This MUST NOT be used in production.",
                model_path,
            )
            return joblib.load(model_path)

        raise RuntimeError(
            f"Signature file missing for model '{model_path}'. "
            "Generate a signature using ml.security.sign_model() before deployment."
        )

    # Compute HMAC-SHA256 and compare in constant time to prevent timing attacks
    mac = hmac.new(key.encode("utf-8"), data, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(mac, expected):
        logger.error(
            "CRITICAL SECURITY ALERT: Signature verification FAILED for '%s'. "
            "Expected: %s... Got: %s... "
            "Refusing to load — file may have been tampered with or corrupted.",
            model_path,
            expected[:12],
            mac[:12],
        )
        raise ModelSignatureError(
            f"Model signature verification failed for '{model_path}' — "
            "the file may have been tampered with or corrupted. "
            "This is a critical security event."
        )

    # Load from already-buffered bytes to avoid re-reading from disk (TOCTOU safe)
    logger.info(
        "HMAC-SHA256 verification passed. Securely loading model '%s'",
        model_path,
    )
    return joblib.load(io.BytesIO(data))
