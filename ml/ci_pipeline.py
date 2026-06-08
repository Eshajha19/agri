"""CI helpers: schema validation, artifact signing/verification."""
import csv
import hashlib
import hmac
from typing import List


def validate_csv_schema(path: str, min_columns: int = 3, required_columns: List[str] = None) -> bool:
    """Lightweight CSV schema validation for CI.

    - ensures file exists and has at least `min_columns` header columns
    - if `required_columns` provided, ensure all present
    """
    if not required_columns:
        required_columns = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError("CSV is empty")
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {path}")
    except OSError as exc:
        raise OSError(f"Failed to open CSV file '{path}': {exc}") from exc

    if len(header) < min_columns:
        raise ValueError(f"CSV header has too few columns: {len(header)} < {min_columns}")

    missing = [c for c in required_columns if c not in header]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return True


def sign_file_hmac(path: str, key: str) -> str:
    """Compute HMAC-SHA256 signature for a file and return hex digest."""
    h = hmac.new(key.encode("utf-8"), digestmod=hashlib.sha256)
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found for signing: {path}")
    except OSError as exc:
        raise OSError(f"Failed to open file for signing '{path}': {exc}") from exc
    return h.hexdigest()


def verify_file_hmac(path: str, sig_hex: str, key: str) -> bool:
    return hmac.compare_digest(sign_file_hmac(path, key), sig_hex)