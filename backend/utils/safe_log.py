import re
import unicodedata
import hashlib

# ---------------------------------------------------------------------------
# Existing protections
# ---------------------------------------------------------------------------

CONTROL_CHARS = re.compile(r'[\x00-\x1F\x7F]')
ANSI_ESCAPE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

# ---------------------------------------------------------------------------
# Unicode spoofing protections
# ---------------------------------------------------------------------------

BIDI_CONTROL_CHARS = re.compile(
    r'[\u202A-\u202E\u2066-\u2069]'
)

ZERO_WIDTH_CHARS = re.compile(
    r'[\u200B\u200C\u200D\u2060\uFEFF]'
)

MULTI_WHITESPACE = re.compile(r'\s+')

MAX_LOG_FIELD_LENGTH = 4096

SANITIZATION_METRICS = {
    "control_chars_removed": 0,
    "ansi_sequences_removed": 0,
    "bidi_chars_removed": 0,
    "zero_width_removed": 0,
    "fields_truncated": 0,
}


def _count_matches(pattern, value):
    return len(pattern.findall(value))


def validate_log_integrity(value: str) -> dict:
    """
    Analyze a log field and return security diagnostics.
    Useful for testing and operational visibility.
    """
    if not isinstance(value, str):
        value = str(value)

    return {
        "contains_control_chars":
            bool(CONTROL_CHARS.search(value)),
        "contains_ansi_escape":
            bool(ANSI_ESCAPE.search(value)),
        "contains_bidi_controls":
            bool(BIDI_CONTROL_CHARS.search(value)),
        "contains_zero_width":
            bool(ZERO_WIDTH_CHARS.search(value)),
        "length":
            len(value),
        "sha256":
            hashlib.sha256(
                value.encode("utf-8", errors="ignore")
            ).hexdigest(),
    }


def sanitize_log_field(value: str) -> str:
    """
    Security-hardened log sanitization.

    Protections:
    - ASCII control character removal
    - ANSI escape stripping
    - Unicode normalization
    - Bidirectional override removal
    - Zero-width character removal
    - Multi-line log injection prevention
    - Length limiting
    """

    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    # Normalize Unicode representations
    value = unicodedata.normalize("NFKC", value)

    control_count = _count_matches(
        CONTROL_CHARS,
        value
    )

    ansi_count = _count_matches(
        ANSI_ESCAPE,
        value
    )

    bidi_count = _count_matches(
        BIDI_CONTROL_CHARS,
        value
    )

    zero_width_count = _count_matches(
        ZERO_WIDTH_CHARS,
        value
    )

    value = CONTROL_CHARS.sub(" ", value)
    value = ANSI_ESCAPE.sub("", value)
    value = BIDI_CONTROL_CHARS.sub("", value)
    value = ZERO_WIDTH_CHARS.sub("", value)

    # Collapse multi-line payloads into a single safe line
    value = MULTI_WHITESPACE.sub(
        " ",
        value
    ).strip()

    if len(value) > MAX_LOG_FIELD_LENGTH:
        value = value[:MAX_LOG_FIELD_LENGTH]
        SANITIZATION_METRICS["fields_truncated"] += 1

    SANITIZATION_METRICS[
        "control_chars_removed"
    ] += control_count

    SANITIZATION_METRICS[
        "ansi_sequences_removed"
    ] += ansi_count

    SANITIZATION_METRICS[
        "bidi_chars_removed"
    ] += bidi_count

    SANITIZATION_METRICS[
        "zero_width_removed"
    ] += zero_width_count

    return value


def get_sanitization_metrics():
    """
    Expose sanitization statistics for diagnostics.
    """
    return SANITIZATION_METRICS.copy()