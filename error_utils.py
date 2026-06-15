"""Safe error detail helpers — log full exception details server-side, return generic client-safe messages."""

import logging
import traceback

logger = logging.getLogger(__name__)

_SAFE_MESSAGES = {
    400: "Invalid request",
    403: "Access denied",
    404: "Resource not found",
    422: "Validation failed",
    429: "Too many requests",
    500: "Internal server error",
    503: "Service unavailable",
}


def safe_detail(e: Exception, status_code: int = 400, request_id: str = "") -> str:
    """Log full exception, return generic message safe for client responses."""
    logger.error(
        "Request %s failed with %s: %s\n%s",
        request_id, type(e).__name__, e, traceback.format_exc(),
    )
    return _SAFE_MESSAGES.get(status_code, "An error occurred")
