import logging

from slowapi.errors import RateLimitExceeded

from rate_limit_config import (
    build_limiter,
    rate_limit_exceeded_handler,
)


logger = logging.getLogger(__name__)


def setup_rate_limiter(app):
    limiter = build_limiter(
        default_limits=["120/minute"]
    )

    app.state.limiter = limiter

    app.add_exception_handler(
        RateLimitExceeded,
        rate_limit_exceeded_handler,
    )

    original_limit = limiter.limit

    def safe_limit(rate):
        def decorator(fn):
            try:
                return original_limit(rate)(fn)

            except Exception as exc:
                logger.error(
                    "Rate limit decorator failed for rate=%s on %s: %s. Endpoint is UNPROTECTED.",
                    rate,
                    getattr(fn, "__name__", fn),
                    exc,
                )
                return fn

        return decorator

    limiter.limit = safe_limit

    return limiter
