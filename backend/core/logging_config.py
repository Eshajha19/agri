import logging
from contextvars import ContextVar

# Thread-safe / async-safe context storage
log_context: ContextVar[dict] = ContextVar("log_context", default={})


class ContextFilter(logging.Filter):
    """Inject contextual information into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = log_context.get()

        record.context = (
            ", ".join(f"{k}={v}" for k, v in context.items())
            if context
            else "-"
        )
        return True


def set_log_context(**kwargs):
    """Add or update logging context."""
    current = log_context.get().copy()
    current.update(kwargs)
    log_context.set(current)


def clear_log_context():
    """Clear logging context."""
    log_context.set({})


def setup_logging(
    name: str = __name__,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a logger."""

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "%(funcName)s:%(lineno)d | %(context)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)
    handler.addFilter(ContextFilter())

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    return logger
