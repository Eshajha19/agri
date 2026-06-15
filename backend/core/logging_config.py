import logging
from contextvars import ContextVar
from pythonjsonlogger import jsonlogger
from typing import Optional, Dict, Any

try:
    from opentelemetry import trace
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

# Thread-safe / async-safe context storage
log_context: ContextVar[dict] = ContextVar("log_context", default={})


class ContextFilter(logging.Filter):
    """Inject contextual information into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = log_context.get()

        # Add context fields directly to the record for JSON formatter
        for k, v in context.items():
            setattr(record, k, v)

        # Add OpenTelemetry trace ID if available
        if HAS_OPENTELEMETRY:
            try:
                span = trace.get_current_span()
                if span.is_recording():
                    trace_id = format(span.get_span_context().trace_id, "016x")
                    record.trace_id = trace_id
            except Exception:
                pass

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
    json_format: bool = True,
) -> logging.Logger:
    """Configure and return a logger."""

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()

    if json_format:
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(funcName)s %(lineno)d %(message)s",
            rename_fields={"levelname": "level"},
            timestamp=True
        )
    else:
        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s | %(levelname)-8s | %(name)s | "
                "%(funcName)s:%(lineno)d | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    handler.addFilter(ContextFilter())

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    return logger
