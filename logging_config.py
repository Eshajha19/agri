"""
Production logging configuration for Render deployment.

Provides structured JSON logging with request context,
standardized error formatting, and log levels compatible
with Render's log aggregation.
"""
import logging
import logging.config
import json
import sys
import time
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        structured_fields = (
            "component",
            "status",
            "phase",
            "duration_ms",
            "status_code",
            "method",
            "path",
            "client_ip",
            "user_agent",
            "error_type",
        )
        for key in structured_fields:
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        msg = (
            f"{color}{timestamp} [{record.levelname}]{self.RESET} "
            f"{record.name}: {record.getMessage()}"
        )
        context = []
        for key in ("component", "status", "phase", "duration_ms", "error_type"):
            if hasattr(record, key):
                context.append(f"{key}={getattr(record, key)}")
        if context:
            msg += f" ({', '.join(context)})"
        if record.exc_info and record.exc_info[0] is not None:
            msg += f"\n{self.formatException(record.exc_info)}"
        return msg


def setup_logging(level: str = "INFO", json_output: bool | None = None):
    """
    Configure structured logging for the application.

    Args:
        level: Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: Force JSON (True) or console (False) format.
                     If None, auto-detect based on RENDER env var.
    """
    if json_output is None:
        json_output = bool(json.loads(os.environ.get("RENDER", "false"))) or \
            os.environ.get("LOG_FORMAT", "").lower() == "json"

    formatter_name = "json" if json_output else "console"

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {"()": "logging_config.JSONFormatter"},
            "console": {"()": "logging_config.ConsoleFormatter"},
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": formatter_name,
            },
            "stderr": {
                "class": "logging.StreamHandler",
                "stream": sys.stderr,
                "formatter": formatter_name,
                "level": "WARNING",
            },
        },
        "root": {
            "level": level,
            "handlers": ["stdout", "stderr"],
        },
        "loggers": {
            "uvicorn": {"level": "INFO", "propagate": False},
            "uvicorn.access": {"level": "WARNING", "propagate": False},
            "uvicorn.error": {"level": "INFO", "propagate": False},
            "fastapi": {"level": "INFO", "propagate": False},
            "firebase_admin": {"level": "WARNING", "propagate": False},
            "tensorflow": {"level": "WARNING", "propagate": False},
            "google": {"level": "WARNING", "propagate": False},
            "httpx": {"level": "WARNING", "propagate": False},
            "httpcore": {"level": "WARNING", "propagate": False},
        },
    }

    logging.config.dictConfig(config)


class RequestLoggingMiddleware:
    """ASGI middleware that logs every request with timing and status."""

    def __init__(self, app):
        self.app = app
        self.logger = logging.getLogger("request")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        method = scope.get("method", "")
        path = scope.get("path", "")
        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            status_code = 500
            raise
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            client = scope.get("client")
            client_ip = client[0] if client else "unknown"
            headers = dict(scope.get("headers", []))
            user_agent = headers.get(b"user-agent", b"").decode("utf-8", errors="replace")

            extra = {
                "duration_ms": duration_ms,
                "status_code": status_code,
                "method": method,
                "path": path,
                "client_ip": client_ip,
                "user_agent": user_agent,
            }

            if status_code >= 500:
                self.logger.error("%s %s -> %d (%sms)", method, path, status_code, duration_ms, extra=extra)
            elif status_code >= 400:
                self.logger.warning("%s %s -> %d (%sms)", method, path, status_code, duration_ms, extra=extra)
            else:
                self.logger.info("%s %s -> %d (%sms)", method, path, status_code, duration_ms, extra=extra)


import os  # noqa: E402 (imported at module level for json_output detection)
