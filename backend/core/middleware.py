import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from .logging_config import set_log_context, clear_log_context


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to add correlation ID and request ID to logs and responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Get or generate correlation ID from request headers
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request_id = str(uuid.uuid4())

        # Store in request state
        request.state.correlation_id = correlation_id
        request.state.request_id = request_id

        # Set log context
        set_log_context(correlation_id=correlation_id, request_id=request_id)

        try:
            response: Response = await call_next(request)
            # Add correlation ID and request ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            # Clear log context
            clear_log_context()
