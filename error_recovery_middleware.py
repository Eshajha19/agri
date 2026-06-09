"""
Error Recovery Middleware for FastAPI
Provides structured error handling and recovery for async operations
"""

import logging
import time
import uuid
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException
import traceback
from typing import Dict

logger = logging.getLogger(__name__)


class ErrorRecoveryMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling errors with structured recovery
    
    Features:
    - Automatic error tracking
    - Structured error responses
    - Request/response logging
    - Circuit breaker integration
    - Rate-limited log emission per endpoint per window
    """
    
    def __init__(self, app, log_cooldown: float = 5.0):
        super().__init__(app)
        self.error_counts = {}  # endpoint -> count
        self.error_timestamps = {}  # endpoint -> timestamp
        self._log_cooldown = log_cooldown  # seconds between same-category logs per endpoint
        self._log_last_emitted: Dict[str, float] = {}  # key -> last log time
        self._log_suppressed: Dict[str, int] = {}  # key -> count of suppressed logs
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Handle request with error recovery"""
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Track timing
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"
        
        try:
            # Call the endpoint
            response = await call_next(request)
            
            # Log successful request
            duration = time.time() - start_time
            self._rate_log(
                f"info:{endpoint}", logging.INFO,
                "[%s] %s - Status: %s - Duration: %.2fs",
                request_id, endpoint, response.status_code, duration,
            )
            
            # Reset error count on success
            if endpoint in self.error_counts:
                self.error_counts[endpoint] = 0
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
        
        except HTTPException as http_exc:
            """Handle HTTP exceptions"""
            duration = time.time() - start_time
            
            self._rate_log(
                f"http_error:{endpoint}", logging.WARNING,
                "[%s] %s - HTTP Error: %s - Detail: %s - Duration: %.2fs",
                request_id, endpoint, http_exc.status_code, http_exc.detail, duration,
            )
            
            return JSONResponse(
                status_code=http_exc.status_code,
                content={
                    "success": False,
                    "request_id": request_id,
                    "error": {
                        "message": http_exc.detail,
                        "status_code": http_exc.status_code,
                        "category": self._categorize_error(http_exc.status_code)
                    }
                }
            )
        
        except ValueError as val_exc:
            """Handle validation errors"""
            duration = time.time() - start_time
            
            self._rate_log(
                f"validation_error:{endpoint}", logging.WARNING,
                "[%s] %s - Validation Error: %s - Duration: %.2fs",
                request_id, endpoint, val_exc, duration,
            )
            
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "request_id": request_id,
                    "error": {
                        "message": str(val_exc),
                        "status_code": 400,
                        "category": "validation"
                    }
                }
            )
        
        except TimeoutError as timeout_exc:
            """Handle timeout errors"""
            duration = time.time() - start_time
            
            self._rate_log(
                f"timeout:{endpoint}", logging.ERROR,
                "[%s] %s - Timeout Error - Duration: %.2fs",
                request_id, endpoint, duration,
            )
            
            # Increment error count
            self.error_counts[endpoint] = self.error_counts.get(endpoint, 0) + 1
            
            return JSONResponse(
                status_code=504,
                content={
                    "success": False,
                    "request_id": request_id,
                    "error": {
                        "message": "Request timeout - please try again",
                        "status_code": 504,
                        "category": "network",
                        "recoverable": True
                    }
                }
            )
        
        except Exception as exc:
            """Handle unexpected errors"""
            duration = time.time() - start_time
            error_id = str(uuid.uuid4())
            
            self._rate_log(
                f"unexpected:{endpoint}", logging.ERROR,
                "[%s] %s - Unexpected Error [%s]: %s - Duration: %.2fs\n%s",
                request_id, endpoint, error_id, exc, duration, traceback.format_exc(),
            )
            
            # Increment error count
            self.error_counts[endpoint] = self.error_counts.get(endpoint, 0) + 1
            
            # Check circuit breaker
            is_broken = self._check_circuit_breaker(endpoint)
            
            if is_broken:
                return JSONResponse(
                    status_code=503,
                    content={
                        "success": False,
                        "request_id": request_id,
                        "error": {
                            "message": "Service temporarily unavailable",
                            "status_code": 503,
                            "category": "service_error",
                            "recoverable": True,
                            "error_id": error_id
                        }
                    }
                )
            
            # Return generic error
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "request_id": request_id,
                    "error": {
                        "message": "An unexpected error occurred",
                        "status_code": 500,
                        "category": "unknown",
                        "error_id": error_id,
                        "recoverable": True
                    }
                }
            )
    
    def _categorize_error(self, status_code: int) -> str:
        """Categorize HTTP error"""
        if 400 <= status_code < 500:
            if status_code == 401:
                return "authentication"
            elif status_code == 403:
                return "authorization"
            elif status_code == 404:
                return "not_found"
            else:
                return "client_error"
        elif status_code >= 500:
            return "server_error"
        return "unknown"
    
    def _check_circuit_breaker(self, endpoint: str) -> bool:
        """Check if circuit breaker should open"""
        
        # Get error count and last error time
        error_count = self.error_counts.get(endpoint, 0)
        error_time = self.error_timestamps.get(endpoint, time.time())
        
        # Update timestamp
        self.error_timestamps[endpoint] = time.time()
        
        # Open circuit if more than 5 errors in 60 seconds
        time_since_error = time.time() - error_time
        
        if error_count >= 5 and time_since_error < 60:
            self._rate_log(
                f"circuit_breaker:{endpoint}", logging.WARNING,
                "Circuit breaker opened for %s: %d errors in %.0fs",
                endpoint, error_count, time_since_error,
            )
            return True
        
        # Reset if 60 seconds have passed
        if time_since_error >= 60:
            self.error_counts[endpoint] = 0
        
        return False
    
    def _rate_log(self, key: str, level: int, msg: str, *args):
        """Emit log at most once per ``_log_cooldown`` seconds per key.
        
        Suppressed calls increment a counter; periodic summary lines are
        emitted so that suppressed events remain measurable.
        """
        now = time.time()
        last = self._log_last_emitted.get(key, 0.0)
        if now - last >= self._log_cooldown:
            self._log_last_emitted[key] = now
            suppressed = self._log_suppressed.pop(key, 0)
            if suppressed:
                logger.log(level, "%s — (%d similar messages suppressed in the last %.0fs)", msg, suppressed, self._log_cooldown, *args)
            else:
                logger.log(level, msg, *args)
        else:
            self._log_suppressed[key] = self._log_suppressed.get(key, 0) + 1

    def get_error_stats(self) -> dict:
        """Get error statistics"""
        return {
            "endpoints_with_errors": len(self.error_counts),
            "error_counts": self.error_counts,
            "timestamps": {
                k: v for k, v in self.error_timestamps.items()
            },
            "log_suppressed": dict(self._log_suppressed),
        }


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request IDs to all requests"""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Add request ID to all responses"""
        
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


async def error_logger_callback(error_context):
    """Callback for logging errors from async error handler"""
    logger.log(
        logging.WARNING if error_context.severity.value == "medium" else logging.ERROR,
        f"Error [{error_context.error_id}] in {error_context.source}: "
        f"{error_context.message}"
    )
