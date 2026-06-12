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
from async_error_handler import CircuitBreakerAsync

logger = logging.getLogger(__name__)


class ErrorRecoveryMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling errors with structured recovery
    
    Features:
    - Automatic error tracking
    - Structured error responses
    - Request/response logging
    - Circuit breaker integration
    """
    
    def __init__(self, app, circuit_breaker=None):
        super().__init__(app)
        self.circuit_breaker = circuit_breaker or CircuitBreakerAsync(
            failure_threshold=5,
            recovery_timeout=60,
            name="middleware"
        )
    
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
            logger.info(
                f"[{request_id}] {endpoint} - Status: {response.status_code} - "
                f"Duration: {duration:.2f}s"
            )
            
            # Reset circuit breaker on 2xx responses
            if 200 <= response.status_code < 300:
                self.circuit_breaker.record_success()
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
        
        except HTTPException as http_exc:
            """Handle HTTP exceptions"""
            duration = time.time() - start_time
            
            logger.warning(
                f"[{request_id}] {endpoint} - HTTP Error: {http_exc.status_code} - "
                f"Detail: {http_exc.detail} - Duration: {duration:.2f}s"
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
            
            logger.warning(
                f"[{request_id}] {endpoint} - Validation Error: {val_exc} - "
                f"Duration: {duration:.2f}s"
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
            
            logger.error(
                f"[{request_id}] {endpoint} - Timeout Error - Duration: {duration:.2f}s"
            )
            
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
            
            logger.error(
                f"[{request_id}] {endpoint} - Unexpected Error [{error_id}]: {exc} - "
                f"Duration: {duration:.2f}s\n{traceback.format_exc()}"
            )
            
            # Track failure with circuit breaker
            is_broken = self.circuit_breaker.record_failure()
            
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
    
    def get_error_stats(self) -> dict:
        """Get error statistics"""
        status = self.circuit_breaker.get_status()
        return {
            "circuit_breaker": status,
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
