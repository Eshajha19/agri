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
    
    def __init__(self, app):
        super().__init__(app)
        self.error_counts = {}  # endpoint -> count
        self.error_timestamps = {}  # endpoint -> timestamp
    
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
            
            # Reset error count on success
            if endpoint in self.error_counts:
                self.error_counts[endpoint] = 0
            
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
                        "message": "Invalid request",
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
            
            logger.error(
                f"[{request_id}] {endpoint} - Unexpected Error [{error_id}]: {exc} - "
                f"Duration: {duration:.2f}s\n{traceback.format_exc()}"
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
            logger.warning(
                f"Circuit breaker opened for {endpoint}: "
                f"{error_count} errors in {time_since_error:.0f}s"
            )
            return True
        
        # Reset if 60 seconds have passed
        if time_since_error >= 60:
            self.error_counts[endpoint] = 0
        
        return False
    
    def get_error_stats(self) -> dict:
        """Get error statistics"""
        return {
            "endpoints_with_errors": len(self.error_counts),
            "error_counts": self.error_counts,
            "timestamps": {
                k: v for k, v in self.error_timestamps.items()
            }
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
