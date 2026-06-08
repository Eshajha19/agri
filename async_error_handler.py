"""
Async Error Boundary & Structured Error Recovery System
Handles errors in async operations with recovery strategies and monitoring
"""

import logging
import asyncio
import random
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, TypeVar, Coroutine, Awaitable, Union
from dataclasses import dataclass
import traceback
import json

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    NETWORK = "network"
    DATABASE = "database"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for an error"""
    error_id: str
    timestamp: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    source: str  # Function/endpoint name
    stack_trace: str
    context_data: Dict[str, Any]
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "source": self.source,
            "stack_trace": self.stack_trace,
            "context_data": self.context_data,
            "user_id": self.user_id,
            "request_id": self.request_id
        }


@dataclass
class RecoveryStrategy:
    """Strategy for recovering from an error"""
    name: str
    retry_count: int = 0
    max_retries: int = 3
    backoff_multiplier: float = 2.0
    timeout: int = 30
    fallback_value: Any = None
    should_log: bool = True


class AsyncErrorHandler:
    """Handles errors in async operations"""
    
    def __init__(self, max_error_history: int = 1000):
        self.error_history: List[ErrorContext] = []
        self.max_error_history = max_error_history
        self.active_recoveries: Dict[str, RecoveryStrategy] = {}
        self.error_callbacks: List[Callable[[ErrorContext], None]] = []
        self.recovery_callbacks: List[Callable[[str, str], None]] = []
    
    def classify_error(self, error: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """Classify error type and severity"""
        error_type = type(error).__name__
        error_str = str(error).lower()
        
        # Database errors (check first, before network)
        if any(x in error_type for x in ['Database', 'Firestore', 'Query']):
            return ErrorCategory.DATABASE, ErrorSeverity.HIGH
        if any(x in error_str for x in ['database', 'firestore', 'query']):
            return ErrorCategory.DATABASE, ErrorSeverity.HIGH
        
        # Network errors
        if any(x in error_type for x in ['Connection', 'Timeout', 'Network']):
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
        if any(x in error_str for x in ['timeout', 'network']):
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
        
        # Validation errors
        if 'Value' in error_type or 'Type' in error_type:
            return ErrorCategory.VALIDATION, ErrorSeverity.LOW
        if 'Validation' in error_type or 'Invalid' in error_type:
            return ErrorCategory.VALIDATION, ErrorSeverity.LOW
        if any(x in error_str for x in ['validation', 'invalid', 'value error']):
            return ErrorCategory.VALIDATION, ErrorSeverity.LOW
        
        # Authentication errors
        if 'Auth' in error_type or 'Unauthorized' in error_type:
            return ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH
        if any(x in error_str for x in ['auth', 'unauthorized']):
            return ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH
        
        # Authorization errors
        if 'Forbidden' in error_type or 'Permission' in error_type:
            return ErrorCategory.AUTHORIZATION, ErrorSeverity.MEDIUM
        if any(x in error_str for x in ['forbidden', 'permission']):
            return ErrorCategory.AUTHORIZATION, ErrorSeverity.MEDIUM
        
        # External service errors
        if 'API' in error_type or 'External' in error_type:
            return ErrorCategory.EXTERNAL_SERVICE, ErrorSeverity.MEDIUM
        if any(x in error_str for x in ['api', 'external', 'service']):
            return ErrorCategory.EXTERNAL_SERVICE, ErrorSeverity.MEDIUM
        
        return ErrorCategory.UNKNOWN, ErrorSeverity.CRITICAL
    
    def record_error(
        self,
        error: Exception,
        source: str,
        context_data: Dict = None,
        user_id: str = None,
        request_id: str = None
    ) -> ErrorContext:
        """Record an error with context"""
        import uuid
        
        category, severity = self.classify_error(error)
        
        error_context = ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            message=str(error),
            category=category,
            severity=severity,
            source=source,
            stack_trace=traceback.format_exc(),
            context_data=context_data or {},
            user_id=user_id,
            request_id=request_id
        )
        
        # Keep history strictly bounded — trim before append so the list
        # never temporarily exceeds max_error_history during bursts.
        self.error_history = (self.error_history + [error_context])[-self.max_error_history:]
        
        # Call error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_context)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
        
        # Log error
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(severity, logging.ERROR)
        
        logger.log(
            log_level,
            f"Error [{error_context.error_id}] in {source}: {error_context.message}"
        )
        
        return error_context
    
    async def with_recovery(
        self,
        coro: Union[Coroutine[Any, Any, T], Callable[[], Awaitable[T]]],
        source: str,
        strategy: RecoveryStrategy = None,
        context_data: Dict = None,
        user_id: str = None,
        request_id: str = None
    ) -> tuple[Optional[T], Optional[ErrorContext]]:
        """
        Execute coroutine with error recovery
        
        Returns:
            (result, error_context) - if error, result is strategy.fallback_value
        """
        
        if strategy is None:
            strategy = RecoveryStrategy(name="default")
        
        last_error = None
        
        for attempt in range(strategy.max_retries + 1):
            try:
                # If a callable is provided, create a fresh coroutine per attempt.
                # This avoids "cannot reuse already awaited coroutine" on retries.
                if callable(coro):
                    current_coro = coro()
                else:
                    if attempt > 0:
                        raise RuntimeError(
                            "Retries require a coroutine factory (callable), "
                            "not a single coroutine instance"
                        )
                    current_coro = coro

                result = await asyncio.wait_for(current_coro, timeout=strategy.timeout)
                
                # Success
                if attempt > 0:
                    logger.info(f"{source} succeeded on retry {attempt}")

                    for callback in self.recovery_callbacks:
                        try:
                            callback(source, "recovered")
                        except Exception:
                            pass
                
                return result, None
                
            except asyncio.TimeoutError as e:
                last_error = e
                if attempt < strategy.max_retries:
                    wait_time = (2 ** attempt * strategy.backoff_multiplier) + random.uniform(0, 1)
                    logger.warning(
                        f"{source} timed out, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{strategy.max_retries + 1})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"{source} failed after {strategy.max_retries + 1} attempts")
            
            except Exception as e:
                last_error = e
                category, _ = self.classify_error(e)
                if category in (ErrorCategory.VALIDATION, ErrorCategory.AUTHENTICATION, ErrorCategory.AUTHORIZATION):
                    logger.warning(f"{source} failed with non-retryable error: {e}")
                    break
                
                # Check if retryable
                if attempt < strategy.max_retries:
                    wait_time = (2 ** attempt * strategy.backoff_multiplier) + random.uniform(0, 1)
                    logger.warning(
                        f"{source} failed: {e}, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{strategy.max_retries + 1})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"{source} failed after {strategy.max_retries + 1} attempts: {e}")
                    break
        
        # All retries exhausted
        error_context = self.record_error(
            last_error or Exception("Unknown error"),
            source,
            context_data,
            user_id,
            request_id
        )
        for callback in self.recovery_callbacks:
            try:
                callback(source, "failed")
            except Exception:
                pass
        
        return strategy.fallback_value, error_context
    
    def add_error_callback(self, callback: Callable[[ErrorContext], None]):
        """Register a callback for error events"""
        self.error_callbacks.append(callback)
    
    def remove_error_callback(self, callback: Callable[[ErrorContext], None]):
        """Remove error callback"""
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)
    
    def add_recovery_callback(
        self,
        callback: Callable[[str, str], None]
    ):
        """Register recovery monitoring callback"""
        self.recovery_callbacks.append(callback)

    def remove_recovery_callback(
        self,
        callback: Callable[[str, str], None]
    ):
        """Remove recovery monitoring callback"""
        if callback in self.recovery_callbacks:
            self.recovery_callbacks.remove(callback)
    
    def get_error_history(
        self,
        limit: int = 100,
        severity: ErrorSeverity = None,
        category: ErrorCategory = None
    ) -> List[ErrorContext]:
        """Get error history with optional filtering"""
        
        history = self.error_history[-limit:]
        
        if severity:
            history = [e for e in history if e.severity == severity]
        
        if category:
            history = [e for e in history if e.category == category]
        
        return history
    
    def get_error_stats(self) -> Dict:
        """Get error statistics"""
        
        if not self.error_history:
            return {"total": 0, "by_category": {}, "by_severity": {}}
        
        by_category = {}
        by_severity = {}
        
        for error in self.error_history:
            # Count by category
            cat = error.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
            
            # Count by severity
            sev = error.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1
        
        return {
            "total": len(self.error_history),
            "by_category": by_category,
            "by_severity": by_severity,
            "recent_errors": [e.to_dict() for e in self.error_history[-10:]]
        }
    
    def clear_history(self):
        """Clear error history"""
        self.error_history.clear()
    
    def export_errors(self) -> str:
        """Export error history as JSON"""
        return json.dumps(
            [e.to_dict() for e in self.error_history],
            indent=2
        )


class CircuitBreakerAsync:
    """Circuit breaker for async operations.

    Callers MUST pass a coroutine factory (a zero-argument callable that
    returns a fresh coroutine each time it is called) rather than a bare
    coroutine object.  This is required because:

    1. A coroutine object can only be awaited once.  If the circuit is open
       and the coroutine is rejected without being awaited, the object must
       be explicitly closed to release its frame resources and suppress the
       ``RuntimeWarning: coroutine '...' was never awaited`` warning.
       Passing a factory means the circuit breaker never even creates the
       coroutine when the circuit is open, so there is nothing to leak.

    2. The previous API accepted a raw ``Coroutine`` object.  When the
       circuit was open it called ``coro.close()`` to suppress the warning,
       but ``close()`` sends a ``GeneratorExit`` into the coroutine frame
       which can raise ``RuntimeError`` if the coroutine has already started
       executing (e.g. in a half-open retry scenario).  Using a factory
       avoids this entirely.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    async def execute(
        self,
        coro_factory: Callable[[], Coroutine[Any, Any, T]],
    ) -> tuple[Optional[T], bool]:
        """
        Execute a coroutine produced by *coro_factory* with circuit-breaker
        protection.

        Parameters
        ----------
        coro_factory:
            A zero-argument callable that returns a **fresh** coroutine each
            time it is called.  Example::

                await cb.execute(lambda: my_async_fn(arg1, arg2))

        Returns
        -------
        (result, is_healthy)
            *result* is ``None`` and *is_healthy* is ``False`` when the
            circuit is open or the execution raised an exception.
        """
        # When the circuit is open, reject immediately without creating a
        # coroutine at all — nothing to close, nothing to leak.
        if self.state == "open":
            if self._should_attempt_recovery():
                self.state = "half_open"
            else:
                logger.debug(
                    "Circuit breaker %s is open — request rejected", self.name
                )
                return None, False

        # Create a fresh coroutine from the factory for this attempt.
        coro = coro_factory()
        try:
            result = await coro
            self._on_success()
            return result, True
        except Exception as e:
            self._on_failure()
            logger.warning("Circuit breaker %s caught exception: %s", self.name, e)
            # Explicitly close the coroutine in case it was only partially
            # driven before the exception propagated, ensuring frame cleanup.
            try:
                coro.close()
            except Exception as e:
            import logging
            logging.error(f"Async error: {e}")
                pass
            return None, False
    
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == "half_open":
            # Standard circuit breaker: a single half-open probe failure
            # immediately reopens the circuit to protect the backend.
            self.state = "open"
        elif self.failure_count >= self.failure_threshold:
            self.state = "open"

        if self.state == "open":
            logger.warning(
                f"Circuit breaker {self.name} opened after "
                f"{self.failure_count} failures"
            )
    
    def _should_attempt_recovery(self) -> bool:
        """Check if should attempt recovery from open state"""
        if not self.last_failure_time:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout
    
    def get_status(self) -> Dict:
        """Get circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


# Global error handler
_error_handler: Optional[AsyncErrorHandler] = None


def get_error_handler() -> AsyncErrorHandler:
    """Get or create global error handler"""
    global _error_handler
    
    if _error_handler is None:
        _error_handler = AsyncErrorHandler()
    
    return _error_handler
