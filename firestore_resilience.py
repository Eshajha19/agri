"""
Firestore Resilience Layer - Circuit Breaker, Retry Logic, and Connection Pooling

Provides a resilient wrapper around Firestore operations with:
- Exponential backoff with jitter
- Circuit breaker pattern
- Connection health monitoring
- Automatic retry for transient failures
"""

import logging
import time
from enum import Enum
from datetime import datetime, timedelta
from typing import Any, Optional, Callable, List, Dict
import firebase_admin
from firebase_admin import firestore

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"           # Normal operation
    OPEN = "open"               # Failing, reject requests
    HALF_OPEN = "half_open"     # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for Firestore operations"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        expected_exception: Exception = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception(
                    f"Circuit breaker OPEN. Service unavailable for {self.recovery_timeout}s"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker CLOSED - service recovered")
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker OPEN after {self.failure_count} failures"
            )
    
    def _should_attempt_reset(self) -> bool:
        """Check if recovery timeout has elapsed"""
        if not self.last_failure_time:
            return False
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout


class ExponentialBackoff:
    """Exponential backoff with jitter"""
    
    def __init__(self, base: float = 1, max_wait: float = 60):
        self.base = base
        self.max_wait = max_wait
    
    def get_wait_time(self, attempt: int) -> float:
        """Calculate wait time with jitter"""
        import random
        
        # Exponential backoff: base * 2^attempt
        wait = self.base * (2 ** attempt)
        
        # Cap at max_wait
        wait = min(wait, self.max_wait)
        
        # Add jitter (±20%)
        jitter = wait * 0.2 * (2 * random.random() - 1)
        return max(0, wait + jitter)


class FirestoreResilientClient:
    """Resilient Firestore client with retry and circuit breaker"""
    
    def __init__(self, max_retries: int = 3, circuit_breaker_threshold: int = 5):
        self.db = firestore.client()
        self.max_retries = max_retries
        self.backoff = ExponentialBackoff()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            recovery_timeout=30
        )
        self.health_status = {
            "status": "healthy",
            "last_check": datetime.now().isoformat(),
            "consecutive_failures": 0
        }
        self.write_queue: List[Dict[str, Any]] = []
    
    def get(self, collection: str, document: str) -> Optional[Dict]:
        """Get document with retry logic"""
        def _get():
            doc = self.db.collection(collection).document(document).get()
            if doc.exists:
                return doc.to_dict()
            return None
        
        return self._execute_with_retry(_get)
    
    def set(self, collection: str, document: str, data: Dict) -> bool:
        """Set document with retry logic"""
        def _set():
            self.db.collection(collection).document(document).set(
                data, merge=True
            )
            return True
        
        return self._execute_with_retry(_set, collection, document, data)
    
    def add(self, collection: str, data: Dict) -> Optional[str]:
        """Add document with retry logic"""
        def _add():
            _, doc_ref = self.db.collection(collection).add(data)
            return doc_ref.id
        
        return self._execute_with_retry(_add, collection, data)
    
    def delete(self, collection: str, document: str) -> bool:
        """Delete document with retry logic"""
        def _delete():
            self.db.collection(collection).document(document).delete()
            return True
        
        return self._execute_with_retry(_delete, collection, document)
    
    def query(self, collection: str, **filters) -> List[Dict]:
        """Query collection with retry logic"""
        def _query():
            q = self.db.collection(collection)
            
            # Apply filters
            for field, value in filters.items():
                q = q.where(field, "==", value)
            
            docs = q.stream()
            return [doc.to_dict() for doc in docs]
        
        return self._execute_with_retry(_query)
    
    def _execute_with_retry(
        self,
        func: Callable,
        *context_args,
        **context_kwargs
    ) -> Any:
        """Execute operation with retry and circuit breaker"""
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Check circuit breaker
                result = self.circuit_breaker.call(func)
                self.health_status["consecutive_failures"] = 0
                self.health_status["status"] = "healthy"
                return result
            
            except Exception as e:
                last_exception = e
                self.health_status["consecutive_failures"] += 1
                
                # Determine if retryable
                error_msg = str(e).lower()
                is_retryable = any(
                    msg in error_msg for msg in [
                        "deadline exceeded",
                        "unavailable",
                        "temporarily",
                        "connection"
                    ]
                )
                
                if not is_retryable or attempt == self.max_retries - 1:
                    self.health_status["status"] = "unhealthy"
                    logger.error(
                        f"Firestore operation failed after {attempt + 1} attempts: {e}"
                    )
                    raise
                
                # Wait before retry
                wait_time = self.backoff.get_wait_time(attempt)
                logger.warning(
                    f"Firestore operation failed (attempt {attempt + 1}). "
                    f"Retrying in {wait_time:.2f}s: {e}"
                )
                time.sleep(wait_time)
        
        raise last_exception
    
    def queue_write(self, collection: str, document: str, data: Dict):
        """Queue write operation for offline support"""
        self.write_queue.append({
            "collection": collection,
            "document": document,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
        logger.info(f"Queued write operation for {collection}/{document}")
    
    def flush_queue(self) -> int:
        """Attempt to flush queued writes. Returns count of successful flushes"""
        flushed = 0
        failed = []
        
        for op in self.write_queue:
            try:
                self.set(op["collection"], op["document"], op["data"])
                flushed += 1
            except Exception as e:
                logger.warning(f"Failed to flush queued write: {e}")
                failed.append(op)
        
        self.write_queue = failed
        logger.info(f"Flushed {flushed} queued operations")
        return flushed
    
    def get_health_status(self) -> Dict:
        """Get health status"""
        self.health_status["last_check"] = datetime.now().isoformat()
        return self.health_status


# Global resilient client instance
_resilient_client: Optional[FirestoreResilientClient] = None


def get_resilient_client() -> FirestoreResilientClient:
    """Get or create resilient Firestore client"""
    global _resilient_client
    
    if _resilient_client is None:
        _resilient_client = FirestoreResilientClient(max_retries=3)
    
    return _resilient_client
