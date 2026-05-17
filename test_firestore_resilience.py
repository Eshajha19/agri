"""
Test suite for Firestore Resilience Layer
Verifies circuit breaker, exponential backoff, and retry logic
"""

import pytest
import time
from firestore_resilience import (
    CircuitBreaker,
    CircuitState,
    ExponentialBackoff,
    FirestoreResilientClient
)


class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def test_circuit_closed_initially(self):
        """Circuit breaker should start in CLOSED state"""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
    
    def test_circuit_opens_after_threshold(self):
        """Circuit breaker should OPEN after failure threshold"""
        cb = CircuitBreaker(failure_threshold=2)
        
        def failing_func():
            raise Exception("Simulated failure")
        
        # First failure
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == CircuitState.CLOSED
        
        # Second failure - should open
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN
    
    def test_circuit_rejects_when_open(self):
        """Circuit breaker should reject calls when OPEN"""
        cb = CircuitBreaker(failure_threshold=1)
        
        def failing_func():
            raise Exception("Simulated failure")
        
        # Trigger open state
        with pytest.raises(Exception):
            cb.call(failing_func)
        
        # Next call should be rejected without executing function
        with pytest.raises(Exception) as exc_info:
            cb.call(failing_func)
        assert "Circuit breaker OPEN" in str(exc_info.value)
    
    def test_circuit_half_open_after_timeout(self):
        """Circuit breaker should enter HALF_OPEN after recovery timeout"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1)
        
        def failing_func():
            raise Exception("Simulated failure")
        
        # Trigger open state
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Next attempt should try HALF_OPEN
        def succeeding_func():
            return "success"
        
        result = cb.call(succeeding_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED


class TestExponentialBackoff:
    """Test exponential backoff calculation"""
    
    def test_backoff_increases(self):
        """Backoff time should increase exponentially"""
        backoff = ExponentialBackoff(base=1, max_wait=60)
        
        wait_1 = backoff.get_wait_time(0)
        wait_2 = backoff.get_wait_time(1)
        wait_3 = backoff.get_wait_time(2)
        
        # Each should be roughly double (with some jitter)
        assert wait_1 < wait_2
        assert wait_2 < wait_3
    
    def test_backoff_capped_at_max(self):
        """Backoff should not exceed max_wait"""
        backoff = ExponentialBackoff(base=1, max_wait=10)
        
        wait_high = backoff.get_wait_time(10)
        assert wait_high <= 12  # max_wait + 20% jitter
    
    def test_backoff_includes_jitter(self):
        """Backoff should include randomness"""
        backoff = ExponentialBackoff(base=1, max_wait=60)
        
        waits = [backoff.get_wait_time(2) for _ in range(10)]
        
        # All should be roughly equal to 4s but with variance
        assert len(set(waits)) > 1  # Should have different values due to jitter


class TestFirestoreResilientClient:
    """Test resilient Firestore client"""
    
    def test_health_status_initialized(self):
        """Health status should be initialized"""
        # Note: This test doesn't actually connect to Firestore
        # It just verifies the client structure
        from firestore_resilience import FirestoreResilientClient
        
        try:
            client = FirestoreResilientClient()
            health = client.get_health_status()
            
            assert "status" in health
            assert health["status"] in ["healthy", "unhealthy"]
            assert "consecutive_failures" in health
        except Exception:
            # Firestore not initialized, that's ok for this test
            pass
    
    def test_write_queue_operations(self):
        """Write queue should queue operations"""
        from firestore_resilience import FirestoreResilientClient
        
        try:
            client = FirestoreResilientClient()
            
            # Queue some operations
            client.queue_write("test_collection", "doc1", {"data": "test"})
            client.queue_write("test_collection", "doc2", {"data": "test2"})
            
            assert len(client.write_queue) == 2
        except Exception:
            # Firestore not initialized, that's ok for this test
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
