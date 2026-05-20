"""
End-to-end verification for Async Error Boundary & Structured Error Recovery
"""

import sys
import asyncio
from async_error_handler import (
    AsyncErrorHandler, CircuitBreakerAsync, ErrorSeverity, ErrorCategory,
    RecoveryStrategy, get_error_handler
)


def verify_error_classification():
    """Verify error classification system"""
    print("\n=== Verifying Error Classification ===")
    
    handler = AsyncErrorHandler()
    
    # Test network error
    category, severity = handler.classify_error(ConnectionError("Network failed"))
    assert category == ErrorCategory.NETWORK
    assert severity == ErrorSeverity.MEDIUM
    print(f"[OK] Network error classification: {category.value} ({severity.value})")
    
    # Test database error
    category, severity = handler.classify_error(Exception("Database query failed"))
    assert category == ErrorCategory.DATABASE
    assert severity == ErrorSeverity.HIGH
    print(f"[OK] Database error classification: {category.value} ({severity.value})")
    
    # Test validation error
    category, severity = handler.classify_error(ValueError("Invalid input"))
    assert category == ErrorCategory.VALIDATION
    assert severity == ErrorSeverity.LOW
    print(f"[OK] Validation error classification: {category.value} ({severity.value})")
    
    return True


def verify_error_recording():
    """Verify error recording system"""
    print("\n=== Verifying Error Recording ===")
    
    handler = AsyncErrorHandler()
    
    # Record errors
    for i in range(5):
        error_context = handler.record_error(
            ValueError(f"Test error {i}"),
            source=f"function_{i}",
            context_data={"attempt": i},
            user_id=f"user_{i}"
        )
        assert error_context.error_id is not None
    
    print(f"[OK] Recorded {len(handler.error_history)} errors")
    
    # Get stats
    stats = handler.get_error_stats()
    assert stats["total"] == 5
    assert stats["by_category"]["validation"] == 5
    print(f"[OK] Error statistics: {stats['total']} total errors")
    
    # Filter by severity
    low_errors = handler.get_error_history(severity=ErrorSeverity.LOW)
    assert len(low_errors) == 5
    print(f"[OK] Error filtering: {len(low_errors)} low-severity errors")
    
    return True


async def verify_async_recovery():
    """Verify async error recovery"""
    print("\n=== Verifying Async Error Recovery ===")
    
    handler = AsyncErrorHandler()
    
    # Test 1: Successful execution
    async def success():
        return "success"
    
    result, error = await handler.with_recovery(
        success(),
        source="test_success"
    )
    assert result == "success"
    assert error is None
    print("[OK] Successful execution: no errors")
    
    # Test 2: Recovery with retries
    attempt = 0
    
    async def retry_until_success():
        nonlocal attempt
        attempt += 1
        if attempt < 3:
            raise ValueError("Temporary failure")
        return "recovered"
    
    strategy = RecoveryStrategy(
        name="retry_test",
        max_retries=3,
        backoff_multiplier=0.01
    )
    
    attempt = 0
    result, error = await handler.with_recovery(
        retry_until_success,
        source="test_retry",
        strategy=strategy
    )
    assert result == "recovered"
    assert error is None
    assert attempt == 3
    print(f"[OK] Retry recovery: succeeded on attempt {attempt}")
    
    # Test 3: Fallback value
    async def always_fails():
        raise ValueError("Permanent failure")
    
    strategy = RecoveryStrategy(
        name="fallback_test",
        max_retries=1,
        fallback_value="fallback_data",
        backoff_multiplier=0.01
    )
    
    result, error = await handler.with_recovery(
        always_fails,
        source="test_fallback",
        strategy=strategy
    )
    assert result == "fallback_data"
    assert error is not None
    print("[OK] Fallback value: used after retries exhausted")
    
    return True


async def verify_circuit_breaker():
    """Verify circuit breaker"""
    print("\n=== Verifying Circuit Breaker ===")
    
    breaker = CircuitBreakerAsync(
        failure_threshold=2,
        recovery_timeout=1,
        name="test_breaker"
    )
    
    # Test 1: Closed state with success
    async def success():
        return "ok"
    
    result, healthy = await breaker.execute(success())
    assert healthy is True
    assert breaker.state == "closed"
    print("[OK] Circuit breaker closed: successful execution")
    
    # Test 2: Open state after failures
    async def fail():
        raise ValueError("Error")
    
    result1, healthy1 = await breaker.execute(fail())
    result2, healthy2 = await breaker.execute(fail())
    
    assert breaker.state == "open"
    assert breaker.failure_count == 2
    print("[OK] Circuit breaker opened: after threshold failures")
    
    # Test 3: Recovery
    await asyncio.sleep(1.1)
    result, healthy = await breaker.execute(success())
    assert healthy is True
    assert breaker.state == "closed"
    print("[OK] Circuit breaker recovered: half-open -> closed")
    
    # Test 4: Status
    status = breaker.get_status()
    assert status["name"] == "test_breaker"
    assert status["state"] == "closed"
    print(f"[OK] Circuit breaker status: {status['state']}")
    
    return True


def verify_error_callbacks():
    """Verify error callbacks"""
    print("\n=== Verifying Error Callbacks ===")
    
    handler = AsyncErrorHandler()
    callback_count = 0
    
    def test_callback(error_context):
        nonlocal callback_count
        callback_count += 1
    
    handler.add_error_callback(test_callback)
    
    # Record an error
    handler.record_error(ValueError("Test"), source="test")
    
    assert callback_count == 1
    print("[OK] Error callback triggered")
    
    # Remove callback
    handler.remove_error_callback(test_callback)
    handler.record_error(ValueError("Test"), source="test")
    
    assert callback_count == 1  # No additional call
    print("[OK] Error callback removed")
    
    return True


def verify_error_export():
    """Verify error export"""
    print("\n=== Verifying Error Export ===")
    
    handler = AsyncErrorHandler()
    
    # Record error
    handler.record_error(ValueError("Export test"), source="test")
    
    # Export as JSON
    export = handler.export_errors()
    
    assert "Export test" in export
    assert "test" in export
    print("[OK] Error export to JSON working")
    
    return True


async def main():
    """Run all verifications"""
    print("=" * 60)
    print("Async Error Boundary & Structured Error Recovery Verification")
    print("=" * 60)
    
    try:
        verify_error_classification()
        verify_error_recording()
        await verify_async_recovery()
        await verify_circuit_breaker()
        verify_error_callbacks()
        verify_error_export()
        
        print("\n" + "=" * 60)
        print("ALL VERIFICATIONS PASSED")
        print("=" * 60)
        print("\nImplementation Summary:")
        print("  [OK] Error classification (severity & category)")
        print("  [OK] Structured error recording with context")
        print("  [OK] Async error recovery with retries")
        print("  [OK] Circuit breaker for async operations")
        print("  [OK] Error callbacks and monitoring")
        print("  [OK] Error history and statistics")
        print("\nReady for production use!")
        return 0
        
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
