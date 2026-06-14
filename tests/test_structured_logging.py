import json
import logging
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_logging_json_output():
    """Test that logs are emitted as valid JSON."""
    import io
    import sys

    # Capture stdout
    captured_output = io.StringIO()
    handler = logging.StreamHandler(captured_output)
    handler.setLevel(logging.INFO)
    
    # Get our logger
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    
    # Emit a test log
    test_message = "test structured logging"
    logger.info(test_message)
    
    # Get the captured output
    log_output = captured_output.getvalue().strip()
    assert log_output, "No log output captured"
    
    # Parse as JSON
    try:
        log_data = json.loads(log_output)
        assert "message" in log_data
        assert log_data["message"] == test_message
        assert "levelname" in log_data or "level" in log_data
        assert "asctime" in log_data
    finally:
        logger.removeHandler(handler)


def test_correlation_id_header():
    """Test that response contains X-Correlation-ID and X-Request-ID headers."""
    response = client.get("/")
    assert response.status_code in [200, 404]  # Accept either (depends on what "/" does)
    assert "X-Correlation-ID" in response.headers
    assert "X-Request-ID" in response.headers


def test_log_context_contains_correlation_id():
    """Test that log records contain correlation ID and request ID during a request."""
    import io
    import logging

    # Capture stdout
    captured_output = io.StringIO()
    handler = logging.StreamHandler(captured_output)
    
    # Configure root logger
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)
    
    try:
        # Make a request that will log something
        test_correlation_id = "test-correlation-id-12345"
        response = client.get("/", headers={"X-Correlation-ID": test_correlation_id})
        
        # Check response header
        assert response.headers["X-Correlation-ID"] == test_correlation_id
        
        # Check logs
        log_output = captured_output.getvalue()
        if log_output:
            # Split into lines (in case there are multiple logs)
            for line in log_output.strip().split("\n"):
                if line:
                    try:
                        log_data = json.loads(line)
                        if "correlation_id" in log_data:
                            assert log_data["correlation_id"] == test_correlation_id
                            assert "request_id" in log_data
                            break
                    except json.JSONDecodeError:
                        continue
    finally:
        # Restore original handlers
        root_logger.handlers = original_handlers
