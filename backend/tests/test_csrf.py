import pytest
import time
import warnings
import threading
import backend.security.csrf
from backend.security.csrf import generate_csrf_token, validate_csrf_token

def test_deprecation_warnings():
    # Verify warnings are raised on calls
    with pytest.warns(DeprecationWarning):
        token = generate_csrf_token()
    with pytest.warns(DeprecationWarning):
        validate_csrf_token(token)

def test_token_valid_once():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        token = generate_csrf_token()
        assert validate_csrf_token(token) is True
        # second use should fail
        assert validate_csrf_token(token) is False

def test_token_expires():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # Temporarily shorten TTL to 2 seconds for testing
        original_ttl = backend.security.csrf.TTL_SECONDS
        backend.security.csrf.TTL_SECONDS = 2
        try:
            token = generate_csrf_token()
            time.sleep(0.5)
            assert validate_csrf_token(token) is True
            
            expired = generate_csrf_token()
            # simulate expiry
            time.sleep(3)
            assert validate_csrf_token(expired) is False
        finally:
            backend.security.csrf.TTL_SECONDS = original_ttl

def test_cache_bounds():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        
        # Test capacity boundary eviction behavior
        from backend.security.csrf import TTLCache
        test_cache = TTLCache(maxsize=3, ttl=100)
        test_cache["k1"] = 1
        test_cache["k2"] = 2
        test_cache["k3"] = 3
        test_cache["k4"] = 4
        assert "k1" not in test_cache
        assert "k4" in test_cache

def test_thread_safety():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        
        def run_generate():
            for _ in range(50):
                generate_csrf_token()
                
        threads = [threading.Thread(target=run_generate) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
