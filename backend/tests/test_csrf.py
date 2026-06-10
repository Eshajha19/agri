from backend.security.csrf import generate_csrf_token, validate_csrf_token
import time

def test_token_valid_once():
    token = generate_csrf_token()
    assert validate_csrf_token(token) is True
    # second use should fail
    assert validate_csrf_token(token) is False

def test_token_expires():
    token = generate_csrf_token()
    time.sleep(1)
    assert validate_csrf_token(token) is True
    expired = generate_csrf_token()
    # simulate expiry
    time.sleep(6)
    assert validate_csrf_token(expired) is False
