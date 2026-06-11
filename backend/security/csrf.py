import secrets, time

CSRF_TOKENS = {}  # token -> expiry timestamp

TTL_SECONDS = 300  # 5 minutes

def generate_csrf_token() -> str:
    token = secrets.token_urlsafe(32)
    CSRF_TOKENS[token] = time.time() + TTL_SECONDS
    return token

def validate_csrf_token(token: str) -> bool:
    expiry = CSRF_TOKENS.get(token)
    if not expiry:
        return False
    if time.time() > expiry:
        CSRF_TOKENS.pop(token, None)
        return False
    # enforce single‑use semantics
    CSRF_TOKENS.pop(token, None)
    return True
