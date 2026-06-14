import secrets
import time
import warnings
import threading

# Deprecation warning at module level
warnings.warn(
    "backend.security.csrf is deprecated and will be removed in a future version. "
    "Use csrf_protection.py instead.",
    DeprecationWarning,
    stacklevel=2
)

TTL_SECONDS = 300  # 5 minutes
_lock = threading.Lock()

try:
    from cachetools import TTLCache
except ImportError:
    # Robust fallback class in case cachetools is not installed in the environment
    class TTLCache(dict):
        def __init__(self, maxsize: int, ttl: float):
            super().__init__()
            self.maxsize = maxsize
            self.ttl = ttl
            self._expiries = {}
            self._cache_lock = threading.Lock()

        def __setitem__(self, key, value):
            with self._cache_lock:
                if len(self) >= self.maxsize and key not in self:
                    now = time.time()
                    expired_keys = [k for k, exp in self._expiries.items() if now > exp]
                    if expired_keys:
                        for k in expired_keys:
                            self.pop(k, None)
                            self._expiries.pop(k, None)
                    if len(self) >= self.maxsize:
                        oldest = next(iter(self.keys()))
                        self.pop(oldest, None)
                        self._expiries.pop(oldest, None)
                super().__setitem__(key, value)
                self._expiries[key] = time.time() + self.ttl

        def __getitem__(self, key):
            with self._cache_lock:
                now = time.time()
                if key in self._expiries and now > self._expiries[key]:
                    self.pop(key, None)
                    self._expiries.pop(key, None)
                return super().__getitem__(key)

        def get(self, key, default=None):
            try:
                return self[key]
            except KeyError:
                return default

        def pop(self, key, default=None):
            with self._cache_lock:
                self._expiries.pop(key, None)
                return super().pop(key, default)

# Thread-safe bounded cache
CSRF_TOKENS = TTLCache(maxsize=10000, ttl=TTL_SECONDS)

def generate_csrf_token() -> str:
    warnings.warn(
        "generate_csrf_token is deprecated. Use csrf_protection.generate_token instead.",
        DeprecationWarning,
        stacklevel=2
    )
    token = secrets.token_urlsafe(32)
    with _lock:
        CSRF_TOKENS[token] = time.time() + TTL_SECONDS
    return token

def validate_csrf_token(token: str) -> bool:
    warnings.warn(
        "validate_csrf_token is deprecated. Use csrf_protection.validate_token instead.",
        DeprecationWarning,
        stacklevel=2
    )
    with _lock:
        expiry = CSRF_TOKENS.get(token)
        if not expiry:
            return False
        if time.time() > expiry:
            CSRF_TOKENS.pop(token, None)
            return False
        # Enforce single-use semantics
        CSRF_TOKENS.pop(token, None)
        return True
