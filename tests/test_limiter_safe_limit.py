from types import SimpleNamespace

from backend.core import limiter as limiter_module


class _FakeApp:
    def __init__(self):
        self.state = SimpleNamespace()
        self.exception_handlers = {}

    def add_exception_handler(self, exception_class, handler):
        self.exception_handlers[exception_class] = handler


class _FailingLimiter:
    def limit(self, rate):
        def decorator(fn):
            raise ValueError("invalid rate")

        return decorator


def test_safe_limit_logs_failed_decorator(monkeypatch, caplog):
    monkeypatch.setattr(limiter_module, "build_limiter", lambda default_limits: _FailingLimiter())
    caplog.set_level("ERROR", logger=limiter_module.__name__)
    app = _FakeApp()

    limiter = limiter_module.setup_rate_limiter(app)

    def endpoint():
        return {"ok": True}

    decorated = limiter.limit("abc/minute")(endpoint)

    assert decorated is endpoint
    assert "Rate limit decorator failed for rate=abc/minute on endpoint" in caplog.text
    assert "Endpoint is UNPROTECTED" in caplog.text
