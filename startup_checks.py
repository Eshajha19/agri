"""
Startup dependency verification.

Checks that critical services (Firebase, external APIs, ML models)
are reachable and configured before the app starts serving traffic.
Failures are logged with clear messages and the app continues in a
degraded state where possible.
"""
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str
    elapsed_ms: float = 0.0


@dataclass
class DependencyReport:
    results: List[CheckResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    skipped: int = 0

    def add(self, result: CheckResult) -> None:
        self.results.append(result)
        if result.ok:
            self.passed += 1
        elif "skipped" in result.message.lower():
            self.skipped += 1
        else:
            self.failed += 1

    def summary(self) -> str:
        total = self.passed + self.failed + self.skipped
        return (
            f"Dependency checks: {self.passed}/{total} passed, "
            f"{self.failed} failed, {self.skipped} skipped"
        )

    def log_all(self) -> None:
        for r in self.results:
            level = logging.INFO if r.ok else logging.WARNING
            logger.log(level, "[dep-check] %s: %s (%.0fms)", r.name, r.message, r.elapsed_ms)
        logger.info(self.summary())


def _timed(fn):
    """Run *fn* and return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = fn()
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_firebase() -> CheckResult:
    """Verify Firebase Admin SDK is initialized and Firestore is reachable."""
    import firebase_admin
    if not firebase_admin._apps:
        return CheckResult("firebase", False, "skipped — no Firebase credentials configured")

    try:
        from firebase_admin import firestore
        db = firestore.client()
        # Lightweight read to confirm Firestore is reachable
        db.collection("_health").document("ping").get()
        return CheckResult("firebase", True, "Firestore reachable")
    except Exception as exc:
        return CheckResult("firebase", False, f"Firestore unreachable: {exc}")


def check_weather_api() -> CheckResult:
    """Verify the weather API provider is reachable."""
    provider = os.getenv("WEATHER_API_PROVIDER", "open_meteo")
    if provider == "open_meteo":
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://api.open-meteo.com/v1/forecast?latitude=28.6&longitude=77.2&current_weather=true",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return CheckResult("weather_api", True, "Open-Meteo reachable")
                return CheckResult("weather_api", False, f"Open-Meteo returned {resp.status}")
        except Exception as exc:
            return CheckResult("weather_api", False, f"Open-Meteo unreachable: {exc}")
    return CheckResult("weather_api", True, f"skipped — custom provider '{provider}' (not validated)")


def check_gemini_api() -> CheckResult:
    """Verify Gemini API key is set (RAG dependency)."""
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        return CheckResult("gemini_api", False, "skipped — GEMINI_API_KEY not set (RAG disabled)")
    return CheckResult("gemini_api", True, "GEMINI_API_KEY configured")


def check_twilio() -> CheckResult:
    """Verify Twilio credentials are set."""
    sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    token = os.getenv("TWILIO_AUTH_TOKEN", "")
    if not sid or not token:
        return CheckResult("twilio", False, "skipped — TWILIO credentials not set (WhatsApp disabled)")
    return CheckResult("twilio", True, "Twilio credentials configured")


def check_ml_models() -> CheckResult:
    """Verify at least one ML model file exists."""
    model_files = ["yield_model.joblib", "sklearn_yield_model.joblib", "trend_forecast_model.joblib"]
    found = [f for f in model_files if os.path.exists(f)]
    if not found:
        return CheckResult("ml_models", False, "skipped — no model files found (ML prediction disabled)")
    return CheckResult("ml_models", True, f"found: {', '.join(found)}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def verify_startup_dependencies() -> DependencyReport:
    """Run all dependency checks and return a summary report."""
    report = DependencyReport()

    checks = [
        check_firebase,
        check_weather_api,
        check_gemini_api,
        check_twilio,
        check_ml_models,
    ]

    for check_fn in checks:
        try:
            result, elapsed = _timed(check_fn)
            result.elapsed_ms = elapsed
            report.add(result)
        except Exception as exc:
            report.add(CheckResult(check_fn.__name__, False, f"check crashed: {exc}"))

    report.log_all()
    return report
