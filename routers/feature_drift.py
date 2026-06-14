"""
Feature Drift Detection & Data Consistency Validation
======================================================
Implements training-serving feature skew detection for the Fasal Saathi
yield prediction model (XGBoost, trained via train_model.py).

Responsibilities
----------------
* FeatureBaseline  — loads / saves feature_baseline.json
* SchemaValidator  — missing fields, extra fields, dtype mismatches
* DriftAnalyzer    — KS-test (numeric) + category-set diff (categorical)
* DriftLogger      — appends JSON-line records to drift_logs/drift.log
* FastAPI router   — 4 public endpoints

Baseline file: feature_baseline.json  (root working directory)
Log file     : drift_logs/drift.log   (root working directory)

No external dependencies beyond what is already in requirements.txt.
scipy.stats.ks_2samp is used for the KS-test; scipy is already listed.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

# Auth — injected at startup by main.py
verify_role_fn = None

def init_auth(vr_fn):
    global verify_role_fn
    verify_role_fn = vr_fn
    logger.info("Feature drift router: auth initialised")

async def _require_authenticated(request):
    """Any valid Firebase token — any role accepted."""
    if verify_role_fn is None:
        raise HTTPException(status_code=503, detail="Authorization service not initialised")
    return await verify_role_fn(request, required_roles=None)

async def _require_admin_or_expert(request):
    """Admin or expert role required."""
    if verify_role_fn is None:
        raise HTTPException(status_code=503, detail="Authorization service not initialised")
    return await verify_role_fn(request, required_roles=["admin", "expert"])

# Allowlist: only portable path characters.
# (../, ..\) and shell metacharacters before the value reaches any filesystem.
_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_./ -]+$")


def init_auth(vr_fn):
    global verify_role_fn
    verify_role_fn = vr_fn


# ---------------------------------------------------------------------------
# Paths (relative to the process working directory, i.e. the repo root)
# ---------------------------------------------------------------------------
BASELINE_PATH = Path("feature_baseline.json")
DRIFT_LOG_DIR = Path("drift_logs")
DRIFT_LOG_PATH = DRIFT_LOG_DIR / "drift.log"

# ---------------------------------------------------------------------------
# Drift-severity thresholds
# ---------------------------------------------------------------------------
# KS-statistic thresholds for numeric features
KS_WARN_THRESHOLD = 0.10   # mild drift — log + warn
KS_ALERT_THRESHOLD = 0.25  # severe drift — log + alert flag

# Fraction of unknown categories that triggers a warning (0.0 = any unknown)
CATEGORY_WARN_FRACTION = 0.0

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ValidateRequest(BaseModel):
    """
    A single inference payload to validate against the training baseline.
    keys   : feature names (pre-dummies, i.e. raw categorical strings are fine)
    values : feature values (str for categoricals, float/int for numerics)
    """
    features: Dict[str, Any] = Field(
        ...,
        description="Raw feature dict matching PredictRequest fields.",
        min_length=1,
    )


class FeatureDriftResult(BaseModel):
    feature: str
    status: str          # "ok" | "warn" | "alert" | "missing" | "extra" | "type_error"
    message: str
    ks_statistic: Optional[float] = None
    p_value: Optional[float] = None


class ValidateResponse(BaseModel):
    valid: bool
    overall_status: str  # "ok" | "warn" | "alert"
    schema_errors: List[str]
    drift_results: List[FeatureDriftResult]
    validated_at: str


class StatusResponse(BaseModel):
    baseline_exists: bool
    baseline_generated_at: Optional[str]
    total_features_tracked: int
    numeric_features: int
    categorical_features: int
    recent_alerts: List[Dict[str, Any]]
    log_entry_count: int


class UpdateBaselineResponse(BaseModel):
    success: bool
    message: str
    features_saved: int
    generated_at: str


class LogsResponse(BaseModel):
    entries: List[Dict[str, Any]]
    total: int


# ---------------------------------------------------------------------------
# FeatureBaseline
# ---------------------------------------------------------------------------

class FeatureBaseline:
    """
    Loads and saves feature_baseline.json.

    Baseline JSON structure
    -----------------------
    {
      "generated_at": "2026-05-27T10:00:00",
      "numeric_features": {
        "CropCoveredArea": {
          "mean": 5.2, "std": 2.1, "min": 0.5, "max": 100.0,
          "sample_values": [1.0, 2.5, 3.0, ...]   // up to 500 values for KS-test
        },
        ...
      },
      "categorical_features": {
        "Crop": {
          "categories": ["Rice", "Wheat", ...],
          "value_counts": {"Rice": 0.45, "Wheat": 0.30, ...}  // fractions
        },
        ...
      }
    }
    """

    # Features that train_model.py drops before training — never in baseline
    _DROP_COLS = frozenset(
        ["FarmID", "category", "State", "District", "Sub-District",
         "SDate", "HDate", "ExpYield", "geometry"]
    )
    # Categorical columns used in pd.get_dummies
    _CAT_COLS = frozenset(
        ["Crop", "CNext", "CLast", "CTransp", "IrriType", "IrriSource", "Season"]
    )
    # Numeric columns (everything remaining after dropping above + categoricals)
    _NUM_COLS = frozenset(
        ["CropCoveredArea", "CHeight", "IrriCount", "WaterCov"]
    )

    def __init__(self):
        self._data: Optional[Dict] = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        """Load baseline from disk. Returns True if loaded, False if not found."""
        with self._lock:
            if not BASELINE_PATH.exists():
                return False
            try:
                with open(BASELINE_PATH, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info("FeatureBaseline: loaded from %s", BASELINE_PATH)
                return True
            except Exception as exc:
                logger.error("FeatureBaseline: failed to load: %s", exc)
                return False

    def save(self, data: Dict) -> None:
        """Persist baseline data to disk atomically."""
        tmp = BASELINE_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, BASELINE_PATH)
        with self._lock:
            self._data = data
        logger.info("FeatureBaseline: saved to %s", BASELINE_PATH)

    @property
    def data(self) -> Optional[Dict]:
        with self._lock:
            return self._data

    @property
    def exists(self) -> bool:
        with self._lock:
            return self._data is not None

    def build_from_csv(self, csv_path: str = "Train.csv") -> Dict:
        """
        Build a fresh baseline by reading the training CSV.
        This mirrors the preprocessing in train_model.py exactly.
        Called by the /baseline/update admin endpoint.
        """
        import pandas as pd  # import here so module loads even without pandas at import time

        df = pd.read_csv(csv_path)
        df["SDate"] = pd.to_datetime(df["SDate"], errors="coerce")
        df = df.dropna(subset=["SDate"])

        # Drop the same columns train_model.py drops
        X = df.drop(
            columns=[c for c in self._DROP_COLS if c in df.columns],
            errors="ignore",
        )

        numeric_features: Dict[str, Any] = {}
        categorical_features: Dict[str, Any] = {}

        for col in X.columns:
            if col in self._CAT_COLS:
                vc = X[col].dropna().value_counts(normalize=True)
                categorical_features[col] = {
                    "categories": vc.index.tolist(),
                    "value_counts": vc.to_dict(),
                }
            else:
                series = X[col].dropna().astype(float)
                # Store up to 500 sample values for KS-test at inference time
                sample = series.sample(
                    min(500, len(series)), random_state=42
                ).tolist()
                numeric_features[col] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "sample_values": sample,
                }

        baseline = {
            "generated_at": datetime.utcnow().isoformat(),
            "csv_path": csv_path,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        }
        return baseline


# ---------------------------------------------------------------------------
# SchemaValidator
# ---------------------------------------------------------------------------

class SchemaValidator:
    """Checks for missing features, extra features, and type mismatches."""

    def __init__(self, baseline: FeatureBaseline):
        self._baseline = baseline

    def validate(self, features: Dict[str, Any]) -> List[str]:
        """
        Returns a list of human-readable schema error strings.
        Empty list means schema is valid.
        """
        errors: List[str] = []
        data = self._baseline.data
        if data is None:
            errors.append("Baseline not loaded — cannot perform schema validation.")
            return errors

        expected_numeric = set(data.get("numeric_features", {}).keys())
        expected_categorical = set(data.get("categorical_features", {}).keys())
        expected_all = expected_numeric | expected_categorical
        provided = set(features.keys())

        # Missing features
        missing = expected_all - provided
        for f in sorted(missing):
            errors.append(f"Missing feature: '{f}'")

        # Extra / unknown features
        extra = provided - expected_all
        for f in sorted(extra):
            errors.append(f"Unknown feature: '{f}' (not present in training data)")

        # Type checks for numeric fields
        for f in expected_numeric & provided:
            val = features[f]
            if val is None:
                errors.append(f"Null value for numeric feature '{f}'")
                continue
            try:
                fv = float(val)
                if not math.isfinite(fv):
                    errors.append(f"Non-finite value for '{f}': {val}")
            except (TypeError, ValueError):
                errors.append(
                    f"Type mismatch for '{f}': expected numeric, got '{type(val).__name__}'"
                )

        # Type checks for categorical fields
        for f in expected_categorical & provided:
            val = features[f]
            if val is None:
                errors.append(f"Null value for categorical feature '{f}'")
                continue
            if not isinstance(val, str):
                errors.append(
                    f"Type mismatch for '{f}': expected string, got '{type(val).__name__}'"
                )

        return errors


# ---------------------------------------------------------------------------
# DriftAnalyzer
# ---------------------------------------------------------------------------

class DriftAnalyzer:
    """
    Compares inference features against the training baseline.

    Numeric  : KS two-sample test (inference value treated as a 1-point sample).
               A single scalar vs the stored sample_values distribution.
    Categorical : Checks if the value is in the training category set.
    """

    def __init__(self, baseline: FeatureBaseline):
        self._baseline = baseline

    def analyze(self, features: Dict[str, Any]) -> List[FeatureDriftResult]:
        results: List[FeatureDriftResult] = []
        data = self._baseline.data
        if data is None:
            return results

        numeric_baseline = data.get("numeric_features", {})
        categorical_baseline = data.get("categorical_features", {})

        # --- Numeric features ---
        for feat, stats in numeric_baseline.items():
            if feat not in features or features[feat] is None:
                continue  # schema validator already flagged this
            try:
                val = float(features[feat])
            except (TypeError, ValueError):
                continue  # schema validator already flagged this

            sample_vals = stats.get("sample_values", [])
            if len(sample_vals) < 2:
                results.append(FeatureDriftResult(
                    feature=feat,
                    status="ok",
                    message="Insufficient baseline samples for KS-test; skipped.",
                ))
                continue

            # KS-test: [val] vs training distribution
            ks_stat, p_val = ks_2samp([val], sample_vals)

            # Range check
            feat_min = stats.get("min", -math.inf)
            feat_max = stats.get("max", math.inf)
            out_of_range = val < feat_min or val > feat_max

            if ks_stat >= KS_ALERT_THRESHOLD or out_of_range:
                status = "alert"
                msg = (
                    f"Severe drift: KS={ks_stat:.3f}, p={p_val:.4f}. "
                    f"Training range [{feat_min:.2f}, {feat_max:.2f}], got {val}."
                )
            elif ks_stat >= KS_WARN_THRESHOLD:
                status = "warn"
                msg = (
                    f"Mild drift: KS={ks_stat:.3f}, p={p_val:.4f}. "
                    f"Training mean={stats.get('mean', 0):.2f}."
                )
            else:
                status = "ok"
                msg = f"No drift detected. KS={ks_stat:.3f}."

            results.append(FeatureDriftResult(
                feature=feat,
                status=status,
                message=msg,
                ks_statistic=round(ks_stat, 4),
                p_value=round(p_val, 4),
            ))

        # --- Categorical features ---
        for feat, stats in categorical_baseline.items():
            if feat not in features or features[feat] is None:
                continue
            val = str(features[feat])
            known_cats = set(stats.get("categories", []))

            if val not in known_cats:
                results.append(FeatureDriftResult(
                    feature=feat,
                    status="alert",
                    message=(
                        f"Unknown category '{val}' for feature '{feat}'. "
                        f"Training categories: {sorted(known_cats)[:10]}{'...' if len(known_cats) > 10 else ''}."
                    ),
                ))
            else:
                freq = stats.get("value_counts", {}).get(val, 0.0)
                results.append(FeatureDriftResult(
                    feature=feat,
                    status="ok",
                    message=f"Category '{val}' is known (training frequency: {freq:.2%}).",
                ))

        return results


# ---------------------------------------------------------------------------
# DriftLogger
# ---------------------------------------------------------------------------

class DriftLogger:
    """
    Appends JSON-line records to drift_logs/drift.log.
    Thread-safe via a per-instance lock.
    Keeps the last N lines in memory for the /logs endpoint.
    """

    MAX_MEMORY_ENTRIES = 200

    def __init__(self):
        self._lock = threading.Lock()
        self._memory: List[Dict] = []
        DRIFT_LOG_DIR.mkdir(exist_ok=True)

    def log(
        self,
        features: Dict[str, Any],
        schema_errors: List[str],
        drift_results: List[FeatureDriftResult],
        overall_status: str,
    ) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": overall_status,
            "schema_error_count": len(schema_errors),
            "schema_errors": schema_errors[:5],  # cap to avoid huge log lines
            "drift_alerts": [
                {"feature": r.feature, "status": r.status, "message": r.message}
                for r in drift_results
                if r.status in ("alert", "warn")
            ],
            "feature_count": len(features),
        }
        with self._lock:
            try:
                with open(DRIFT_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except OSError as exc:
                logger.warning("DriftLogger: could not write log: %s", exc)
            # Keep in-memory ring buffer
            self._memory.append(entry)
            if len(self._memory) > self.MAX_MEMORY_ENTRIES:
                self._memory = self._memory[-self.MAX_MEMORY_ENTRIES:]

    def recent(self, n: int = 50) -> List[Dict]:
        with self._lock:
            return list(self._memory[-n:])

    def count(self) -> int:
        with self._lock:
            return len(self._memory)


# ---------------------------------------------------------------------------
# Module-level singletons (initialised once at import time)
# ---------------------------------------------------------------------------
_baseline = FeatureBaseline()
_baseline.load()   # loads if file exists, silently skips if not yet generated
_schema_validator = SchemaValidator(_baseline)
_drift_analyzer = DriftAnalyzer(_baseline)
_drift_logger = DriftLogger()


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api/feature-drift", tags=["Feature Drift"])


@router.post("/validate", response_model=ValidateResponse)
async def validate_features(payload: ValidateRequest, request: Request):
    await _require_authenticated(request)
    """
    Validate a single inference payload against the training baseline.

    Performs:
    1. Schema check  — missing / extra fields, type mismatches
    2. Drift check   — KS-test for numerics, category-set check for categoricals
    3. Logging       — always logs to drift_logs/drift.log

    Returns validation result with per-feature drift details.
    Does NOT block the prediction — callers use this for observability only.
    """
    if not _baseline.exists:
        raise HTTPException(
            status_code=503,
            detail=(
                "Feature baseline not yet generated. "
                "Run train_model.py first, or call POST /api/feature-drift/baseline/update."
            ),
        )

    features = payload.features

    # Step 1: Schema validation
    schema_errors = _schema_validator.validate(features)

    # Step 2: Drift analysis (only for features that passed schema check)
    drift_results = _drift_analyzer.analyze(features)

    # Step 3: Determine overall status
    has_alert = bool(schema_errors) or any(r.status == "alert" for r in drift_results)
    has_warn = any(r.status == "warn" for r in drift_results)

    if has_alert:
        overall_status = "alert"
    elif has_warn:
        overall_status = "warn"
    else:
        overall_status = "ok"

    # Step 4: Log (always — even for clean requests, for observability)
    _drift_logger.log(features, schema_errors, drift_results, overall_status)

    return ValidateResponse(
        valid=not bool(schema_errors),
        overall_status=overall_status,
        schema_errors=schema_errors,
        drift_results=drift_results,
        validated_at=datetime.utcnow().isoformat(),
    )


@router.get("/status", response_model=StatusResponse)
async def get_drift_status(request: Request):
    await _require_authenticated(request)
    """
    Returns the current state of the feature drift monitoring system:
    - whether a baseline exists and when it was generated
    - how many features are being tracked
    - recent alerts from the in-memory log buffer
    """
    if not _baseline.exists:
        return StatusResponse(
            baseline_exists=False,
            baseline_generated_at=None,
            total_features_tracked=0,
            numeric_features=0,
            categorical_features=0,
            recent_alerts=[],
            log_entry_count=0,
        )

    data = _baseline.data
    num_feats = data.get("numeric_features", {})
    cat_feats = data.get("categorical_features", {})

    recent_all = _drift_logger.recent(50)
    alerts_only = [e for e in recent_all if e["overall_status"] in ("alert", "warn")]

    return StatusResponse(
        baseline_exists=True,
        baseline_generated_at=data.get("generated_at"),
        total_features_tracked=len(num_feats) + len(cat_feats),
        numeric_features=len(num_feats),
        categorical_features=len(cat_feats),
        recent_alerts=alerts_only[-20:],
        log_entry_count=_drift_logger.count(),
    )


@router.post("/baseline/update", response_model=UpdateBaselineResponse)
async def update_baseline(request: Request, csv_path: str = "Train.csv"):
    """
    Admin endpoint: rebuild the feature baseline from the training CSV.

    Call this after retraining the model so the drift detector stays
    aligned with the new training distribution.

    csv_path: path to the training CSV (default: Train.csv in working dir).
    """
    await _require_admin_or_expert(request)

    components = re.split(r"[/\\]", csv_path)
    if ".." in components:
        raise HTTPException(
            status_code=400,
            detail="csv_path must not contain path traversal sequences (..)",
        )
    if not _SAFE_PATH_RE.match(csv_path):
        raise HTTPException(
            status_code=400,
            detail="csv_path contains invalid characters.",
        )

    if not Path(csv_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Training CSV not found at '{csv_path}'.",
        )

    try:
        new_baseline = _baseline.build_from_csv(csv_path)
        _baseline.save(new_baseline)
        # Reload singletons so live requests immediately use the new baseline
        _baseline.load()

        num_count = len(new_baseline.get("numeric_features", {}))
        cat_count = len(new_baseline.get("categorical_features", {}))
        total = num_count + cat_count

        logger.info(
            "Feature baseline updated: %d numeric + %d categorical = %d features",
            num_count, cat_count, total,
        )

        return UpdateBaselineResponse(
            success=True,
            message=f"Baseline rebuilt from '{csv_path}' with {total} features.",
            features_saved=total,
            generated_at=new_baseline["generated_at"],
        )
    except Exception as exc:
        logger.error("Baseline update failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build baseline: {str(exc)}",
        )


@router.get("/logs", response_model=LogsResponse)
async def get_drift_logs(request: Request, limit: int = 50):
    await _require_admin_or_expert(request)
    """
    Returns the most recent drift log entries (from in-memory buffer).
    Limit: 1–200 entries.
    """
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200.")

    entries = _drift_logger.recent(limit)
    return LogsResponse(entries=entries, total=_drift_logger.count())