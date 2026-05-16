"""
flag_store.py
─────────────
In-memory + Firestore-backed feature flag storage.

Flags are loaded once at startup (or on first request) from Firestore and
cached locally with a configurable TTL (default 5 min).  Every write
immediately updates both the local cache and Firestore so the source of
truth is always the database.

Flag document shape (Firestore collection: feature_flags):
{
  "id":           "rag_advisor_v2",
  "enabled":      true,
  "rollout_pct":  25,          // 0-100 — % of users who receive the feature
  "cohorts":      ["beta"],    // optional whitelist of cohort labels
  "description":  "RAG advisor second generation model",
  "owner":        "ml-team",
  "tags":         ["rag", "ml"],
  "created_at":   "2026-05-16T00:00:00Z",
  "updated_at":   "2026-05-16T00:00:00Z",
}
"""

import logging
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Firestore client (optional — graceful fallback when not configured) ───────
try:
    import firebase_admin
    from firebase_admin import credentials, firestore as fs_admin

    if not firebase_admin._apps:
        firebase_admin.initialize_app()

    _fs_client = fs_admin.client()
    _FIRESTORE_AVAILABLE = True
except Exception as _e:
    logger.warning("Firestore unavailable for feature flags — using in-memory only: %s", _e)
    _fs_client = None
    _FIRESTORE_AVAILABLE = False

COLLECTION = "feature_flags"
CACHE_TTL_SECONDS = 300  # 5 minutes

# ── In-memory cache ────────────────────────────────────────────────────────────
_cache: Dict[str, Dict] = {}
_cache_loaded_at: float = 0.0

# ── Default flags (used when Firestore is unavailable) ────────────────────────
DEFAULT_FLAGS: Dict[str, Dict] = {
    "rag_advisor_v2": {
        "enabled": False, "rollout_pct": 0, "cohorts": [],
        "description": "RAG Advisor v2 with hybrid retrieval",
        "owner": "ml-team", "tags": ["rag", "ml"],
    },
    "yield_prediction_lstm": {
        "enabled": False, "rollout_pct": 0, "cohorts": [],
        "description": "LSTM-based yield prediction model",
        "owner": "ml-team", "tags": ["lstm", "ml"],
    },
    "smart_crop_recommendation": {
        "enabled": True, "rollout_pct": 100, "cohorts": [],
        "description": "ML-powered crop recommendation engine",
        "owner": "product", "tags": ["ml", "crops"],
    },
    "climate_simulator_ml": {
        "enabled": False, "rollout_pct": 10, "cohorts": ["beta"],
        "description": "ML-enhanced climate simulation",
        "owner": "ml-team", "tags": ["ml", "climate"],
    },
    "pest_detection_v2": {
        "enabled": False, "rollout_pct": 0, "cohorts": [],
        "description": "Improved CV-based pest detection",
        "owner": "ml-team", "tags": ["cv", "ml"],
    },
    "soil_analysis_advanced": {
        "enabled": False, "rollout_pct": 5, "cohorts": [],
        "description": "Advanced soil health ML analysis",
        "owner": "ml-team", "tags": ["ml", "soil"],
    },
    "market_price_forecast": {
        "enabled": True, "rollout_pct": 100, "cohorts": [],
        "description": "ML market price forecasting",
        "owner": "product", "tags": ["ml", "market"],
    },
    "personalized_advisory_v2": {
        "enabled": False, "rollout_pct": 0, "cohorts": [],
        "description": "Personalized ML advisory engine v2",
        "owner": "ml-team", "tags": ["ml", "advisory"],
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _enrich(flag_id: str, data: Dict) -> Dict:
    """Ensure all required fields are present on a flag dict."""
    enriched = deepcopy(data)
    enriched.setdefault("id", flag_id)
    enriched.setdefault("enabled", False)
    enriched.setdefault("rollout_pct", 0)
    enriched.setdefault("cohorts", [])
    enriched.setdefault("description", "")
    enriched.setdefault("owner", "unknown")
    enriched.setdefault("tags", [])
    enriched.setdefault("created_at", _now_iso())
    enriched.setdefault("updated_at", _now_iso())
    return enriched


# ── Public API ─────────────────────────────────────────────────────────────────

def _load_from_firestore() -> Dict[str, Dict]:
    """Fetch all flags from Firestore. Returns empty dict on failure."""
    if not _FIRESTORE_AVAILABLE:
        return {}
    try:
        docs = _fs_client.collection(COLLECTION).stream()
        return {d.id: _enrich(d.id, d.to_dict()) for d in docs}
    except Exception as e:
        logger.error("Failed to load flags from Firestore: %s", e)
        return {}


def _refresh_cache():
    global _cache, _cache_loaded_at
    loaded = _load_from_firestore()
    if not loaded:
        # Seed Firestore with defaults on first run
        for fid, fdata in DEFAULT_FLAGS.items():
            if fid not in loaded:
                loaded[fid] = _enrich(fid, fdata)
                _write_to_firestore(fid, loaded[fid])
    _cache = loaded or {fid: _enrich(fid, fd) for fid, fd in DEFAULT_FLAGS.items()}
    _cache_loaded_at = time.monotonic()


def _ensure_cache():
    if not _cache or (time.monotonic() - _cache_loaded_at) > CACHE_TTL_SECONDS:
        _refresh_cache()


def _write_to_firestore(flag_id: str, data: Dict):
    if not _FIRESTORE_AVAILABLE:
        return
    try:
        _fs_client.collection(COLLECTION).document(flag_id).set(data)
    except Exception as e:
        logger.error("Failed to write flag '%s' to Firestore: %s", flag_id, e)


def list_flags() -> List[Dict]:
    _ensure_cache()
    return list(_cache.values())


def get_flag(flag_id: str) -> Optional[Dict]:
    _ensure_cache()
    return _cache.get(flag_id)


def upsert_flag(flag_id: str, data: Dict) -> Dict:
    global _cache, _cache_loaded_at
    _ensure_cache()
    existing = _cache.get(flag_id, {})
    merged = {**existing, **data, "id": flag_id, "updated_at": _now_iso()}
    if "created_at" not in merged:
        merged["created_at"] = _now_iso()
    merged = _enrich(flag_id, merged)
    
    _cache[flag_id] = merged
    _cache_loaded_at = time.monotonic()
    
    _write_to_firestore(flag_id, merged)
    return merged


def delete_flag(flag_id: str) -> bool:
    _ensure_cache()
    if flag_id not in _cache:
        return False
    del _cache[flag_id]
    if _FIRESTORE_AVAILABLE:
        try:
            _fs_client.collection(COLLECTION).document(flag_id).delete()
        except Exception as e:
            logger.error("Failed to delete flag '%s' from Firestore: %s", flag_id, e)
    return True


def rollback_flag(flag_id: str) -> Optional[Dict]:
    """Disable a flag immediately and reset rollout to 0%."""
    _ensure_cache()
    if flag_id not in _cache:
        return None
    return upsert_flag(flag_id, {"enabled": False, "rollout_pct": 0,
                                  "rolled_back_at": _now_iso()})
