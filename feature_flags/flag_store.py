"""
flag_store.py
─────────────
In-memory + Firestore-backed feature flag storage.
"""

import logging
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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
CACHE_TTL_SECONDS = 30

_cache: Dict[str, Dict] = {}
_cache_loaded_at: float = 0.0
_cache_lock = threading.Lock()


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


def _load_from_firestore() -> Dict[str, Dict]:
    if not _FIRESTORE_AVAILABLE:
        return {}
    try:
        docs = _fs_client.collection(COLLECTION).stream()
        return {d.id: _enrich(d.id, d.to_dict()) for d in docs}
    except Exception as e:
        logger.error("Failed to load flags from Firestore: %s", e)
        return {}


def _refresh_cache_locked() -> None:
    """Must be called with _cache_lock already held."""
    global _cache, _cache_loaded_at, _defaults_seeded
    loaded = _load_from_firestore()
    if not loaded:
        loaded = {fid: _enrich(fid, fdata) for fid, fdata in DEFAULT_FLAGS.items()}
        if _FIRESTORE_AVAILABLE:
            for fid, fdata in loaded.items():
                 doc_ref = _fs_client.collection(COLLECTION).document(fid)
                 try:
                     
                     doc_ref = _fs_client.collection(COLLECTION).document(fid)
                     if not doc_ref.get().exists:
                         doc_ref.set(fdata)
                 except Exception as e:
                     logger.error("Seeding failed for flag '%s': %s", fid, e)

    _cache = loaded
    _cache_loaded_at = time.monotonic()


def _ensure_cache():
    with _cache_lock:
        if not _cache or (time.monotonic() - _cache_loaded_at) > CACHE_TTL_SECONDS:
            _refresh_cache_locked()

def force_refresh():
    """Force refresh cache immediately (bypasses TTL)."""
    with _cache_lock:
        _refresh_cache_locked()


def _write_to_firestore(flag_id: str, data: Dict):
    if not _FIRESTORE_AVAILABLE:
        return
    try:
        _fs_client.collection(COLLECTION).document(flag_id).set(data)
    except Exception as e:
        logger.error("Failed to write flag '%s' to Firestore: %s", flag_id, e)


def list_flags() -> List[Dict]:
    _ensure_cache()
    with _cache_lock:
        return list(_cache.values())


def get_flag(flag_id: str) -> Optional[Dict]:
    _ensure_cache()
    with _cache_lock:
        return _cache.get(flag_id)
    
def _validate_flag_data(data: Dict) -> None:
    if "enabled" in data and not isinstance(data["enabled"], bool):
        raise ValueError("enabled must be a boolean")

    if "rollout_pct" in data:
        if not isinstance(data["rollout_pct"], int) or not (0 <= data["rollout_pct"] <= 100):
            raise ValueError("rollout_pct must be int between 0 and 100")

    if "cohorts" in data and not isinstance(data["cohorts"], list):
        raise ValueError("cohorts must be a list")

    if "tags" in data and not isinstance(data["tags"], list):
        raise ValueError("tags must be a list")

def upsert_flag(flag_id: str, data: Dict) -> Dict:
    _ensure_cache() 
    _validate_flag_data(data)
    with _cache_lock:
        existing = _cache.get(flag_id, {})
        merged = {**existing, **data, "id": flag_id, "updated_at": _now_iso()}
        if "created_at" not in merged:
            merged["created_at"] = _now_iso()
        merged = _enrich(flag_id, merged)
        _cache[flag_id] = merged
    _write_to_firestore(flag_id, merged)
    force_refresh()
    return merged


def delete_flag(flag_id: str) -> bool:
    _ensure_cache()
    with _cache_lock:
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
    _ensure_cache()
    with _cache_lock:
        if flag_id not in _cache:
            return None
    return upsert_flag(flag_id, {"enabled": False, "rollout_pct": 0,
                                  "rolled_back_at": _now_iso()})