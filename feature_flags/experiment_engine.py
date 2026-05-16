"""
experiment_engine.py
─────────────────────
Deterministic user-to-variant assignment for A/B experiments.

Assignment is deterministic: the same (user_id, experiment_id) pair always
resolves to the same variant.  This is done via SHA-256 hashing so:
  - No database lookup is needed to retrieve a cached assignment.
  - Assignment is consistent across sessions and devices.
  - Changing the `salt` on an experiment re-randomises all users.

Experiment document shape (Firestore collection: experiments):
{
  "id":          "yield_model_ab",
  "name":        "Yield Model A/B Test",
  "description": "Compare LSTM vs sklearn yield predictions",
  "status":      "running",        // draft | running | paused | completed
  "variants": [
    {"id": "control",     "name": "Control (sklearn)", "weight": 50},
    {"id": "treatment_a", "name": "LSTM model",        "weight": 50},
  ],
  "salt":        "abc123",         // random string; change to re-randomise
  "start_date":  "2026-05-16",
  "end_date":    null,
  "created_at":  "2026-05-16T00:00:00Z",
  "updated_at":  "2026-05-16T00:00:00Z",
  "owner":       "ml-team",
  "tags":        ["ml", "yield"],
}
"""

import hashlib
import logging
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Firestore client ───────────────────────────────────────────────────────────
try:
    from firebase_admin import firestore as fs_admin
    _fs_client = fs_admin.client()
    _FIRESTORE_AVAILABLE = True
except Exception:
    _fs_client = None
    _FIRESTORE_AVAILABLE = False

EXP_COLLECTION   = "experiments"
ASSIGN_COLLECTION = "experiment_assignments"
CACHE_TTL_SECONDS = 300

_exp_cache: Dict[str, Dict] = {}
_exp_cache_at: float = 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Deterministic assignment ───────────────────────────────────────────────────

def _assign_variant(user_id: str, experiment_id: str, salt: str,
                    variants: List[Dict]) -> str:
    """
    Hash (user_id + experiment_id + salt) → [0,100) integer → select variant
    by cumulative weight.  Always returns a variant id string.
    """
    if not variants:
        return "control"

    raw = f"{user_id}:{experiment_id}:{salt}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    # Use first 8 hex chars → 32-bit number → mod 100
    bucket = int(digest[:8], 16) % 100

    cumulative = 0
    for variant in variants:
        cumulative += int(variant.get("weight", 0))
        if bucket < cumulative:
            return variant["id"]

    # Fallback to last variant if weights don't sum to 100
    return variants[-1]["id"]


# ── Experiment cache ──────────────────────────────────────────────────────────

def _ensure_exp_cache():
    global _exp_cache, _exp_cache_at
    if not _exp_cache or (time.monotonic() - _exp_cache_at) > CACHE_TTL_SECONDS:
        _exp_cache = _load_experiments()
        _exp_cache_at = time.monotonic()


def _load_experiments() -> Dict[str, Dict]:
    if not _FIRESTORE_AVAILABLE:
        return _default_experiments()
    try:
        docs = _fs_client.collection(EXP_COLLECTION).stream()
        result = {d.id: d.to_dict() for d in docs}
        return result if result else _default_experiments()
    except Exception as e:
        logger.error("Failed to load experiments: %s", e)
        return _default_experiments()


def _default_experiments() -> Dict[str, Dict]:
    return {
        "yield_model_ab": {
            "id": "yield_model_ab",
            "name": "Yield Model A/B Test",
            "description": "Compare LSTM vs sklearn yield predictions",
            "status": "running",
            "variants": [
                {"id": "control",     "name": "sklearn (baseline)", "weight": 50},
                {"id": "treatment_a", "name": "LSTM model",         "weight": 50},
            ],
            "salt": "ym_salt_2026",
            "start_date": "2026-05-16",
            "end_date": None,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "owner": "ml-team",
            "tags": ["ml", "yield"],
        },
        "rag_retrieval_ab": {
            "id": "rag_retrieval_ab",
            "name": "RAG Retrieval Strategy",
            "description": "Dense vs hybrid retrieval for RAG advisor",
            "status": "draft",
            "variants": [
                {"id": "control",     "name": "Dense retrieval",  "weight": 50},
                {"id": "treatment_a", "name": "Hybrid retrieval", "weight": 50},
            ],
            "salt": "rag_salt_2026",
            "start_date": None,
            "end_date": None,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "owner": "ml-team",
            "tags": ["rag", "ml"],
        },
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def list_experiments() -> List[Dict]:
    _ensure_exp_cache()
    return list(_exp_cache.values())


def get_experiment(exp_id: str) -> Optional[Dict]:
    _ensure_exp_cache()
    return _exp_cache.get(exp_id)


def create_experiment(data: Dict) -> Dict:
    global _exp_cache, _exp_cache_at
    exp_id = data.get("id") or data.get("name", "").lower().replace(" ", "_")
    now = _now_iso()
    exp = {**data, "id": exp_id, "created_at": now, "updated_at": now,
           "status": data.get("status", "draft")}
    exp.setdefault("salt", hashlib.sha256(exp_id.encode()).hexdigest()[:12])
    
    _exp_cache[exp_id] = exp
    _exp_cache_at = time.monotonic()
    
    if _FIRESTORE_AVAILABLE:
        try:
            _fs_client.collection(EXP_COLLECTION).document(exp_id).set(exp)
        except Exception as e:
            logger.error("Failed to persist experiment: %s", e)
    return exp


def assign_user(user_id: str, experiment_id: str) -> Dict:
    """
    Assign a user to an experiment variant.
    Returns assignment dict with: user_id, experiment_id, variant, assigned_at.
    """
    _ensure_exp_cache()
    exp = _exp_cache.get(experiment_id)
    if not exp:
        return {"user_id": user_id, "experiment_id": experiment_id,
                "variant": "control", "reason": "experiment_not_found"}

    if exp.get("status") not in ("running",):
        return {"user_id": user_id, "experiment_id": experiment_id,
                "variant": "control", "reason": f"experiment_status_{exp.get('status')}"}

    variant_id = _assign_variant(
        user_id, experiment_id,
        exp.get("salt", "default_salt"),
        exp.get("variants", [])
    )

    assignment = {
        "user_id":       user_id,
        "experiment_id": experiment_id,
        "variant":       variant_id,
        "assigned_at":   _now_iso(),
        "salt":          exp.get("salt"),
    }

    # Persist assignment to Firestore (best-effort)
    if _FIRESTORE_AVAILABLE:
        try:
            doc_id = f"{user_id}_{experiment_id}"
            _fs_client.collection(ASSIGN_COLLECTION).document(doc_id).set(
                assignment, merge=True
            )
        except Exception as e:
            logger.warning("Could not persist assignment: %s", e)

    return assignment


def update_experiment_status(exp_id: str, status: str) -> Optional[Dict]:
    """Update experiment status: draft | running | paused | completed"""
    _ensure_exp_cache()
    if exp_id not in _exp_cache:
        return None
    _exp_cache[exp_id]["status"] = status
    _exp_cache[exp_id]["updated_at"] = _now_iso()
    if _FIRESTORE_AVAILABLE:
        try:
            _fs_client.collection(EXP_COLLECTION).document(exp_id).update(
                {"status": status, "updated_at": _now_iso()}
            )
        except Exception as e:
            logger.error("Failed to update experiment status: %s", e)
    return _exp_cache[exp_id]
