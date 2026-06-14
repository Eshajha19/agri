"""
experiment_engine.py
─────────────────────
Deterministic user-to-variant assignment for A/B experiments.
"""

import hashlib
import logging
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from firebase_admin import firestore as fs_admin
    _fs_client = fs_admin.client()
    _FIRESTORE_AVAILABLE = True
except Exception:
    _fs_client = None
    _FIRESTORE_AVAILABLE = False

EXP_COLLECTION    = "experiments"
ASSIGN_COLLECTION = "experiment_assignments"
CACHE_TTL_SECONDS = 300

_exp_cache: Dict[str, Dict] = {}
_exp_cache_at: float = 0.0
_exp_cache_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _persist_experiment(exp_id: str) -> None:
    if not _FIRESTORE_AVAILABLE:
        return
    with _exp_cache_lock:
        exp = _exp_cache.get(exp_id)
    if not exp:
        return
    try:
        _fs_client.collection(EXP_COLLECTION).document(exp_id).set(exp)
    except Exception as exc:
        logger.error("Failed to persist experiment '%s': %s", exp_id, exc)


def _assign_variant(user_id: str, experiment_id: str, salt: str,
                    variants: List[Dict]) -> str:
    if not variants:
        return "control"

    raw = f"{user_id}:{experiment_id}:{salt}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100

    cumulative = 0
    for variant in variants:
        cumulative += int(variant.get("weight", 0))
        if bucket < cumulative:
            return variant["id"]

    return variants[-1]["id"]


def _ensure_exp_cache():
    global _exp_cache, _exp_cache_at
    with _exp_cache_lock:
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


def list_experiments() -> List[Dict]:
    _ensure_exp_cache()
    with _exp_cache_lock:
        return list(_exp_cache.values())


def get_experiment(exp_id: str) -> Optional[Dict]:
    _ensure_exp_cache()
    with _exp_cache_lock:
        return _exp_cache.get(exp_id)


from typing import Dict
from copy import deepcopy
import hashlib
import re
import time

VALID_STATUSES = {
    "draft",
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
}


def create_experiment(data: Dict) -> Dict:
    """
    Create a new experiment.

    Raises:
        ValueError: If experiment already exists or input is invalid.
    """
    global _exp_cache_at

    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")

    exp_data = deepcopy(data)

    exp_id = exp_data.get("id")
    if not exp_id:
        name = exp_data.get("name", "").strip()
        if not name:
            raise ValueError("Either 'id' or 'name' must be provided")

        exp_id = re.sub(r"[^a-z0-9_]", "_", name.lower())
        exp_id = re.sub(r"_+", "_", exp_id).strip("_")

    status = exp_data.get("status", "draft")
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. "
            f"Allowed values: {sorted(VALID_STATUSES)}"
        )

    now = _now_iso()

    exp = {
        **exp_data,
        "id": exp_id,
        "status": status,
        "created_at": now,
        "updated_at": now,
        "salt": exp_data.get(
            "salt",
            hashlib.sha256(exp_id.encode("utf-8")).hexdigest()[:12],
        ),
    }

    with _exp_cache_lock:
        existing = _exp_cache.get(exp_id)

        if existing and existing.get("status") != "draft":
            raise ValueError(
                f"Experiment '{exp_id}' already exists "
                f"with status '{existing.get('status')}'"
            )

        _exp_cache[exp_id] = exp
        _exp_cache_at = time.monotonic()

    try:
        _persist_experiment(exp_id)
        logger.info("Created experiment: %s", exp_id)
    except Exception:
        # Optional rollback to keep cache and storage consistent
        with _exp_cache_lock:
            _exp_cache.pop(exp_id, None)

        logger.exception(
            "Failed to persist experiment '%s'",
            exp_id,
        )
        raise

    return deepcopy(exp)

def set_traffic_split(exp_id: str, traffic_split: Dict[str, int]) -> Optional[Dict]:
    _ensure_exp_cache()
    with _exp_cache_lock:
        exp = _exp_cache.get(exp_id)
        if not exp:
            return None

        variants = exp.get("variants", [])
        if not variants:
            return None

        normalized_weights = []
        total_weight = 0
        for variant in variants:
            variant_id = variant.get("id")
            if variant_id not in traffic_split:
                raise ValueError(f"Missing traffic weight for variant '{variant_id}'")
            weight = int(traffic_split[variant_id])
            if weight < 0:
                raise ValueError("Traffic weights must be non-negative")
            normalized_weights.append((variant_id, weight))
            total_weight += weight

        if total_weight <= 0:
            raise ValueError("Traffic split must allocate positive weight")
        if total_weight != 100:
            raise ValueError("Traffic split weights must sum to 100")

        exp["variants"] = [
            {**variant, "weight": dict(normalized_weights)[variant.get("id")]}
            for variant in variants
        ]
        exp["traffic_split"] = {vid: w for vid, w in normalized_weights}
        exp["updated_at"] = _now_iso()

    variants = exp.get("variants", [])
    if not variants:
        return None

    normalized_weights = []
    total_weight = 0
    for variant in variants:
        variant_id = variant.get("id")
        if variant_id not in traffic_split:
            raise ValueError(f"Missing traffic weight for variant '{variant_id}'")
        weight = int(traffic_split[variant_id])
        if weight < 0:
            raise ValueError("Traffic weights must be non-negative")
        normalized_weights.append((variant_id, weight))
        total_weight += weight

    if total_weight <= 0:
        raise ValueError("Traffic split must allocate positive weight")
    if total_weight != 100:
        raise ValueError("Traffic split weights must sum to 100")

    exp["variants"] = [
        {**variant, "weight": dict(normalized_weights)[variant.get("id")]}
        for variant in variants
    ]
    exp["traffic_split"] = {vid: w for vid, w in normalized_weights}
    exp["updated_at"] = _now_iso()
    _persist_experiment(exp_id)
    return exp


def promote_winner(exp_id: str, winner_variant_id: str, reason: str = "auto_winner_promotion") -> Optional[Dict]:
    _ensure_exp_cache()
    with _exp_cache_lock:
        exp = _exp_cache.get(exp_id)
        if not exp:
            return None

        variants = exp.get("variants", [])
        if not any(variant.get("id") == winner_variant_id for variant in variants):
            raise ValueError(f"Variant '{winner_variant_id}' not found in experiment '{exp_id}'")

        exp["status"] = "completed"
        exp["winner_variant"] = winner_variant_id
        exp["promotion_reason"] = reason
        exp["promoted_at"] = _now_iso()
        exp["updated_at"] = _now_iso()
        exp["variants"] = [
            {**variant, "weight": 100 if variant.get("id") == winner_variant_id else 0}
            for variant in variants
        ]
        exp["traffic_split"] = {
            variant.get("id"): (100 if variant.get("id") == winner_variant_id else 0)
            for variant in variants
        }

    _persist_experiment(exp_id)
    return exp


def assign_user(user_id: str, experiment_id: str) -> Dict:
    _ensure_exp_cache()
    with _exp_cache_lock:
        exp = _exp_cache.get(experiment_id)

    if not exp:
        return {"user_id": user_id, "experiment_id": experiment_id,
                "variant": "control", "reason": "experiment_not_found"}

    if exp.get("status") == "completed" and exp.get("winner_variant"):
        return {
            "user_id": user_id,
            "experiment_id": experiment_id,
            "variant": exp["winner_variant"],
            "reason": "winner_promoted",
        }

    if exp.get("status") not in ("running",):
        return {"user_id": user_id, "experiment_id": experiment_id,
                "variant": "control", "reason": f"experiment_status_{exp.get('status')}"}

    current_salt = exp.get("salt", "default_salt")
    variant_id = _assign_variant(
        user_id, experiment_id,
        current_salt,
        exp.get("variants", [])
    )

    assignment = {
        "user_id":       user_id,
        "experiment_id": experiment_id,
        "variant":       variant_id,
        "assigned_at":   _now_iso(),
        "salt":          current_salt,
    }

    if _FIRESTORE_AVAILABLE:
        try:
            doc_id = f"{user_id}_{experiment_id}"
            doc_ref = _fs_client.collection(ASSIGN_COLLECTION).document(doc_id)
            existing = doc_ref.get()
            if not existing.exists or existing.to_dict().get("salt") != current_salt:
                doc_ref.set(assignment)
        except Exception as e:
            logger.warning("Could not persist assignment: %s", e)

    return assignment


def update_experiment_status(exp_id: str, status: str) -> Optional[Dict]:
    _ensure_exp_cache()
    with _exp_cache_lock:
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
