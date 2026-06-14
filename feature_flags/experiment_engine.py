"""
experiment_engine.py
─────────────────────
Deterministic user-to-variant assignment for A/B experiments.
"""

import hashlib
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from firebase_admin import firestore as fs_admin
    _fs_client = fs_admin.client()
    _FIRESTORE_AVAILABLE = True
except Exception as e:
    logger.warning("Firestore not available: %s", e)
    _fs_client = None
    _FIRESTORE_AVAILABLE = False


EXP_COLLECTION = "experiments"
ASSIGN_COLLECTION = "experiment_assignments"
CACHE_TTL_SECONDS = 300

CACHE_VERSION = 1

ASSIGNMENT_AUDIT = {
    "total_assignments": 0,
    "cache_version_mismatches": 0,
    "consistency_failures": 0,
}


_exp_cache: Dict[str, Dict] = {}
_exp_cache_at: float = 0.0
_exp_cache_lock = threading.Lock()


# -------------------------
# Helpers
# -------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _create_audit(user_id: str, experiment_id: str, variant_id: str, salt: str) -> Dict[str, Any]:
    return {
        "timestamp": _now_iso(),
        "user_id": user_id,
        "experiment_id": experiment_id,
        "variant": variant_id,
        "salt": salt,
    }

def _verify_assignment_consistency(
    user_id: str,
    experiment_id: str,
    variant_id: str,
    salt: str,
) -> bool:
    expected = _assign_variant(
        user_id,
        experiment_id,
        salt,
        _exp_cache.get(experiment_id, {}).get("variants", []),
    )

    return expected == variant_id


def _validate_cache_version(
    experiment: Dict,
) -> bool:
    return experiment.get(
        "cache_version",
        CACHE_VERSION,
    ) == CACHE_VERSION



# -------------------------
# Cache Layer
# -------------------------

def _load_experiments() -> Dict[str, Dict]:
    if not _FIRESTORE_AVAILABLE:
        return {}

    try:
        docs = _fs_client.collection(EXP_COLLECTION).stream()
        return {d.id: d.to_dict() for d in docs}
    except Exception as e:
        logger.error("Failed to load experiments: %s", e)
        return {}


def _ensure_cache():
    global _exp_cache, _exp_cache_at

    with _exp_cache_lock:
        if not _exp_cache or (time.monotonic() - _exp_cache_at) > CACHE_TTL_SECONDS:
            _exp_cache = _load_experiments()
            _exp_cache_at = time.monotonic()


# -------------------------
# Public API
# -------------------------

def list_experiments() -> List[Dict]:
    _ensure_cache()
    with _exp_cache_lock:
        return list(_exp_cache.values())


def get_experiment(exp_id: str) -> Optional[Dict]:
    _ensure_cache()
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

        logger.exception(
            "Failed to persist experiment '%s'",
            exp_id,
        )
        raise

    return deepcopy(exp)

def set_traffic_split(exp_id: str, traffic_split: Dict[str, int]) -> Optional[Dict]:
    _ensure_cache()

    with _exp_cache_lock:
        exp = _exp_cache.get(exp_id)
        if not exp:
            return None

        variants = exp.get("variants", [])
        if not variants:
            return None

        total = 0
        updated_variants = []

        for v in variants:
            vid = v["id"]

            if vid not in traffic_split:
                raise ValueError(f"Missing weight for {vid}")

            w = int(traffic_split[vid])
            if w < 0:
                raise ValueError("Negative weight not allowed")

            total += w
            updated_variants.append({**v, "weight": w})

        if total != 100:
            raise ValueError("Traffic split must sum to 100")

        exp["variants"] = updated_variants
        exp["traffic_split"] = traffic_split
        exp["updated_at"] = _now_iso()

    _persist(exp_id)
    return exp


def promote_winner(exp_id: str, winner_variant_id: str, reason: str = "auto") -> Optional[Dict]:
    _ensure_cache()

    with _exp_cache_lock:
        exp = _exp_cache.get(exp_id)
        if not exp:
            return None

        variants = exp.get("variants", [])

        if not any(v["id"] == winner_variant_id for v in variants):
            raise ValueError("Winner variant not found")

        exp["status"] = "completed"
        exp["winner_variant"] = winner_variant_id
        exp["promotion_reason"] = reason
        exp["updated_at"] = _now_iso()

        exp["variants"] = [
            {**v, "weight": 100 if v["id"] == winner_variant_id else 0}
            for v in variants
        ]

    _persist(exp_id)
    return exp


def assign_user(user_id: str, experiment_id: str) -> Dict:
    _ensure_cache()

    with _exp_cache_lock:
        exp = _exp_cache.get(experiment_id)

    if not exp:
        return {
            "user_id": user_id,
            "experiment_id": experiment_id,
            "variant": "control",
            "reason": "experiment_not_found",
        }

    if exp.get("status") == "completed":
        return {
            "user_id": user_id,
            "experiment_id": experiment_id,
            "variant": exp.get("winner_variant", "control"),
            "reason": "winner_promoted",
        }

    if exp.get("status") != "running":
        return {
            "user_id": user_id,
            "experiment_id": experiment_id,
            "variant": "control",
            "reason": "not_running",
        }
    
    if not _validate_cache_version(exp):
        ASSIGNMENT_AUDIT[
            "cache_version_mismatches"
        ] += 1

        logger.warning(
            "Cache version mismatch for experiment %s",
            experiment_id,
        )

    variant_id = _assign_variant(
        user_id,
        experiment_id,
        exp.get("salt", "default"),
        exp.get("variants", []),
    )

    is_consistent = _verify_assignment_consistency(
        user_id,
        experiment_id,
        variant_id,
        exp.get("salt", "default"),
    )

    if not is_consistent:
        ASSIGNMENT_AUDIT[
            "consistency_failures"
        ] += 1

        logger.warning(
            "Assignment consistency verification failed "
            "for user=%s experiment=%s",
            user_id,
            experiment_id,
        )

    assignment = {
        "user_id": user_id,
        "experiment_id": experiment_id,
        "variant": variant_id,
        "assigned_at": _now_iso(),
        "salt": exp.get("salt"),
        "assignment_hash": hashlib.sha256(
            f"{user_id}:{experiment_id}:{variant_id}".encode()
        ).hexdigest(),
        "audit": _create_audit(
            user_id, experiment_id, variant_id, exp.get("salt")
        ),
        "cache_version": CACHE_VERSION,
        "consistency_verified": is_consistent,
    }

    if _FIRESTORE_AVAILABLE:
        try:
            doc_id = f"{user_id}_{experiment_id}"
            doc_ref = _fs_client.collection(ASSIGN_COLLECTION).document(doc_id)
            existing = doc_ref.get()
            current_salt = exp.get("salt")

            if not existing.exists or existing.to_dict().get("salt") != current_salt:
                doc_ref.set(assignment)
        except Exception as e:
            logger.warning("Could not persist assignment: %s", e)
    _persist_assignment(user_id, experiment_id, assignment)

    logger.info(
        "Assignment | exp=%s user=%s variant=%s",
        experiment_id,
        user_id,
        variant_id,
    )

    return assignment


from typing import Optional, Dict
import copy

VALID_STATUSES = {
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
}


def update_experiment_status(exp_id: str, status: str) -> Optional[Dict]:
    """
    Update experiment status in cache and Firestore.

    Args:
        exp_id: Experiment identifier.
        status: New experiment status.

    Returns:
        Updated experiment dictionary or None if not found.
    """
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. "
            f"Allowed values: {', '.join(sorted(VALID_STATUSES))}"
        )

    _ensure_cache()

    timestamp = _now_iso()

    with _exp_cache_lock:
        experiment = _exp_cache.get(exp_id)
        if experiment is None:
            logger.warning("Experiment not found: %s", exp_id)
            return None

        experiment["status"] = status
        experiment["updated_at"] = timestamp

        updated_experiment = copy.deepcopy(experiment)

    if _FIRESTORE_AVAILABLE:
        try:
            _fs_client.collection(EXP_COLLECTION).document(exp_id).update(
                {
                    "status": status,
                    "updated_at": timestamp,
                }
            )
            logger.info(
                "Updated experiment %s status to %s",
                exp_id,
                status,
            )
        except Exception as e:
            logger.exception(
                "Failed to update Firestore status for experiment %s: %s",
                exp_id,
                e,
            )

    return updated_experiment
