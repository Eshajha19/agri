"""
metrics_collector.py
─────────────────────
Experiment event ingestion and metrics aggregation.

Event document shape (Firestore collection: experiment_events):
{
  "event_type":    "impression" | "conversion" | "error" | "flag_evaluated",
  "experiment_id": "yield_model_ab",
  "variant":       "treatment_a",
  "flag_id":       "rag_advisor_v2",   // for flag_evaluated events
  "user_id":       "uid_abc123",
  "metadata":      {},                 // arbitrary key-value pairs
  "timestamp":     "2026-05-16T...",
  "session_id":    "sess_xyz",
}
"""

import logging
import threading
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from collections import deque
logger = logging.getLogger(__name__)

try:
    from firebase_admin import firestore as fs_admin
    from google.cloud.firestore_v1.base_query import FieldFilter
    _fs_client = fs_admin.client()
    _FIRESTORE_AVAILABLE = True
except Exception:
    _fs_client = None
    _FIRESTORE_AVAILABLE = False
    FieldFilter = None

EVENTS_COLLECTION = "experiment_events"

_event_buffer: deque = deque(maxlen=10000)
_event_lock = threading.Lock()

_duplicate_events_detected = 0
_missing_metric_entries = 0

VALID_EVENT_TYPES = frozenset({
    "impression", "conversion", "error", "flag_evaluated", "custom"
})

REQUIRED_EVENT_FIELDS = frozenset({
    "event_type",
    "user_id",
    "timestamp",
})


def _is_duplicate_event(event: Dict[str, Any]) -> bool:
    """
    Lightweight duplicate detection against recent buffered events.
    """
    with _event_lock:
        return any(
            existing.get("event_type") == event.get("event_type")
            and existing.get("user_id") == event.get("user_id")
            and existing.get("experiment_id") == event.get("experiment_id")
            and existing.get("timestamp") == event.get("timestamp")
            for existing in _event_buffer
        )


def _validate_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate event completeness and consistency.
    """

    missing_fields = [
        field for field in REQUIRED_EVENT_FIELDS
        if not event.get(field)
    ]

    metadata_valid = isinstance(
        event.get("metadata", {}),
        dict,
    )

    return {
        "is_valid": not missing_fields and metadata_valid,
        "missing_fields": missing_fields,
        "metadata_valid": metadata_valid,
    }

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Public API ─────────────────────────────────────────────────────────────────

def log_event(
    event_type: str,
    user_id: str,
    experiment_id: Optional[str] = None,
    variant: Optional[str] = None,
    flag_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
    session_id: Optional[str] = None,
) -> Dict:
    """Persist a single experiment/flag event to Firestore."""
    if event_type not in VALID_EVENT_TYPES:
        event_type = "custom"

    event = {
        "event_type":    event_type,
        "user_id":       user_id,
        "experiment_id": experiment_id,
        "variant":       variant,
        "flag_id":       flag_id,
        "metadata":      metadata or {},
        "session_id":    session_id,
        "timestamp":     _now_iso(),
    }

    validation = _validate_event(event)

    if not validation["is_valid"]:
        global _missing_metric_entries
        _missing_metric_entries += 1

        logger.warning(
            "Invalid experiment metric detected: %s",
            validation,
        )

    if _is_duplicate_event(event):
        global _duplicate_events_detected
        _duplicate_events_detected += 1

        logger.warning(
            "Duplicate experiment metric detected for user=%s experiment=%s",
            user_id,
            experiment_id,
        )

    with _event_lock:
        _event_buffer.append(deepcopy(event))

    if _FIRESTORE_AVAILABLE:
        try:
            _fs_client.collection(EVENTS_COLLECTION).add(event)
        except Exception as e:
            logger.warning("Failed to log experiment event: %s", e)

    return event


def log_events_batch(events: List[Dict]) -> int:
    """Batch-log multiple events. Returns number of events successfully queued."""
    normalized_events: list[dict[str, Any]] = []
    for raw in events[:100]:  # cap at 100 per batch
        normalized_events.append({
            "event_type":    raw.get("event_type", "custom"),
            "user_id":       raw.get("user_id", "anonymous"),
            "experiment_id": raw.get("experiment_id"),
            "variant":       raw.get("variant"),
            "flag_id":       raw.get("flag_id"),
            "metadata":      raw.get("metadata", {}),
            "session_id":    raw.get("session_id"),
            "timestamp":     raw.get("timestamp", _now_iso()),
        })

    with _event_lock:
        _event_buffer.extend(deepcopy(normalized_events))

    if not _FIRESTORE_AVAILABLE:
        return len(normalized_events)

    success = 0
    try:
        batch = _fs_client.batch()
        for raw in normalized_events:
            ref = _fs_client.collection(EVENTS_COLLECTION).document()
            batch.set(ref, {
                "event_type":    raw["event_type"],
                "user_id":       raw["user_id"],
                "experiment_id": raw["experiment_id"],
                "variant":       raw["variant"],
                "flag_id":       raw["flag_id"],
                "metadata":      raw["metadata"],
                "session_id":    raw["session_id"],
                "timestamp":     raw["timestamp"],
            })
            success += 1
        batch.commit()
    except Exception as e:
        logger.error("Batch event log failed: %s", e)

    return success


def get_experiment_metrics(experiment_id: str) -> Dict:
    """
    Aggregate impression / conversion / error counts per variant.
    Returns a summary dict suitable for the dashboard.
    """
    if not _FIRESTORE_AVAILABLE:
        with _event_lock:
            events = [deepcopy(event) for event in _event_buffer if event.get("experiment_id") == experiment_id]
        return _aggregate_events(experiment_id, events)

    try:
        docs = (
            _fs_client.collection(EVENTS_COLLECTION)
            .where("experiment_id", "==", experiment_id)
            .stream()
        )

        counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"impressions": 0, "conversions": 0, "errors": 0}
        )
        total = 0

        for doc in docs:
            d = doc.to_dict()
            variant = d.get("variant", "unknown")
            etype = d.get("event_type", "")
            total += 1
            if etype == "impression":
                counts[variant]["impressions"] += 1
            elif etype == "conversion":
                counts[variant]["conversions"] += 1
            elif etype == "error":
                counts[variant]["errors"] += 1

        return _summarize_counts(
            experiment_id,
            total,
            counts,
        )


    except Exception as e:
        logger.error("Failed to compute metrics for '%s': %s", experiment_id, e)
        return _empty_metrics(experiment_id)


def _empty_metrics(experiment_id: str) -> Dict:
    return {
        "experiment_id": experiment_id,
        "total_events":  0,
        "variants":      {},
    }


def _summarize_counts(experiment_id: str, total: int, counts: Dict[str, Dict[str, int]]) -> Dict:
    summary = {}
    for variant, c in counts.items():
        imp = c["impressions"]
        if imp == 0:
            conversion_rate = 0.0
            error_rate = 0.0
        else:
            conversion_rate = round(c["conversions"] / imp * 100, 2)
            error_rate = round(c["errors"] / imp * 100, 2)
        summary[variant] = {
            **c,
            "conversion_rate": conversion_rate,
            "error_rate": error_rate,
        }

    validation_summary = {
        "missing_metric_entries": _missing_metric_entries,
        "duplicate_metric_entries": _duplicate_events_detected,
        "metadata_validation_enabled": True,
    }

    return {
        "experiment_id": experiment_id,
        "total_events": total,
        "variants": summary,
        "validation": validation_summary,
    }


def _aggregate_events(experiment_id: str, events: List[Dict[str, Any]]) -> Dict:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"impressions": 0, "conversions": 0, "errors": 0})
    total = 0
    for event in events:
        variant = event.get("variant", "unknown") or "unknown"
        etype = event.get("event_type", "")
        total += 1
        if etype == "impression":
            counts[variant]["impressions"] += 1
        elif etype == "conversion":
            counts[variant]["conversions"] += 1
        elif etype == "error":
            counts[variant]["errors"] += 1
    return _summarize_counts(experiment_id, total, counts)


def clear_events() -> None:
    with _event_lock:
        _event_buffer.clear()