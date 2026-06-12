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
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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

VALID_EVENT_TYPES = frozenset({
    "impression", "conversion", "error", "flag_evaluated", "custom"
})


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

    if _FIRESTORE_AVAILABLE:
        try:
            _fs_client.collection(EVENTS_COLLECTION).add(event)
        except Exception as e:
            logger.warning("Failed to log experiment event: %s", e)

    return event


def log_events_batch(events: List[Dict]) -> int:
    """Batch-log multiple events. Returns number of events successfully queued."""
    if not _FIRESTORE_AVAILABLE:
        return len(events)

    success = 0
    try:
        batch = _fs_client.batch()
        for raw in events[:100]:  # cap at 100 per batch
            ref = _fs_client.collection(EVENTS_COLLECTION).document()
            batch.set(ref, {
                "event_type":    raw.get("event_type", "custom"),
                "user_id":       raw.get("user_id", "anonymous"),
                "experiment_id": raw.get("experiment_id"),
                "variant":       raw.get("variant"),
                "flag_id":       raw.get("flag_id"),
                "metadata":      raw.get("metadata", {}),
                "session_id":    raw.get("session_id"),
                "timestamp":     raw.get("timestamp", _now_iso()),
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
        return _empty_metrics(experiment_id)

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

        # Compute conversion rates
        summary = {}
        for variant, c in counts.items():
            imp = c["impressions"]
            if imp == 0:
                summary[variant] = {
                    **c,
                    "conversion_rate": None,
                    "error_rate":      None,
                }
            else:
                summary[variant] = {
                    **c,
                    "conversion_rate": round(c["conversions"] / imp * 100, 2),
                    "error_rate":      round(c["errors"] / imp * 100, 2),
                }

        return {
            "experiment_id": experiment_id,
            "total_events":  total,
            "variants":      summary,
        }

    except Exception as e:
        logger.error("Failed to compute metrics for '%s': %s", experiment_id, e)
        return _empty_metrics(experiment_id)


def _empty_metrics(experiment_id: str) -> Dict:
    return {
        "experiment_id": experiment_id,
        "total_events":  0,
        "variants":      {},
    }
