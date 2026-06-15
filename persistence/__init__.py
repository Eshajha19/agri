"""
Persistence layer for domain entities using Firestore backend.
Provides repository interfaces for finance, notifications, and supply chain domains.
Also provides offline-first sync layer with CRDT conflict resolution.
"""

from .repositories import (
    FinanceApplicationRepository,
    NotificationRepository,
    SupplyChainRepository,
)
from .models import (
    PersistenceModel,
    FinanceApplicationModel,
    NotificationModel,
    SupplyChainNodeModel,
)
from .offline_sync import (
    queue_write,
    get_pending,
    mark_synced,
    mark_failed,
    prune_synced,
    resolve_conflict,
    get_sync_stats,
    init_schema,
)

__all__ = [
    "FinanceApplicationRepository",
    "NotificationRepository",
    "SupplyChainRepository",
    "PersistenceModel",
    "FinanceApplicationModel",
    "NotificationModel",
    "SupplyChainNodeModel",
    "queue_write",
    "get_pending",
    "mark_synced",
    "mark_failed",
    "prune_synced",
    "resolve_conflict",
    "get_sync_stats",
    "init_schema",
]
