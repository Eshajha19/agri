"""
Migration utilities for transitioning from in-memory storage to Firestore.
Provides helpers to export existing state and import it into persistent storage.
"""

import logging
from dataclasses import asdict
from typing import Dict, List, Any
from datetime import datetime

from .repositories import (
    FinanceApplicationRepository,
    NotificationRepository,
    SupplyChainRepository,
)

logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages data migration from in-memory to persistent storage."""

    def __init__(self):
        self.finance_repo = FinanceApplicationRepository()
        self.notification_repo = NotificationRepository()
        self.supply_chain_repo = SupplyChainRepository()

    def migrate_finance_applications(
        self, applications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Migrate in-memory finance applications to Firestore.

        Parameters
        ----------
        applications : dict
            Dictionary of in-memory finance applications keyed by ID.

        Returns
        -------
        dict
            Migration report with counts and status.
        """
        report = {
            "entity_type": "finance_applications",
            "total": len(applications),
            "migrated": 0,
            "failed": 0,
            "errors": [],
        }

        for app_id, app_data in applications.items():
            try:
                # Normalize to dict if needed
                app_dict = (
                    app_data.to_dict()
                    if hasattr(app_data, "to_dict")
                    else app_data
                )
                self.finance_repo.create(app_dict)
                report["migrated"] += 1
            except Exception as exc:
                report["failed"] += 1
                report["errors"].append({f"application_id": app_id, "error": str(exc)})
                logger.error("Failed to migrate application %s: %s", app_id, exc)

        logger.info(
            "Finance applications migration: %d migrated, %d failed",
            report["migrated"],
            report["failed"],
        )
        return report

    def migrate_notifications(self, notifications: List[Dict]) -> Dict[str, Any]:
        """
        Migrate in-memory notifications to Firestore.

        Parameters
        ----------
        notifications : list
            List of in-memory notification records.

        Returns
        -------
        dict
            Migration report with counts and status.
        """
        report = {
            "entity_type": "notifications",
            "total": len(notifications),
            "migrated": 0,
            "failed": 0,
            "errors": [],
        }

        for idx, notification in enumerate(notifications):
            try:
                notification_id = notification.get("id", idx)
                alert_type = notification.get("type", "unknown")
                message = notification.get("message", "")

                self.notification_repo.create(notification_id, alert_type, message)
                report["migrated"] += 1
            except Exception as exc:
                report["failed"] += 1
                report["errors"].append({"index": idx, "error": str(exc)})
                logger.error("Failed to migrate notification %d: %s", idx, exc)

        logger.info(
            "Notifications migration: %d migrated, %d failed",
            report["migrated"],
            report["failed"],
        )
        return report

    def migrate_supply_chain(self, supply_chain_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate in-memory supply chain records to Firestore.

        Parameters
        ----------
        supply_chain_data : dict
            Dictionary with 'nodes' and 'batches' keys containing supply chain data.

        Returns
        -------
        dict
            Migration report with counts and status.
        """
        report = {
            "entity_type": "supply_chain",
            "total_batches": len(supply_chain_data.get("batches", {})),
            "total_nodes": len(supply_chain_data.get("nodes", [])),
            "migrated_batches": 0,
            "migrated_nodes": 0,
            "failed": 0,
            "errors": [],
        }

        # Migrate batches (export key is "products")
        for batch_id, batch_data in supply_chain_data.get("products", {}).items():
            try:
                batch_dict = (
                    batch_data.to_dict()
                    if hasattr(batch_data, "to_dict")
                    else batch_data
                )
                # Assuming batch has an id and can be stored directly
                self.supply_chain_repo.db.collection("supply_chain_batches").document(
                    batch_id
                ).set(batch_dict)
                report["migrated_batches"] += 1
            except Exception as exc:
                report["failed"] += 1
                report["errors"].append({"batch_id": batch_id, "error": str(exc)})
                logger.error("Failed to migrate batch %s: %s", batch_id, exc)

        # Migrate nodes (export key is "supply_chain_nodes", value is dict keyed by batch_id)
        for batch_id, node_list in supply_chain_data.get("supply_chain_nodes", {}).items():
            for node in (node_list if isinstance(node_list, list) else [node_list]):
                try:
                    node_dict = node
                    batch_id = node_dict.get("batch_id")
                    node_id = node_dict.get("node_id")

                    if batch_id and node_id:
                        self.supply_chain_repo.create(node_dict)
                        report["migrated_nodes"] += 1
                except Exception as exc:
                    report["failed"] += 1
                    report["errors"].append(
                        {"node_id": node.get("node_id", ""), "error": str(exc)}
                    )
                    logger.error("Failed to migrate supply chain node: %s", exc)

        logger.info(
            "Supply chain migration: %d batches, %d nodes migrated, %d failed",
            report["migrated_batches"],
            report["migrated_nodes"],
            report["failed"],
        )
        return report

    def run_full_migration(
        self,
        finance_apps: Dict[str, Any],
        notifications: List[Dict],
        supply_chain: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run full migration of all domain entities.

        Parameters
        ----------
        finance_apps : dict
            In-memory finance applications.
        notifications : list
            In-memory notifications.
        supply_chain : dict
            In-memory supply chain data.

        Returns
        -------
        dict
            Aggregated migration report.
        """
        start_time = datetime.now()

        finance_report = self.migrate_finance_applications(finance_apps)
        notification_report = self.migrate_notifications(notifications)
        supply_chain_report = self.migrate_supply_chain(supply_chain)

        total_migrated = (
            finance_report["migrated"]
            + notification_report["migrated"]
            + supply_chain_report["migrated_batches"]
            + supply_chain_report["migrated_nodes"]
        )
        total_failed = (
            finance_report["failed"]
            + notification_report["failed"]
            + supply_chain_report["failed"]
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return {
            "migration_completed_at": end_time.isoformat(),
            "duration_seconds": duration,
            "total_migrated": total_migrated,
            "total_failed": total_failed,
            "reports": {
                "finance_applications": finance_report,
                "notifications": notification_report,
                "supply_chain": supply_chain_report,
            },
        }


def export_in_memory_state(
    finance_engine: Any, notification_store: Any, supply_chain: Any
) -> Dict[str, Any]:
    """
    Export current in-memory state for backup or migration.

    Parameters
    ----------
    finance_engine : FarmFinanceAI
        Current finance engine instance.
    notification_store : NotificationStore
        Current notification store instance.
    supply_chain : SupplyChainBlockchain
        Current supply chain blockchain instance.

    Returns
    -------
    dict
        Structured export of all in-memory state.
    """
    def _to_dict(obj: Any) -> Any:
        """Convert dataclass to dict via asdict, or passthrough for dicts."""
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return obj

    return {
        "exported_at": datetime.now().isoformat(),
        "finance_applications": _to_dict(finance_engine.applications),
        "notifications": list(notification_store.get_recent()),
        "supply_chain": {
            "chain": [_to_dict(r) for r in supply_chain.chain],
            "products": _to_dict(supply_chain.products),
            "supply_chain_nodes": _to_dict(supply_chain.supply_chain_nodes),
            "smart_contracts": _to_dict(supply_chain.smart_contracts),
        },
    }
