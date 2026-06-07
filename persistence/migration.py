"""
Migration utilities for transitioning from in-memory storage to Firestore.
Provides helpers to export existing state and import it into persistent storage.
"""

import logging
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
            "migrated_ids": [],
        }

        for app_id, app_data in applications.items():
            try:
                app_dict = (
                    app_data.to_dict()
                    if hasattr(app_data, "to_dict")
                    else app_data
                )
                self.finance_repo.create(app_dict)
                report["migrated"] += 1
                report["migrated_ids"].append(app_id)
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
            "migrated_ids": [],
        }

        for idx, notification in enumerate(notifications):
            try:
                notification_id = notification.get("id", idx)
                alert_type = notification.get("type", "unknown")
                message = notification.get("message", "")

                self.notification_repo.create(notification_id, alert_type, message)
                report["migrated"] += 1
                report["migrated_ids"].append(notification_id)
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
            "migrated_batch_ids": [],
            "migrated_node_ids": [],
        }

        for batch_id, batch_data in supply_chain_data.get("batches", {}).items():
            try:
                batch_dict = (
                    batch_data.to_dict()
                    if hasattr(batch_data, "to_dict")
                    else batch_data
                )
                self.supply_chain_repo.db.collection("supply_chain_batches").document(
                    batch_id
                ).set(batch_dict)
                report["migrated_batches"] += 1
                report["migrated_batch_ids"].append(batch_id)
            except Exception as exc:
                report["failed"] += 1
                report["errors"].append({"batch_id": batch_id, "error": str(exc)})
                logger.error("Failed to migrate batch %s: %s", batch_id, exc)

        for node in supply_chain_data.get("nodes", []):
            try:
                node_dict = node.to_dict() if hasattr(node, "to_dict") else node
                batch_id = node_dict.get("batch_id")
                node_id = node_dict.get("node_id")

                if batch_id and node_id:
                    self.supply_chain_repo.create(node_dict)
                    report["migrated_nodes"] += 1
                    report["migrated_node_ids"].append(node_id)
            except Exception as exc:
                report["failed"] += 1
                report["errors"].append(
                    {"node_id": node.get("node_id"), "error": str(exc)}
                )
                logger.error("Failed to migrate supply chain node: %s", exc)

        logger.info(
            "Supply chain migration: %d batches, %d nodes migrated, %d failed",
            report["migrated_batches"],
            report["migrated_nodes"],
            report["failed"],
        )
        return report

    def _rollback(self, finance_report: Dict, notification_report: Dict, supply_chain_report: Dict) -> None:
        """Best-effort rollback of all migrated records on partial failure."""
        logger.warning("Rolling back partial migration...")

        for app_id in finance_report.get("migrated_ids", []):
            try:
                self.finance_repo.delete(app_id)
            except Exception as exc:
                logger.error("Rollback failed for finance app %s: %s", app_id, exc)

        for notification_id in notification_report.get("migrated_ids", []):
            try:
                self.notification_repo.delete(notification_id)
            except Exception as exc:
                logger.error("Rollback failed for notification %s: %s", notification_id, exc)

        for batch_id in supply_chain_report.get("migrated_batch_ids", []):
            try:
                self.supply_chain_repo.db.collection("supply_chain_batches").document(batch_id).delete()
            except Exception as exc:
                logger.error("Rollback failed for batch %s: %s", batch_id, exc)

        for node_id in supply_chain_report.get("migrated_node_ids", []):
            try:
                self.supply_chain_repo.delete(node_id)
            except Exception as exc:
                logger.error("Rollback failed for supply chain node %s: %s", node_id, exc)

        logger.warning("Rollback complete.")

    def run_full_migration(
        self,
        finance_apps: Dict[str, Any],
        notifications: List[Dict],
        supply_chain: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run full migration of all domain entities.
        On partial failure, rolls back all successfully migrated records
        and raises a RuntimeError to signal incomplete migration.

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

        finance_report = {"migrated": 0, "failed": 0, "errors": [], "migrated_ids": []}
        notification_report = {"migrated": 0, "failed": 0, "errors": [], "migrated_ids": []}
        supply_chain_report = {"migrated_batches": 0, "migrated_nodes": 0, "failed": 0, "errors": [], "migrated_batch_ids": [], "migrated_node_ids": []}

        try:
            finance_report = self.migrate_finance_applications(finance_apps)
            if finance_report["failed"] > 0:
                raise RuntimeError(
                    f"Finance migration had {finance_report['failed']} failure(s): {finance_report['errors']}"
                )

            notification_report = self.migrate_notifications(notifications)
            if notification_report["failed"] > 0:
                raise RuntimeError(
                    f"Notification migration had {notification_report['failed']} failure(s): {notification_report['errors']}"
                )

            supply_chain_report = self.migrate_supply_chain(supply_chain)
            if supply_chain_report["failed"] > 0:
                raise RuntimeError(
                    f"Supply chain migration had {supply_chain_report['failed']} failure(s): {supply_chain_report['errors']}"
                )

        except Exception as exc:
            self._rollback(finance_report, notification_report, supply_chain_report)
            raise RuntimeError(f"Full migration failed and was rolled back: {exc}") from exc

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
    return {
        "exported_at": datetime.now().isoformat(),
        "finance_applications": finance_engine.applications,
        "notifications": list(notification_store.get_recent()),
        "supply_chain": {
            "chain": supply_chain.chain,
            "products": supply_chain.products,
            "supply_chain_nodes": supply_chain.supply_chain_nodes,
            "smart_contracts": supply_chain.smart_contracts,
        },
    }