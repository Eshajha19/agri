"""
Repository interfaces and implementations for persistent storage.
Uses Firestore as the backing store for all domain entities.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from .connections import get_firestore_client, firestore_manager
from .models import (
    FinanceApplicationModel,
    NotificationModel,
    SupplyChainNodeModel,
    ProductBatchModel,
)

logger = logging.getLogger(__name__)


class BaseRepository(ABC):
    """Abstract base repository for all domain repositories."""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.db = get_firestore_client()

    def refresh_client(self) -> None:
        """Refresh the Firestore client (useful after re-initialization)."""
        self.db = get_firestore_client()

    @abstractmethod
    def create(self, entity: Any) -> Dict[str, Any]:
        """Create a new entity."""
        pass

    @abstractmethod
    def get(self, entity_id: str) -> Optional[Any]:
        """Retrieve an entity by ID."""
        pass

    @abstractmethod
    def list(self, filters: Optional[Dict] = None) -> List[Any]:
        """List entities with optional filtering."""
        pass

    @abstractmethod
    def update(self, entity_id: str, data: Dict[str, Any]) -> bool:
        """Update an entity."""
        pass

    @abstractmethod
    def delete(self, entity_id: str) -> bool:
        """Delete an entity."""
        pass


class FinanceApplicationRepository(BaseRepository):
    """Repository for persisting finance applications."""

    def __init__(self):
        super().__init__("finance_applications")

    def create(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new finance application."""
        if self.db is None:
            logger.error("Firestore not available; application not persisted.")
            return application_data

        try:
            application_id = application_data.get("application_id")
            self.db.collection(self.collection_name).document(application_id).set(
                {
                    **application_data,
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                }
            )
            logger.info("Finance application %s persisted to Firestore.", application_id)
            return application_data
        except Exception as exc:
            logger.error("Failed to create finance application: %s", exc)
            return application_data

    def get(self, application_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a finance application by ID."""
        if self.db is None:
            logger.warning("Firestore not available; cannot retrieve application.")
            return None

        try:
            doc = (
                self.db.collection(self.collection_name)
                .document(application_id)
                .get()
            )
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as exc:
            logger.error("Failed to retrieve finance application: %s", exc)
            return None

    def list(self, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """List finance applications with optional filtering."""
        if self.db is None:
            logger.warning("Firestore not available; returning empty list.")
            return []

        try:
            query = self.db.collection(self.collection_name)

            if filters:
                if "farmer_name" in filters:
                    query = query.where("farmer_name", "==", filters["farmer_name"])
                if "status" in filters:
                    query = query.where("status", "==", filters["status"])

            docs = query.stream()
            return [doc.to_dict() for doc in docs]
        except Exception as exc:
            logger.error("Failed to list finance applications: %s", exc)
            return []

    def update(self, application_id: str, data: Dict[str, Any]) -> bool:
        """Update a finance application."""
        if self.db is None:
            logger.warning("Firestore not available; cannot update application.")
            return False

        try:
            self.db.collection(self.collection_name).document(application_id).update(
                {**data, "last_updated": datetime.now().isoformat()}
            )
            logger.info("Finance application %s updated in Firestore.", application_id)
            return True
        except Exception as exc:
            logger.error("Failed to update finance application: %s", exc)
            return False

    def delete(self, application_id: str) -> bool:
        """Delete a finance application."""
        if self.db is None:
            logger.warning("Firestore not available; cannot delete application.")
            return False

        try:
            self.db.collection(self.collection_name).document(application_id).delete()
            logger.info("Finance application %s deleted from Firestore.", application_id)
            return True
        except Exception as exc:
            logger.error("Failed to delete finance application: %s", exc)
            return False


class NotificationRepository(BaseRepository):
    """Repository for persisting notifications with TTL support."""

    def __init__(self, ttl_hours: int = 24):
        super().__init__("notifications")
        self.ttl_hours = ttl_hours

    def create(self, notification_id: int, alert_type: str, message: str) -> bool:
        """Create a new notification."""
        if self.db is None:
            logger.warning("Firestore not available; notification not persisted.")
            return False

        try:
            now = datetime.now()
            ttl_expiry = (now + timedelta(hours=self.ttl_hours)).isoformat()

            self.db.collection(self.collection_name).document(str(notification_id)).set(
                {
                    "notification_id": notification_id,
                    "alert_type": alert_type,
                    "message": message,
                    "timestamp": now.isoformat(),
                    "ttl_expiry": ttl_expiry,
                }
            )
            logger.info("Notification %s persisted to Firestore.", notification_id)
            return True
        except Exception as exc:
            logger.error("Failed to create notification: %s", exc)
            return False

    def get(self, notification_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a notification by ID."""
        if self.db is None:
            return None

        try:
            doc = self.db.collection(self.collection_name).document(notification_id).get()
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as exc:
            logger.error("Failed to retrieve notification: %s", exc)
            return None

    def list(self, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """List recent notifications excluding expired ones."""
        if self.db is None:
            return []

        try:
            now = datetime.now().isoformat()
            query = self.db.collection(self.collection_name).where(
                "ttl_expiry", ">=", now
            )

            if filters and "alert_type" in filters:
                query = query.where("alert_type", "==", filters["alert_type"])

            docs = query.order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
            return [doc.to_dict() for doc in docs]
        except Exception as exc:
            logger.error("Failed to list notifications: %s", exc)
            return []

    def cleanup_expired(self) -> int:
        """Delete expired notifications (maintenance task)."""
        if self.db is None:
            return 0

        try:
            now = datetime.now().isoformat()
            query = self.db.collection(self.collection_name).where("ttl_expiry", "<", now)
            docs = query.stream()
            count = 0
            for doc in docs:
                doc.reference.delete()
                count += 1
            logger.info("Cleaned up %d expired notifications.", count)
            return count
        except Exception as exc:
            logger.error("Failed to cleanup expired notifications: %s", exc)
            return 0

    def update(self, notification_id: str, data: Dict[str, Any]) -> bool:
        """Update a notification (rarely used)."""
        if self.db is None:
            return False

        try:
            self.db.collection(self.collection_name).document(notification_id).update(data)
            return True
        except Exception as exc:
            logger.error("Failed to update notification: %s", exc)
            return False

    def delete(self, notification_id: str) -> bool:
        """Delete a notification."""
        if self.db is None:
            return False

        try:
            self.db.collection(self.collection_name).document(notification_id).delete()
            return True
        except Exception as exc:
            logger.error("Failed to delete notification: %s", exc)
            return False


class SupplyChainRepository(BaseRepository):
    """Repository for persisting supply chain records."""

    def __init__(self):
        super().__init__("supply_chain_records")

    def create(self, record_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new supply chain record."""
        if self.db is None:
            logger.warning("Firestore not available; record not persisted.")
            return record_data

        try:
            node_id = record_data.get("node_id")
            batch_id = record_data.get("batch_id")

            # Store in nested collection: batches/{batch_id}/nodes/{node_id}
            self.db.collection("supply_chain_batches").document(batch_id).collection(
                "nodes"
            ).document(node_id).set(
                {
                    **record_data,
                    "created_at": datetime.now().isoformat(),
                }
            )
            logger.info(
                "Supply chain record %s (batch: %s) persisted to Firestore.",
                node_id,
                batch_id,
            )
            return record_data
        except Exception as exc:
            logger.error("Failed to create supply chain record: %s", exc)
            return record_data

    def get(self, batch_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a supply chain record by batch and node IDs."""
        if self.db is None:
            return None

        try:
            doc = (
                self.db.collection("supply_chain_batches")
                .document(batch_id)
                .collection("nodes")
                .document(node_id)
                .get()
            )
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as exc:
            logger.error("Failed to retrieve supply chain record: %s", exc)
            return None

    def list(self, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """List supply chain records with optional filtering."""
        if self.db is None:
            return []

        try:
            results = []
            batch_id = filters.get("batch_id") if filters else None

            if batch_id:
                docs = (
                    self.db.collection("supply_chain_batches")
                    .document(batch_id)
                    .collection("nodes")
                    .stream()
                )
                results = [doc.to_dict() for doc in docs]
            else:
                # List all batches and their nodes
                batches = self.db.collection("supply_chain_batches").stream()
                for batch_doc in batches:
                    nodes = batch_doc.reference.collection("nodes").stream()
                    results.extend([doc.to_dict() for doc in nodes])

            return results
        except Exception as exc:
            logger.error("Failed to list supply chain records: %s", exc)
            return []

    def update(self, batch_id: str, node_id: str, data: Dict[str, Any]) -> bool:
        """Update a supply chain record."""
        if self.db is None:
            return False

        try:
            self.db.collection("supply_chain_batches").document(batch_id).collection(
                "nodes"
            ).document(node_id).update(
                {**data, "last_updated": datetime.now().isoformat()}
            )
            logger.info(
                "Supply chain record %s (batch: %s) updated in Firestore.",
                node_id,
                batch_id,
            )
            return True
        except Exception as exc:
            logger.error("Failed to update supply chain record: %s", exc)
            return False

    def delete(self, batch_id: str, node_id: str) -> bool:
        """Delete a supply chain record."""
        if self.db is None:
            return False

        try:
            self.db.collection("supply_chain_batches").document(batch_id).collection(
                "nodes"
            ).document(node_id).delete()
            logger.info(
                "Supply chain record %s (batch: %s) deleted from Firestore.",
                node_id,
                batch_id,
            )
            return True
        except Exception as exc:
            logger.error("Failed to delete supply chain record: %s", exc)
            return False

    def save_actor(self, actor_id: str, actor_data: Dict[str, Any]) -> bool:
        """Persist a verified supply chain actor to Firestore."""
        if self.db is None:
            logger.warning("Firestore not available; actor %s not persisted.", actor_id)
            return False

        try:
            self.db.collection("supply_chain_actors").document(actor_id).set(
                {**actor_data, "saved_at": datetime.now().isoformat()}
            )
            logger.info("Supply chain actor %s persisted to Firestore.", actor_id)
            return True
        except Exception as exc:
            logger.error("Failed to persist supply chain actor %s: %s", actor_id, exc)
            return False
