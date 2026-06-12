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
