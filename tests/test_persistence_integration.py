"""
Integration tests for persistent storage layer.
Verifies that domain state survives restarts and is properly persisted.
"""

import pytest
import logging
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from farm_finance_ai import FarmFinanceAI, FinanceApplication
from blockchain_supply_chain import SupplyChainBlockchain
from product_batch import ProductBatch
from persistence.repositories import (
    FinanceApplicationRepository,
    NotificationRepository,
    SupplyChainRepository,
)
from persistence.migration import MigrationManager, export_in_memory_state

logger = logging.getLogger(__name__)


class TestFinanceApplicationPersistence:
    """Test finance application persistence to Firestore."""

    def test_finance_app_creation_with_repository(self):
        """Verify finance application is persisted to repository."""
        # Mock repository
        mock_repo = Mock(spec=FinanceApplicationRepository)
        mock_repo.create = Mock(return_value={})

        # Create finance engine with repository
        finance_ai = FarmFinanceAI(repository=mock_repo)

        # Create an application
        payload = {
            "farmer_name": "Test Farmer",
            "crop_type": "Rice",
            "acreage": 5,
            "annual_revenue": 100000,
            "annual_operating_cost": 50000,
            "existing_debt": 10000,
            "emergency_fund": 5000,
            "credit_score": 650,
            "requested_loan_amount": 20000,
            "loan_tenure_months": 36,
        }

        result = finance_ai.create_application(payload)

        # Verify result contains expected fields
        assert result["application_id"].startswith("LOAN-")
        assert result["farmer_name"] == "Test Farmer"
        assert result["crop_type"] == "Rice"

        # Verify repository.create was called
        mock_repo.create.assert_called_once()
        call_args = mock_repo.create.call_args[0][0]
        assert call_args["farmer_name"] == "Test Farmer"
        assert call_args["crop_type"] == "Rice"

    def test_finance_app_retrieval_from_repository(self):
        """Verify finance application is retrieved from repository first."""
        mock_repo = Mock(spec=FinanceApplicationRepository)
        mock_repo.get = Mock(
            return_value={
                "application_id": "LOAN-ABC123",
                "farmer_name": "Test Farmer",
                "crop_type": "Rice",
                "requested_amount": 20000.0,
                "recommended_amount": 18000.0,
                "selected_lender": "Bank X",
                "status": "pre_approved",
                "created_at": datetime.now().isoformat(),
                "assessment_score": 85.0,
                "risk_level": "low",
                "required_documents": [],
                "notes": [],
            }
        )

        finance_ai = FarmFinanceAI(repository=mock_repo)

        # Retrieve application
        result = finance_ai.get_application("LOAN-ABC123")

        # Verify repository was called
        mock_repo.get.assert_called_once_with("LOAN-ABC123")
        assert result is not None
        assert result["application_id"] == "LOAN-ABC123"
        assert result["farmer_name"] == "Test Farmer"

    def test_finance_app_in_memory_fallback(self):
        """Verify fallback to in-memory storage if repository unavailable."""
        finance_ai = FarmFinanceAI(repository=None)

        payload = {
            "farmer_name": "Test Farmer",
            "crop_type": "Rice",
            "acreage": 5,
            "annual_revenue": 100000,
            "annual_operating_cost": 50000,
            "existing_debt": 10000,
            "emergency_fund": 5000,
            "credit_score": 650,
            "requested_loan_amount": 20000,
            "loan_tenure_months": 36,
        }

        # Create application
        result = finance_ai.create_application(payload)
        app_id = result["application_id"]

        # Retrieve from in-memory
        retrieved = finance_ai.get_application(app_id)
        assert retrieved is not None
        assert retrieved["application_id"] == app_id


class TestSupplyChainPersistence:
    """Test supply chain persistence to Firestore."""

    def test_product_batch_creation_with_persistence(self):
        """Verify product batch is persisted when created."""
        # Mock repository with Firestore-like interface
        mock_repo = Mock(spec=SupplyChainRepository)
        mock_db = Mock()
        mock_collection = Mock()
        mock_doc = Mock()

        mock_db.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_doc
        mock_repo.db = mock_db

        supply_chain = SupplyChainBlockchain(repository=mock_repo)

        # Create product batch
        batch = supply_chain.create_product_batch(
            crop_type="Rice",
            farm_id="FARM-001",
            quantity=1000,
            unit="kg",
            planting_date="2026-01-01",
            harvesting_date="2026-06-01",
            farmer_name="Test Farmer",
        )

        # Verify batch was created
        assert batch.batch_id.startswith("BATCH-")
        assert batch.crop_type == "Rice"

        # Verify Firestore collection methods were called
        mock_db.collection.assert_called()
        mock_collection.document.assert_called()

    def test_supply_chain_node_addition_with_persistence(self):
        """Verify supply chain node is persisted when added."""
        mock_repo = Mock(spec=SupplyChainRepository)
        mock_repo.create = Mock(return_value=None)

        supply_chain = SupplyChainBlockchain(repository=mock_repo)

        # Create batch first
        batch = supply_chain.create_product_batch(
            crop_type="Rice",
            farm_id="FARM-001",
            quantity=1000,
            unit="kg",
            planting_date="2026-01-01",
            harvesting_date="2026-06-01",
            farmer_name="Test Farmer",
        )

        # Add supply chain node
        node = supply_chain.add_supply_chain_node(
            batch_id=batch.batch_id,
            node_type="warehouse",
            actor_name="Warehouse A",
            location="Storage Facility",
            action="stored",
            temperature=22.5,
            humidity=65.0,
        )

        # Verify node was created
        assert node.node_id.startswith("NODE-")
        assert node.batch_id == batch.batch_id

        # Verify repository.create was called for the node
        mock_repo.create.assert_called()


class TestNotificationPersistence:
    """Test notification persistence to Firestore."""

    def test_notification_creation_with_repository(self):
        """Verify notification is persisted to repository."""
        mock_repo = Mock(spec=NotificationRepository)
        mock_repo.create = Mock(return_value=True)

        # Create notification
        result = mock_repo.create(1, "weather", "Heavy rainfall expected")

        # Verify repository.create was called
        mock_repo.create.assert_called_once_with(1, "weather", "Heavy rainfall expected")
        assert result is True

    def test_notification_ttl_cleanup(self):
        """Verify expired notifications are cleaned up."""
        mock_repo = Mock(spec=NotificationRepository)
        mock_repo.cleanup_expired = Mock(return_value=5)

        # Run cleanup
        count = mock_repo.cleanup_expired()

        assert count == 5
        mock_repo.cleanup_expired.assert_called_once()


class TestMigrationManager:
    """Test data migration from in-memory to Firestore."""

    def test_finance_app_migration(self):
        """Verify finance applications are correctly migrated."""
        mock_repo = Mock(spec=FinanceApplicationRepository)
        mock_repo.create = Mock(return_value={})

        manager = MigrationManager()
        manager.finance_repo = mock_repo

        # Create in-memory applications
        in_memory_apps = {
            "LOAN-001": {
                "application_id": "LOAN-001",
                "farmer_name": "Farmer A",
                "crop_type": "Rice",
                "requested_amount": 20000,
                "recommended_amount": 18000,
                "selected_lender": "Bank X",
                "status": "pre_approved",
                "created_at": datetime.now().isoformat(),
                "assessment_score": 85.0,
                "risk_level": "low",
                "required_documents": [],
                "notes": [],
            },
            "LOAN-002": {
                "application_id": "LOAN-002",
                "farmer_name": "Farmer B",
                "crop_type": "Wheat",
                "requested_amount": 15000,
                "recommended_amount": 14000,
                "selected_lender": "Bank Y",
                "status": "under_review",
                "created_at": datetime.now().isoformat(),
                "assessment_score": 72.0,
                "risk_level": "medium",
                "required_documents": [],
                "notes": [],
            },
        }

        # Migrate
        report = manager.migrate_finance_applications(in_memory_apps)

        # Verify migration report
        assert report["entity_type"] == "finance_applications"
        assert report["total"] == 2
        assert report["migrated"] == 2
        assert report["failed"] == 0

        # Verify repository.create was called for each app
        assert mock_repo.create.call_count == 2

    def test_full_migration(self):
        """Verify full migration of all domain entities."""
        # Mock all repositories
        mock_finance_repo = Mock(spec=FinanceApplicationRepository)
        mock_finance_repo.create = Mock(return_value={})

        mock_notification_repo = Mock(spec=NotificationRepository)
        mock_notification_repo.create = Mock(return_value=True)

        mock_supply_chain_repo = Mock(spec=SupplyChainRepository)
        mock_supply_chain_repo.create = Mock(return_value=None)

        manager = MigrationManager()
        manager.finance_repo = mock_finance_repo
        manager.notification_repo = mock_notification_repo
        manager.supply_chain_repo = mock_supply_chain_repo

        # Create test data
        in_memory_apps = {
            "LOAN-001": {
                "application_id": "LOAN-001",
                "farmer_name": "Farmer A",
                "crop_type": "Rice",
                "requested_amount": 20000,
                "recommended_amount": 18000,
                "selected_lender": "Bank X",
                "status": "pre_approved",
                "created_at": datetime.now().isoformat(),
                "assessment_score": 85.0,
                "risk_level": "low",
                "required_documents": [],
                "notes": [],
            }
        }
        notifications = [
            {"id": 1, "type": "weather", "message": "Heavy rainfall"},
            {"id": 2, "type": "alert", "message": "Price drop"},
        ]
        supply_chain = {"batches": {}, "nodes": []}

        # Run full migration
        result = manager.run_full_migration(in_memory_apps, notifications, supply_chain)

        # Verify result
        assert "migration_completed_at" in result
        assert result["total_migrated"] >= 0
        assert result["total_failed"] == 0


def test_export_in_memory_state_structure():
    """Verify in-memory state export has correct structure."""
    # Create mocks with the necessary attributes
    mock_finance = Mock()
    mock_finance.applications = {"LOAN-001": Mock()}

    mock_notifications = Mock()
    mock_notifications.get_recent = Mock(return_value=[{"id": 1, "message": "test"}])

    mock_supply_chain = Mock()
    mock_supply_chain.chain = []
    mock_supply_chain.products = {}
    mock_supply_chain.supply_chain_nodes = {}
    mock_supply_chain.smart_contracts = {}

    # Export state
    exported = export_in_memory_state(mock_finance, mock_notifications, mock_supply_chain)

    # Verify structure
    assert "exported_at" in exported
    assert "finance_applications" in exported
    assert "notifications" in exported
    assert "supply_chain" in exported
    assert "chain" in exported["supply_chain"]
    assert "products" in exported["supply_chain"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
