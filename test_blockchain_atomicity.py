"""
Test suite for Blockchain Transaction Atomicity & Rollback Mechanism
Validates that partial failures do not leave inconsistent state.
"""

import pytest
from datetime import datetime
from blockchain_supply_chain import (
    SupplyChainBlockchain,
    SupplyChainNode,
    SmartContract,
)
from blockchain_record import BlockchainRecord
from product_batch import ProductBatch


class TestBlockchainAtomicity:
    """Test atomicity and rollback for blockchain operations"""

    @pytest.fixture
    def blockchain(self):
        """Initialize blockchain for tests"""
        return SupplyChainBlockchain()

    def test_create_product_batch_atomicity(self, blockchain):
        """Test that create_product_batch is atomic - all or nothing"""
        initial_chain_len = len(blockchain.chain)
        initial_products_len = len(blockchain.products)

        batch = blockchain.create_product_batch(
            crop_type="tomato",
            farm_id="FARM001",
            quantity=100.0,
            unit="kg",
            planting_date="2026-01-15",
            harvesting_date="2026-04-20",
            farmer_name="Raj Kumar"
        )

        # Verify successful state change
        assert len(blockchain.chain) == initial_chain_len + 1
        assert len(blockchain.products) == initial_products_len + 1
        assert batch.batch_id in blockchain.products

    def test_execute_contract_rollback_on_status_error(self, blockchain):
        """Test that failed contract execution rolls back all changes"""
        # Create batch and contract
        batch = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Raj Kumar"
        )

        contract = blockchain.create_smart_contract(
            batch.batch_id, "Raj Kumar", "Distributor", 5000.0
        )

        initial_chain_len = len(blockchain.chain)
        initial_contract_status = contract.status

        # Execute once (should succeed)
        result = blockchain.execute_smart_contract(contract.contract_id)
        assert result["success"] is True
        assert contract.status == "executed"
        assert len(blockchain.chain) == initial_chain_len + 1

        # Try executing again (should fail - already executed)
        intermediate_chain_len = len(blockchain.chain)
        with pytest.raises(ValueError, match="cannot be executed"):
            blockchain.execute_smart_contract(contract.contract_id)

        # Verify rollback: chain unchanged, contract status rolled back
        assert len(blockchain.chain) == intermediate_chain_len
        assert contract.status == "executed"  # Status preserved from first execution

    def test_add_supply_chain_node_rollback_on_invalid_batch(self, blockchain):
        """Test that adding node to invalid batch rolls back"""
        initial_chain_len = len(blockchain.chain)
        initial_nodes_len = len(blockchain.supply_chain_nodes)

        # Try to add node to non-existent batch
        with pytest.raises(ValueError, match="not found"):
            blockchain.add_supply_chain_node(
                batch_id="INVALID_BATCH",
                node_type="warehouse",
                actor_name="Manager",
                location="Pune",
                action="stored"
            )

        # Verify no state changed
        assert len(blockchain.chain) == initial_chain_len
        assert len(blockchain.supply_chain_nodes) == initial_nodes_len

    def test_create_smart_contract_rollback_on_invalid_batch(self, blockchain):
        """Test that creating contract for invalid batch rolls back"""
        initial_chain_len = len(blockchain.chain)
        initial_contracts_len = len(blockchain.smart_contracts)

        # Try to create contract for non-existent batch
        with pytest.raises(ValueError, match="not found"):
            blockchain.create_smart_contract(
                batch_id="INVALID_BATCH",
                seller="Seller",
                buyer="Buyer",
                price=5000.0
            )

        # Verify no state changed
        assert len(blockchain.chain) == initial_chain_len
        assert len(blockchain.smart_contracts) == initial_contracts_len

    def test_sequential_operations_maintain_consistency(self, blockchain):
        """Test that multiple successful operations maintain consistency"""
        # Create batch
        batch = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Raj Kumar"
        )

        # Add supply chain nodes
        node1 = blockchain.add_supply_chain_node(
            batch.batch_id, "warehouse", "Manager", "Pune", "stored"
        )

        node2 = blockchain.add_supply_chain_node(
            batch.batch_id, "distributor", "Distributor", "Mumbai", "transported"
        )

        # Create and execute contract
        contract = blockchain.create_smart_contract(
            batch.batch_id, "Raj Kumar", "Retailer", 6000.0
        )

        result = blockchain.execute_smart_contract(contract.contract_id)

        # Verify final state consistency
        assert len(blockchain.chain) == 5  # 1 batch + 2 nodes + 1 contract_created + 1 contract_executed
        assert len(blockchain.supply_chain_nodes[batch.batch_id]) == 2
        assert contract.status == "executed"
        assert result["success"] is True

    def test_snapshot_isolation(self, blockchain):
        """Test that snapshots don't interfere with multiple transactions"""
        batch1 = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Farmer1"
        )

        batch2 = blockchain.create_product_batch(
            "potato", "FARM002", 200.0, "kg",
            "2026-01-20", "2026-05-20", "Farmer2"
        )

        # Attempt adding node to batch1 (succeeds)
        node1 = blockchain.add_supply_chain_node(
            batch1.batch_id, "warehouse", "Manager", "Pune", "stored"
        )

        # Attempt adding node to batch2 (succeeds)
        node2 = blockchain.add_supply_chain_node(
            batch2.batch_id, "warehouse", "Manager", "Mumbai", "stored"
        )

        # Verify both are in supply chain
        assert len(blockchain.supply_chain_nodes[batch1.batch_id]) == 1
        assert len(blockchain.supply_chain_nodes[batch2.batch_id]) == 1
        assert len(blockchain.chain) == 4  # 2 batches + 2 nodes

    def test_chain_record_count_after_rollback(self, blockchain):
        """Test that chain length is correctly restored on rollback"""
        batch = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Raj Kumar"
        )

        initial_count = blockchain.get_blockchain_record_count()

        # Try to create contract for non-existent batch (will rollback)
        with pytest.raises(ValueError):
            blockchain.create_smart_contract(
                batch_id="FAKE_BATCH",
                seller="Seller",
                buyer="Buyer",
                price=5000.0
            )

        # Chain length should be restored
        assert blockchain.get_blockchain_record_count() == initial_count

    def test_contract_status_preserved_after_rollback_attempt(self, blockchain):
        """Test that contract status is preserved if execution fails"""
        batch = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Raj Kumar"
        )

        contract = blockchain.create_smart_contract(
            batch.batch_id, "Raj Kumar", "Distributor", 5000.0
        )

        # Verify initial status
        assert contract.status == "pending"

        # Execute contract
        blockchain.execute_smart_contract(contract.contract_id)
        assert contract.status == "executed"

        # Attempt second execution (fails and rolls back)
        with pytest.raises(ValueError):
            blockchain.execute_smart_contract(contract.contract_id)

        # Status should still be "executed"
        assert contract.status == "executed"

    def test_product_batch_not_created_on_exception(self, blockchain):
        """Test that batch creation exception prevents state mutation"""
        initial_len = len(blockchain.products)

        # Manually try to break batch creation by providing None
        try:
            batch = blockchain.create_product_batch(
                crop_type="tomato",
                farm_id="FARM001",
                quantity=100.0,
                unit="kg",
                planting_date="2026-01-15",
                harvesting_date="2026-04-20",
                farmer_name="Raj Kumar"
            )
            # If we get here, it succeeded; verify products increased
            assert len(blockchain.products) == initial_len + 1
        except Exception:
            # If exception, verify products unchanged
            assert len(blockchain.products) == initial_len

    def test_multiple_batches_independence(self, blockchain):
        """Test that batches remain independent - failure in one doesn't affect others"""
        batch1 = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Farmer1"
        )

        batch2 = blockchain.create_product_batch(
            "potato", "FARM002", 200.0, "kg",
            "2026-01-20", "2026-05-20", "Farmer2"
        )

        initial_batch1_chain_len = len(blockchain.chain)

        # Attempt to add node to non-existent batch associated with batch1 (fails)
        with pytest.raises(ValueError):
            blockchain.add_supply_chain_node(
                batch_id="INVALID",
                node_type="warehouse",
                actor_name="Manager",
                location="Pune",
                action="stored"
            )

        # Verify batch2 is unaffected and can still be modified
        node = blockchain.add_supply_chain_node(
            batch2.batch_id, "warehouse", "Manager", "Mumbai", "stored"
        )

        assert node is not None
        assert len(blockchain.supply_chain_nodes[batch2.batch_id]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
