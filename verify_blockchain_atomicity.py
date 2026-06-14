"""
End-to-end verification of blockchain atomicity and transactional semantics.
Tests both backward compatibility and new atomicity features.
"""

import sys
from blockchain_supply_chain import (
    SupplyChainBlockchain,
    SupplyChainNode,
    SmartContract,
)
from blockchain_record import BlockchainRecord
from product_batch import ProductBatch


def test_basic_operations():
    """Test backward compatible basic operations"""
    blockchain = SupplyChainBlockchain()

    # Create batch
    batch = blockchain.create_product_batch(
        crop_type="tomato",
        farm_id="FARM001",
        quantity=100.0,
        unit="kg",
        planting_date="2026-01-15",
        harvesting_date="2026-04-20",
        farmer_name="Raj Kumar"
    )
    assert batch.batch_id.startswith("BATCH-"), "Batch ID not generated"
    assert len(blockchain.chain) == 1, "Chain record not created"
    print("[OK] Basic batch creation works")

    # Register actor
    actor = blockchain.register_actor("FARM001", "Raj Kumar", "farmer", "Maharashtra")
    assert actor["verified"] is True, "Actor not verified"
    print("[OK] Actor registration works")

    # Add supply chain node
    node = blockchain.add_supply_chain_node(
        batch_id=batch.batch_id,
        node_type="warehouse",
        actor_name="Manager",
        location="Pune",
        action="stored",
        temperature=22.5,
        humidity=65.0,
        quality_check="passed"
    )
    assert node.node_id.startswith("NODE-"), "Node ID not generated"
    assert len(blockchain.supply_chain_nodes[batch.batch_id]) == 1, "Node not added"
    print("[OK] Supply chain node addition works")

    # Create contract
    contract = blockchain.create_smart_contract(
        batch_id=batch.batch_id,
        seller="Raj Kumar",
        buyer="Distributor Co",
        price=5000.0
    )
    assert contract.contract_id.startswith("CONTRACT-"), "Contract ID not generated"
    assert contract.status == "pending", "Contract status not pending"
    print("[OK] Contract creation works")

    # Execute contract
    result = blockchain.execute_smart_contract(contract.contract_id)
    assert result["success"] is True, "Contract execution failed"
    assert contract.status == "executed", "Contract status not updated"
    print("[OK] Contract execution works")

    # Verify batch
    verification = blockchain.verify_batch(batch.batch_id)
    assert verification["success"] is True, "Batch verification failed"
    print("[OK] Batch verification works")

    # Get journey
    journey = blockchain.get_supply_chain_journey(batch.batch_id)
    assert len(journey["nodes"]) == 1, "Journey nodes not retrieved"
    print("[OK] Supply chain journey retrieval works")

    # Get analytics
    analytics = blockchain.get_supply_chain_analytics(batch.batch_id)
    assert analytics["product"] == "tomato", "Analytics product mismatch"
    print("[OK] Supply chain analytics works")

    return blockchain


def test_atomicity_and_rollback():
    """Test new atomicity features and rollback on failures"""
    blockchain = SupplyChainBlockchain()

    # Create batch
    batch = blockchain.create_product_batch(
        crop_type="wheat",
        farm_id="FARM002",
        quantity=500.0,
        unit="kg",
        planting_date="2026-02-01",
        harvesting_date="2026-06-01",
        farmer_name="Farmer Two"
    )
    print("[OK] Batch created for atomicity test")

    # Test atomicity: attempt to add node to invalid batch should not modify state
    initial_chain_len = len(blockchain.chain)
    try:
        blockchain.add_supply_chain_node(
            batch_id="INVALID_BATCH",
            node_type="warehouse",
            actor_name="Manager",
            location="Pune",
            action="stored"
        )
        print("[FAIL] Should have raised ValueError for invalid batch")
        return False
    except ValueError:
        if len(blockchain.chain) != initial_chain_len:
            print("[FAIL] Chain was modified after failed operation (rollback failed)")
            return False
        print("[OK] Invalid batch operation rolled back (chain unchanged)")

    # Test atomicity: attempt to create contract for invalid batch should not modify state
    initial_contracts_len = len(blockchain.smart_contracts)
    try:
        blockchain.create_smart_contract(
            batch_id="INVALID_BATCH",
            seller="Seller",
            buyer="Buyer",
            price=5000.0
        )
        print("[FAIL] Should have raised ValueError for invalid batch")
        return False
    except ValueError:
        if len(blockchain.smart_contracts) != initial_contracts_len:
            print("[FAIL] Contracts dict was modified after failed operation")
            return False
        print("[OK] Invalid contract operation rolled back (contracts unchanged)")

    # Test contract execution rollback: re-execution should fail and rollback
    contract = blockchain.create_smart_contract(
        batch_id=batch.batch_id,
        seller="Farmer Two",
        buyer="Buyer",
        price=10000.0
    )
    print("[OK] Contract created for execution atomicity test")

    # First execution should succeed
    result1 = blockchain.execute_smart_contract(contract.contract_id)
    assert result1["success"] is True, "First execution should succeed"
    assert contract.status == "executed", "Status should be executed"
    chain_len_after_first = len(blockchain.chain)
    print("[OK] First contract execution succeeded")

    # Second execution should fail and rollback
    try:
        blockchain.execute_smart_contract(contract.contract_id)
        print("[FAIL] Second execution should have failed")
        return False
    except ValueError:
        if len(blockchain.chain) != chain_len_after_first:
            print("[FAIL] Chain was modified on second execution rollback")
            return False
        if contract.status != "executed":
            print("[FAIL] Contract status was modified on rollback")
            return False
        print("[OK] Second execution rolled back (state preserved)")

    # Verify chain integrity
    if not blockchain._verify_blockchain_integrity():
        print("[FAIL] Blockchain integrity check failed")
        return False
    print("[OK] Blockchain integrity verified")

    return True


def test_multiple_independent_transactions():
    """Test that multiple batches remain independent"""
    blockchain = SupplyChainBlockchain()

    # Create two batches
    batch1 = blockchain.create_product_batch(
        "rice", "FARM001", 200.0, "kg", "2026-01-10", "2026-05-10", "Farmer A"
    )
    batch2 = blockchain.create_product_batch(
        "corn", "FARM002", 300.0, "kg", "2026-01-15", "2026-05-15", "Farmer B"
    )
    print("[OK] Two batches created")

    # Add nodes to both
    blockchain.add_supply_chain_node(batch1.batch_id, "warehouse", "Mgr1", "City1", "stored")
    blockchain.add_supply_chain_node(batch2.batch_id, "warehouse", "Mgr2", "City2", "stored")
    print("[OK] Nodes added to both batches")

    # Create and execute contract for batch1
    contract1 = blockchain.create_smart_contract(batch1.batch_id, "Farmer A", "Buyer1", 8000.0)
    result1 = blockchain.execute_smart_contract(contract1.contract_id)
    assert result1["success"] is True

    # Attempt invalid operation on batch1 supply chain
    try:
        blockchain.add_supply_chain_node(
            batch_id="INVALID",
            node_type="warehouse",
            actor_name="Mgr",
            location="City",
            action="stored"
        )
    except ValueError:
        pass

    # Verify batch2 is still intact and can be modified
    contract2 = blockchain.create_smart_contract(batch2.batch_id, "Farmer B", "Buyer2", 9000.0)
    result2 = blockchain.execute_smart_contract(contract2.contract_id)
    assert result2["success"] is True
    print("[OK] Multiple batches remain independent after errors")

    return True


def main():
    print("\n=== Blockchain Atomicity & Transaction Verification ===\n")

    print("Phase 1: Testing backward-compatible operations...")
    try:
        blockchain = test_basic_operations()
        print("\n[PASS] All backward-compatible operations work")
    except Exception as e:
        print(f"\n[FAIL] Basic operations failed: {e}")
        return False

    print("\nPhase 2: Testing atomicity and rollback features...")
    try:
        if not test_atomicity_and_rollback():
            print("[FAIL] Atomicity tests failed")
            return False
        print("\n[PASS] All atomicity tests passed")
    except Exception as e:
        print(f"\n[FAIL] Atomicity tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\nPhase 3: Testing multiple independent transactions...")
    try:
        if not test_multiple_independent_transactions():
            print("[FAIL] Independent transaction tests failed")
            return False
        print("\n[PASS] Multiple independent transactions work correctly")
    except Exception as e:
        print(f"\n[FAIL] Independent transaction tests failed: {e}")
        return False

    print("\n=== All verifications passed! ===\n")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
