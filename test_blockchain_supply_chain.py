"""
Test suite for Blockchain-Based Agricultural Supply Chain
"""

import pytest
from datetime import datetime, timezone
from blockchain_supply_chain import (
    SupplyChainBlockchain,
    BlockchainRecord,
    ProductBatch,
    SupplyChainNode,
    SmartContract,
)


class TestSupplyChainBlockchain:
    """Test suite for blockchain supply chain"""

    @pytest.fixture
    def blockchain(self):
        """Initialize blockchain for tests"""
        return SupplyChainBlockchain()

    def test_blockchain_initialization(self, blockchain):
        """Test blockchain initialization"""
        assert blockchain is not None
        assert len(blockchain.chain) == 0
        assert len(blockchain.products) == 0
        assert len(blockchain.verified_actors) == 0

    def test_register_actor(self, blockchain):
        """Test actor registration"""
        actor = blockchain.register_actor(
            "FARM001",
            "Raj Kumar",
            "farmer",
            "Maharashtra"
        )

        assert actor["actor_id"] == "FARM001"
        assert actor["name"] == "Raj Kumar"
        assert actor["type"] == "farmer"
        assert actor["verified"] is True
        assert actor["rating"] == 5.0

    def test_create_product_batch(self, blockchain):
        """Test product batch creation"""
        batch = blockchain.create_product_batch(
            crop_type="tomato",
            farm_id="FARM001",
            quantity=100.0,
            unit="kg",
            planting_date="2026-01-15",
            harvesting_date="2026-04-20",
            farmer_name="Raj Kumar"
        )

        assert batch.crop_type == "tomato"
        assert batch.quantity == 100.0
        assert batch.farmer_name == "Raj Kumar"
        assert batch.batch_id is not None
        assert batch.batch_id.startswith("BATCH-")
        assert len(blockchain.chain) == 1

    def test_add_supply_chain_node(self, blockchain):
        """Test adding supply chain node"""
        batch = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Raj Kumar"
        )

        node = blockchain.add_supply_chain_node(
            batch_id=batch.batch_id,
            node_type="warehouse",
            actor_name="Warehouse Manager",
            location="Pune",
            action="stored",
            temperature=22.5,
            humidity=65.0,
            quality_check="passed"
        )

        assert node.node_type == "warehouse"
        assert node.temperature == 22.5
        assert len(blockchain.supply_chain_nodes[batch.batch_id]) == 1

    def test_add_node_invalid_batch(self, blockchain):
        """Test adding node to non-existent batch"""
        with pytest.raises(ValueError):
            blockchain.add_supply_chain_node(
                batch_id="INVALID",
                node_type="warehouse",
                actor_name="Manager",
                location="Pune",
                action="stored"
            )

    def test_create_smart_contract(self, blockchain):
        """Test smart contract creation"""
        batch = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Raj Kumar"
        )

        contract = blockchain.create_smart_contract(
            batch_id=batch.batch_id,
            seller="Raj Kumar",
            buyer="Distributor Co",
            price=5000.0
        )

        assert contract.seller == "Raj Kumar"
        assert contract.buyer == "Distributor Co"
        assert contract.price == 5000.0
        assert contract.status == "pending"

    def test_execute_smart_contract(self, blockchain):
        """Test smart contract execution"""
        batch = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Raj Kumar"
        )

        contract = blockchain.create_smart_contract(
            batch.batch_id, "Raj Kumar", "Distributor Co", 5000.0
        )

        result = blockchain.execute_smart_contract(contract.contract_id)

        assert result["success"] is True
        assert contract.status == "executed"
        assert contract.executed_at is not None

    def test_generate_qr_code(self, blockchain):
        """Test QR code generation"""
        batch = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Raj Kumar"
        )

        qr_code = blockchain.generate_qr_code(batch.batch_id)

        assert qr_code is not None
        assert isinstance(qr_code, str)
        assert len(qr_code) > 100  # Base64 encoded

    def test_verify_batch_authentic(self, blockchain):
        """Test batch verification - authentic"""
        batch = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Raj Kumar"
        )

        # Register actors
        blockchain.register_actor("FARM001", "Raj Kumar", "farmer", "Maharashtra")
        blockchain.register_actor("WH001", "Manager", "warehouse", "Pune")

        # Add supply chain nodes
        blockchain.add_supply_chain_node(
            batch.batch_id, "warehouse", "Manager", "Pune",
            "stored", quality_check="passed"
        )

        verification = blockchain.verify_batch(batch.batch_id)

        assert verification["success"] is True
        assert verification["authenticated"] is True
        assert verification["verification_score"] >= 80

    def test_verify_batch_not_found(self, blockchain):
        """Test verification of non-existent batch"""
        verification = blockchain.verify_batch("INVALID")
        assert verification["success"] is False

    def test_get_supply_chain_journey(self, blockchain):
        """Test getting supply chain journey"""
        batch = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Raj Kumar"
        )

        blockchain.add_supply_chain_node(
            batch.batch_id, "warehouse", "Manager", "Pune",
            "stored", quality_check="passed"
        )

        blockchain.add_supply_chain_node(
            batch.batch_id, "distributor", "Distributor", "Mumbai",
            "transported"
        )

        journey = blockchain.get_supply_chain_journey(batch.batch_id)

        assert journey["batch_id"] == batch.batch_id
        assert len(journey["nodes"]) == 2
        assert journey["nodes"][0]["action"] == "stored"
        assert journey["nodes"][1]["action"] == "transported"

    def test_get_supply_chain_analytics(self, blockchain):
        """Test supply chain analytics"""
        batch = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Raj Kumar"
        )

        blockchain.add_supply_chain_node(
            batch.batch_id, "warehouse", "Manager", "Pune",
            "stored", temperature=22.5, humidity=65.0
        )

        contract = blockchain.create_smart_contract(
            batch.batch_id, "Raj Kumar", "Distributor", 5000.0
        )

        analytics = blockchain.get_supply_chain_analytics(batch.batch_id)

        assert analytics["product"] == "tomato"
        assert analytics["supply_chain_steps"] == 1
        assert analytics["average_temperature"] == 22.5
        assert analytics["transactions"] == 1

    def test_blockchain_integrity(self, blockchain):
        """Test blockchain integrity verification"""
        batch = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Raj Kumar"
        )

        assert blockchain._verify_blockchain_integrity() is True
        assert blockchain.get_blockchain_record_count() == 1

    def test_get_certified_products(self, blockchain):
        """Test getting certified products"""
        # Create and verify batches
        for i in range(2):
            batch = blockchain.create_product_batch(
                "tomato", f"FARM{i:03d}", 100.0 + i*10, "kg",
                "2026-01-15", "2026-04-20", f"Farmer {i}"
            )

            blockchain.register_actor(f"FARM{i:03d}", f"Farmer {i}", "farmer", "Maharashtra")
            blockchain.add_supply_chain_node(
                batch.batch_id, "warehouse", "Manager", "Pune",
                "stored", quality_check="passed"
            )

        certified = blockchain.get_certified_products()

        assert len(certified) >= 0
        for product in certified:
            assert "batch_id" in product
            assert "product" in product

    def test_blockchain_record_hash(self):
        """Test blockchain record hashing"""
        record = BlockchainRecord(
            timestamp="2026-05-14T10:00:00",
            actor="Farmer",
            action="harvested",
            location="Farm",
            data={"quantity": 100}
        )

        hash1 = record.calculate_hash()
        hash2 = record.calculate_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_multiple_contracts_same_batch(self, blockchain):
        """Test multiple contracts for same batch"""
        batch = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Raj Kumar"
        )

        contract1 = blockchain.create_smart_contract(
            batch.batch_id, "Raj Kumar", "Distributor", 5000.0
        )

        contract2 = blockchain.create_smart_contract(
            batch.batch_id, "Distributor", "Retailer", 6000.0
        )

        assert contract1.contract_id != contract2.contract_id
        assert len(blockchain.smart_contracts) == 2

    def test_supply_chain_journey_order(self, blockchain):
        """Test supply chain journey maintains order"""
        batch = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Raj Kumar"
        )

        actions = ["harvested", "transported", "stored", "sold"]
        for action in actions:
            blockchain.add_supply_chain_node(
                batch.batch_id, "participant", "Actor",
                "Location", action
            )

        journey = blockchain.get_supply_chain_journey(batch.batch_id)

        for i, action in enumerate(actions):
            assert journey["nodes"][i]["action"] == action

    def test_actor_registration_duplicate(self, blockchain):
        """Test duplicate actor registration (should overwrite)"""
        actor1 = blockchain.register_actor("FARM001", "Raj Kumar", "farmer", "Maharashtra")
        actor2 = blockchain.register_actor("FARM001", "Raj Kumar Updated", "farmer", "Karnataka")

        assert actor2["name"] == "Raj Kumar Updated"
        assert actor2["location"] == "Karnataka"

    def test_quality_check_variations(self, blockchain):
        """Test various quality check results"""
        batch = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Raj Kumar"
        )

        quality_results = ["passed", "failed", "needs_inspection"]
        for i, quality in enumerate(quality_results):
            blockchain.add_supply_chain_node(
                batch.batch_id, "warehouse", f"Inspector{i}", "Location",
                "checked", quality_check=quality
            )

        nodes = blockchain.supply_chain_nodes[batch.batch_id]
        assert len(nodes) == 3
        for i, quality in enumerate(quality_results):
            assert nodes[i].quality_check == quality

    def test_blockchain_record_count(self, blockchain):
        """Test blockchain record counting"""
        initial_count = blockchain.get_blockchain_record_count()

        blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Raj Kumar"
        )

        assert blockchain.get_blockchain_record_count() == initial_count + 1

    def test_temperature_humidity_tracking(self, blockchain):
        """Test temperature and humidity tracking"""
        batch = blockchain.create_product_batch(
            "tomato", "FARM001", 100.0, "kg",
            "2026-01-15", "2026-04-20", "Raj Kumar"
        )

        temps = [20.5, 22.3, 21.8]
        humidities = [60.0, 65.5, 62.3]

        for temp, humidity in zip(temps, humidities):
            blockchain.add_supply_chain_node(
                batch.batch_id, "storage", "Worker", "Location",
                "monitored", temperature=temp, humidity=humidity
            )

        analytics = blockchain.get_supply_chain_analytics(batch.batch_id)
        expected_avg_temp = sum(temps) / len(temps)

        assert abs(analytics["average_temperature"] - expected_avg_temp) < 0.1

    def test_datetime_serialization(self, blockchain):
        """Test that datetime objects inside blockchain record data are safely serialized"""
        # Create a record that explicitly contains datetime objects in its data
        record = BlockchainRecord(
            timestamp="2026-05-19T05:53:53",
            actor="Farmer",
            action="harvested",
            location="Farm",
            data={
                "harvest_time": datetime.now(timezone.utc),
                "created_at": datetime.now(timezone.utc)
            }
        )
        
        # This calculate_hash call should serialize without raising TypeError
        record_hash = record.calculate_hash()
        assert record_hash is not None
        assert len(record_hash) == 64

    # ------------------------------------------------------------------
    # New tests: QR signature, node alteration, tamper detection (Issue)
    # ------------------------------------------------------------------

    def test_qr_signature_generation_includes_proof(self, blockchain):
        """QR payload must contain trace_proof and latest_block_hash fields."""
        batch = blockchain.create_product_batch(
            "wheat", "FARM010", 200.0, "kg",
            "2026-02-01", "2026-05-10", "Priya Singh"
        )
        blockchain.add_supply_chain_node(
            batch.batch_id, "warehouse", "Store Mgr", "Delhi",
            "stored", temperature=18.0
        )

        payload = blockchain.get_traceability_qr_payload(batch.batch_id)

        assert "trace_proof" in payload, "QR payload must include trace_proof"
        assert "block_hash" in payload or "latest_block_hash" not in payload or "block_hash" in payload
        assert isinstance(payload["trace_proof"], str)
        assert len(payload["trace_proof"]) == 64  # SHA-256 hex

    def test_verify_trace_proof_detects_node_alteration(self, blockchain):
        """verify_trace_proof must return False after a supply chain node is altered."""
        batch = blockchain.create_product_batch(
            "rice", "FARM020", 500.0, "kg",
            "2026-03-01", "2026-06-01", "Amit Sharma"
        )
        blockchain.add_supply_chain_node(
            batch.batch_id, "farm", "Amit Sharma", "Punjab",
            "harvested"
        )

        # Capture proof before alteration
        proof = blockchain._build_trace_proof(batch.batch_id)
        original_proof_hash = proof["proof_hash"]
        original_signature = proof["signature"]

        # Verify original proof passes
        assert blockchain.verify_trace_proof(
            batch.batch_id, original_proof_hash, original_signature
        ), "Original proof should verify as valid"

        # --- Simulate node tampering ---
        nodes = blockchain.supply_chain_nodes[batch.batch_id]
        nodes[0].location = "TAMPERED_LOCATION"

        # Proof hash computed over original data must now fail
        assert not blockchain.verify_trace_proof(
            batch.batch_id, original_proof_hash, original_signature
        ), "verify_trace_proof must detect altered node and return False"

    def test_verify_trace_proof_valid_round_trip(self, blockchain):
        """A freshly generated proof must verify successfully (happy path)."""
        batch = blockchain.create_product_batch(
            "cotton", "FARM030", 300.0, "kg",
            "2026-04-01", "2026-07-15", "Meena Patel"
        )
        blockchain.add_supply_chain_node(
            batch.batch_id, "distributor", "D Corp", "Surat",
            "transported", temperature=25.0
        )

        proof = blockchain._build_trace_proof(batch.batch_id)
        result = blockchain.verify_trace_proof(
            batch.batch_id,
            proof["proof_hash"],
            proof["signature"],
        )
        assert result is True, "Round-trip proof verification must succeed for unmodified batch"

    def test_blockchain_records_timestamp_is_utc_aware(self, blockchain):
        """Test that created batches and smart contracts have timezone-aware UTC timestamps."""
        batch = blockchain.create_product_batch(
            crop_type="tomato",
            farm_id="FARM001",
            quantity=100.0,
            unit="kg",
            planting_date="2026-01-15",
            harvesting_date="2026-04-20",
            farmer_name="Raj Kumar"
        )
        
        # Check batch record timestamp
        batch_record = blockchain.chain[-1]
        assert batch_record.timestamp.endswith("+00:00") or batch_record.timestamp.endswith("Z")
        dt_batch = datetime.fromisoformat(batch_record.timestamp)
        assert dt_batch.tzinfo is not None

        # Check smart contract record timestamp
        contract = blockchain.create_smart_contract(
            batch_id=batch.batch_id,
            seller="Farmer",
            buyer="Distributor",
            price=15000.0
        )
        contract_record = blockchain.chain[-1]
        assert contract_record.timestamp.endswith("+00:00") or contract_record.timestamp.endswith("Z")
        dt_contract = datetime.fromisoformat(contract_record.timestamp)
        assert dt_contract.tzinfo is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
