import pytest
from datetime import datetime, timezone
from backend.blockchain_supply_chain import BlockchainRecord

def test_record_hash_consistency():
    record = BlockchainRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        actor="Farmer A",
        action="harvested",
        location="Farm 1",
        data={"crop": "wheat"},
    )
    record.hash = record.calculate_hash()
    serialized = record.serialize()

    # ✅ Rehydration should match original hash
    rehydrated = BlockchainRecord.from_dict(serialized)
    assert rehydrated.hash == record.hash

def test_tampering_detected():
    record = BlockchainRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        actor="Farmer A",
        action="harvested",
        location="Farm 1",
        data={"crop": "wheat"},
    )
    record.hash = record.calculate_hash()
    serialized = record.serialize()

    # Tamper with data
    serialized["data"]["crop"] = "rice"
    with pytest.raises(ValueError):
        BlockchainRecord.from_dict(serialized)
