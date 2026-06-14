import hashlib
import json
from typing import Dict
from dataclasses import dataclass

@dataclass
class BlockchainRecord:
    """Record stored in blockchain"""
    timestamp: str
    actor: str
    action: str
    location: str
    data: Dict
    previous_hash: str = ""
    hash: str = ""

    def to_dict(self) -> Dict:
        """Serialize record to dict (hash excluded — for calculate_hash input)"""
        return {
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "location": self.location,
            "data": self.data,
            "previous_hash": self.previous_hash,
        }

    def serialize(self) -> Dict:
        """Serialize record for persistence (includes hash)"""
        result = self.to_dict()
        result["hash"] = self.hash
        return result

    def calculate_hash(self) -> str:
        """Calculate SHA256 hash of record (excludes hash and previous_hash fields)"""
        record_string = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(record_string.encode()).hexdigest()

    @staticmethod
    def from_dict(data: Dict) -> 'BlockchainRecord':
        """Reconstruct record from dict, compute hash, and verify integrity"""
        record = BlockchainRecord(
            timestamp=data["timestamp"],
            actor=data["actor"],
            action=data["action"],
            location=data["location"],
            data=data.get("data", {}),
            previous_hash=data.get("previous_hash", ""),
        )
        computed = record.calculate_hash()
        if "hash" in data and data["hash"]:
            if data["hash"] != computed:
                raise ValueError(
                    f"Hash mismatch: stored hash '{data['hash']}' "
                    f"does not match computed hash '{computed}'. "
                    "Record has been tampered with."
                )
            record.hash = data["hash"]
        else:
            record.hash = computed
        return record
