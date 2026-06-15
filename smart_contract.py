from dataclasses import dataclass, field
from typing import Dict
from datetime import datetime, timezone

@dataclass
class SmartContract:
    """Smart contract for supply chain"""
    contract_id: str
    batch_id: str
    seller: str
    buyer: str
    price: float
    created_by_uid: str = ""
    currency: str = "INR"
    terms: Dict = field(default_factory=dict)
    status: str = "pending"  # pending, executed, completed, disputed
    created_at: str = ""
    executed_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
