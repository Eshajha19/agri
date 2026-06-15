from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime, timezone

@dataclass
class ProductBatch:
    """Agricultural product batch"""
    batch_id: str
    crop_type: str
    farm_id: str
    quantity: float
    unit: str  # kg, tons, etc
    planting_date: str
    harvesting_date: str
    farmer_name: str
    owner_uid: str = ""
    certifications: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    created_at: str = ""
    blockchain_records: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
