"""
Persistence models for serialization/deserialization.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import json


@dataclass
class PersistenceModel:
    """Base model for persistent entities."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Firestore storage."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersistenceModel":
        """Create instance from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "PersistenceModel":
        """Create instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class FinanceApplicationModel(PersistenceModel):
    """Model for persisted finance application."""

    application_id: str
    farmer_name: str
    crop_type: str
    requested_amount: float
    recommended_amount: float
    selected_lender: str
    status: str
    created_at: str
    assessment_score: float
    risk_level: str
    required_documents: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class NotificationModel(PersistenceModel):
    """Model for persisted notification."""

    notification_id: int
    alert_type: str
    message: str
    timestamp: str
    ttl_expiry: str  # ISO format timestamp for TTL cutoff


@dataclass
class SupplyChainNodeModel(PersistenceModel):
    """Model for persisted supply chain node."""

    node_id: str
    batch_id: str
    node_type: str
    actor_name: str
    location: str
    timestamp: str
    action: str
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    quality_check: Optional[str] = None
    notes: str = ""
    blockchain_hash: str = ""


@dataclass
class ProductBatchModel(PersistenceModel):
    """Model for persisted product batch."""

    batch_id: str
    crop_type: str
    farm_id: str
    quantity: float
    unit: str
    planting_date: str
    harvesting_date: str
    farmer_name: str
    certifications: list = field(default_factory=list)
    quality_score: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    blockchain_records: list = field(default_factory=list)
