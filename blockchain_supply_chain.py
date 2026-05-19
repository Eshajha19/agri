"""
Blockchain-Based Agricultural Supply Chain Traceability System
End-to-end transparency from farm to consumer
"""

import hashlib
import json
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
import qrcode
import io
import base64

logger = logging.getLogger(__name__)

def safe_json_serializer(obj: Any) -> str:
    """Custom serializer for non-JSON-native objects (like datetime)."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


@dataclass
class BlockchainRecord:
    """Record stored in blockchain"""
    timestamp: str
    actor: str
    action: str
    location: str
    data: Dict
    hash: str = ""
    previous_hash: str = ""

    def calculate_hash(self) -> str:
        """Calculate SHA256 hash of record"""
        record_string = json.dumps({
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "location": self.location,
            "data": self.data,
            "previous_hash": self.previous_hash
        }, sort_keys=True, default=safe_json_serializer)
        return hashlib.sha256(record_string.encode()).hexdigest()


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
    certifications: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    created_at: str = ""
    blockchain_records: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class SupplyChainNode:
    """Supply chain transaction node"""
    node_id: str
    batch_id: str
    node_type: str  # farm, warehouse, distributor, retailer, consumer
    actor_name: str
    location: str
    timestamp: str
    action: str  # harvested, stored, transported, verified, sold
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    quality_check: Optional[str] = None
    notes: str = ""


@dataclass
class SmartContract:
    """Smart contract for supply chain"""
    contract_id: str
    batch_id: str
    seller: str
    buyer: str
    price: float
    currency: str = "INR"
    terms: Dict = field(default_factory=dict)
    status: str = "pending"  # pending, executed, completed, disputed
    created_at: str = ""
    executed_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class SupplyChainBlockchain:
    """Blockchain for agricultural supply chain"""

    def __init__(self, repository: Any = None):
        """
        Initialize Supply Chain Blockchain with optional persistent repository.
        
        Parameters
        ----------
        repository : SupplyChainRepository, optional
            Persistent repository for storing supply chain records. If None, uses in-memory storage only.
        """
        self.chain: List[BlockchainRecord] = []
        self.pending_records: List[BlockchainRecord] = []
        self.products: Dict[str, ProductBatch] = {}
        self.supply_chain_nodes: Dict[str, List[SupplyChainNode]] = {}
        self.smart_contracts: Dict[str, SmartContract] = {}
        self.verified_actors: Dict[str, Dict] = {}
        self.repository = repository
        logger.info("SupplyChainBlockchain initialized with %s", "persistent repository" if repository else "in-memory storage")

    def register_actor(self, actor_id: str, name: str, actor_type: str, location: str) -> Dict:
        """Register supply chain participant"""
        actor_data = {
            "actor_id": actor_id,
            "name": name,
            "type": actor_type,  # farmer, warehouse, distributor, retailer
            "location": location,
            "registered_at": datetime.now().isoformat(),
            "verified": True,
            "transactions": 0,
            "rating": 5.0
        }
        self.verified_actors[actor_id] = actor_data
        return actor_data

    def create_product_batch(
        self,
        crop_type: str,
        farm_id: str,
        quantity: float,
        unit: str,
        planting_date: str,
        harvesting_date: str,
        farmer_name: str,
    ) -> ProductBatch:
        """Create new product batch"""
        batch_id = f"BATCH-{uuid.uuid4().hex[:12].upper()}"

        batch = ProductBatch(
            batch_id=batch_id,
            crop_type=crop_type,
            farm_id=farm_id,
            quantity=quantity,
            unit=unit,
            planting_date=planting_date,
            harvesting_date=harvesting_date,
            farmer_name=farmer_name,
        )

        self.products[batch_id] = batch
        self.supply_chain_nodes[batch_id] = []

        # Create initial blockchain record
        record = BlockchainRecord(
            timestamp=datetime.now().isoformat(),
            actor=farmer_name,
            action="created_batch",
            location=farm_id,
            data=asdict(batch)
        )
        if self.chain:
            record.previous_hash = self.chain[-1].hash
        record.hash = record.calculate_hash()
        self.chain.append(record)
        batch.blockchain_records.append(asdict(record))
        
        # Persist to repository if available
        if self.repository:
            try:
                batch_dict = asdict(batch)
                self.repository.db.collection("supply_chain_batches").document(batch_id).set(batch_dict)
                logger.info("Product batch %s persisted to repository.", batch_id)
            except Exception as exc:
                logger.error("Failed to persist product batch %s: %s", batch_id, exc)

        return batch

    def add_supply_chain_node(
        self,
        batch_id: str,
        node_type: str,
        actor_name: str,
        location: str,
        action: str,
        **kwargs
    ) -> SupplyChainNode:
        """Add node to supply chain"""
        if batch_id not in self.products:
            raise ValueError(f"Batch {batch_id} not found")

        node_id = f"NODE-{uuid.uuid4().hex[:12].upper()}"
        node = SupplyChainNode(
            node_id=node_id,
            batch_id=batch_id,
            node_type=node_type,
            actor_name=actor_name,
            location=location,
            timestamp=datetime.now().isoformat(),
            action=action,
            temperature=kwargs.get("temperature"),
            humidity=kwargs.get("humidity"),
            quality_check=kwargs.get("quality_check"),
            notes=kwargs.get("notes", "")
        )

        if batch_id not in self.supply_chain_nodes:
            self.supply_chain_nodes[batch_id] = []

        self.supply_chain_nodes[batch_id].append(node)

        # Record on blockchain
        record = BlockchainRecord(
            timestamp=node.timestamp,
            actor=actor_name,
            action=action,
            location=location,
            data=asdict(node)
        )
        if self.chain:
            record.previous_hash = self.chain[-1].hash
        record.hash = record.calculate_hash()
        self.chain.append(record)
        self.products[batch_id].blockchain_records.append(asdict(record))
        
        # Persist to repository if available
        if self.repository:
            try:
                node_dict = asdict(node)
                self.repository.create(node_dict)
                logger.info("Supply chain node %s (batch: %s) persisted to repository.", node_id, batch_id)
            except Exception as exc:
                logger.error("Failed to persist supply chain node %s: %s", node_id, exc)

        return node

    def create_smart_contract(
        self,
        batch_id: str,
        seller: str,
        buyer: str,
        price: float,
        terms: Optional[Dict] = None,
    ) -> SmartContract:
        """Create smart contract for transaction"""
        if batch_id not in self.products:
            raise ValueError(f"Batch {batch_id} not found")

        contract_id = f"CONTRACT-{uuid.uuid4().hex[:12].upper()}"
        contract = SmartContract(
            contract_id=contract_id,
            batch_id=batch_id,
            seller=seller,
            buyer=buyer,
            price=price,
            terms=terms or {}
        )

        self.smart_contracts[contract_id] = contract

        # Log contract on blockchain
        record = BlockchainRecord(
            timestamp=datetime.now().isoformat(),
            actor=seller,
            action="contract_created",
            location="contract",
            data=asdict(contract)
        )
        if self.chain:
            record.previous_hash = self.chain[-1].hash
        record.hash = record.calculate_hash()
        self.chain.append(record)

        return contract

    def execute_smart_contract(self, contract_id: str) -> Dict:
        """Execute smart contract"""
        if contract_id not in self.smart_contracts:
            raise ValueError(f"Contract {contract_id} not found")

        contract = self.smart_contracts[contract_id]
        if contract.status != "pending":
            raise ValueError(f"Contract {contract_id} cannot be executed (status: {contract.status})")

        contract.status = "executed"
        contract.executed_at = datetime.now().isoformat()

        # Log execution on blockchain
        record = BlockchainRecord(
            timestamp=datetime.now().isoformat(),
            actor=contract.buyer,
            action="contract_executed",
            location="contract",
            data={
                "contract_id": contract_id,
                "batch_id": contract.batch_id,
                "amount": contract.price,
                "currency": contract.currency
            }
        )
        if self.chain:
            record.previous_hash = self.chain[-1].hash
        record.hash = record.calculate_hash()
        self.chain.append(record)

        return {
            "success": True,
            "contract_id": contract_id,
            "executed_at": contract.executed_at,
            "amount": contract.price
        }

    def generate_qr_code(self, batch_id: str) -> str:
        """Generate QR code for product batch"""
        if batch_id not in self.products:
            raise ValueError(f"Batch {batch_id} not found")

        batch = self.products[batch_id]
        qr_data = {
            "batch_id": batch_id,
            "crop_type": batch.crop_type,
            "quantity": batch.quantity,
            "unit": batch.unit,
            "farmer": batch.farmer_name,
            "harvested": batch.harvesting_date,
            "verification_url": f"https://fasalsaathi.agri/verify/{batch_id}"
        }

        qr_code = qrcode.QRCode(version=1, box_size=10, border=5)
        qr_code.add_data(json.dumps(qr_data, default=safe_json_serializer))
        qr_code.make(fit=True)

        qr_image = qr_code.make_image(fill_color="black", back_color="white")
        qr_buffer = io.BytesIO()
        qr_image.save(qr_buffer, format="PNG")
        qr_base64 = base64.b64encode(qr_buffer.getvalue()).decode()

        return qr_base64

    def verify_batch(self, batch_id: str) -> Dict:
        """Verify product batch authenticity"""
        if batch_id not in self.products:
            return {"success": False, "message": "Batch not found"}

        batch = self.products[batch_id]
        records = self.supply_chain_nodes.get(batch_id, [])

        # Calculate verification score
        verification_score = 80.0  # Start with 80 base score

        # Check completeness of supply chain
        if len(records) >= 1:
            verification_score += 10

        # Check for quality verification
        quality_verifications = [
            r for r in records if r.quality_check == "passed"
        ]
        if quality_verifications:
            verification_score += 5

        # Check all actors are registered
        registered_count = 0
        for record in records:
            if record.actor_name in self.verified_actors:
                registered_count += 1
        
        if registered_count > 0:
            verification_score += 5

        # Check blockchain integrity
        blockchain_intact = self._verify_blockchain_integrity()
        if blockchain_intact:
            verification_score = min(100, verification_score + 5)

        return {
            "success": True,
            "batch_id": batch_id,
            "product": batch.crop_type,
            "quantity": batch.quantity,
            "farmer": batch.farmer_name,
            "verification_score": min(100, verification_score),
            "authenticated": verification_score >= 70,
            "blockchain_records": len(batch.blockchain_records),
            "supply_chain_nodes": len(records),
            "certifications": batch.certifications,
            "quality_score": batch.quality_score,
            "harvested_date": batch.harvesting_date
        }

    def get_supply_chain_journey(self, batch_id: str) -> Dict:
        """Get complete supply chain journey"""
        if batch_id not in self.products:
            raise ValueError(f"Batch {batch_id} not found")

        batch = self.products[batch_id]
        nodes = self.supply_chain_nodes.get(batch_id, [])

        journey = {
            "batch_id": batch_id,
            "product": batch.crop_type,
            "quantity": batch.quantity,
            "farmer": batch.farmer_name,
            "created_at": batch.created_at,
            "nodes": []
        }

        for node in nodes:
            journey["nodes"].append({
                "timestamp": node.timestamp,
                "actor": node.actor_name,
                "type": node.node_type,
                "location": node.location,
                "action": node.action,
                "temperature": node.temperature,
                "humidity": node.humidity,
                "quality_check": node.quality_check,
                "notes": node.notes
            })

        return journey

    def get_supply_chain_analytics(self, batch_id: str) -> Dict:
        """Get analytics for supply chain"""
        if batch_id not in self.products:
            raise ValueError(f"Batch {batch_id} not found")

        batch = self.products[batch_id]
        nodes = self.supply_chain_nodes.get(batch_id, [])
        contracts = [c for c in self.smart_contracts.values() if c.batch_id == batch_id]

        # Calculate metrics
        total_journey_time = 0
        if len(nodes) >= 2:
            start_time = datetime.fromisoformat(nodes[0].timestamp)
            end_time = datetime.fromisoformat(nodes[-1].timestamp)
            total_journey_time = (end_time - start_time).total_seconds() / 3600  # hours

        avg_temperature = None
        temps = [n.temperature for n in nodes if n.temperature is not None]
        if temps:
            avg_temperature = sum(temps) / len(temps)

        node_types = {}
        for node in nodes:
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1

        return {
            "batch_id": batch_id,
            "product": batch.crop_type,
            "total_journey_hours": round(total_journey_time, 2),
            "supply_chain_steps": len(nodes),
            "node_types_distribution": node_types,
            "average_temperature": round(avg_temperature, 2) if avg_temperature else None,
            "quality_verifications": len([n for n in nodes if n.quality_check]),
            "transactions": len(contracts),
            "final_price": contracts[-1].price if contracts else None
        }

    def _verify_blockchain_integrity(self) -> bool:
        """Verify blockchain hasn't been tampered with"""
        for i, record in enumerate(self.chain):
            if record.hash != record.calculate_hash():
                return False
            if i > 0 and record.previous_hash != self.chain[i - 1].hash:
                return False
        return True

    def get_blockchain_record_count(self) -> int:
        """Get total records in blockchain"""
        return len(self.chain)

    def get_certified_products(self) -> List[Dict]:
        """Get all certified products ready for marketplace"""
        certified = []
        for batch_id, batch in self.products.items():
            verification = self.verify_batch(batch_id)
            if verification.get("authenticated"):
                certified.append({
                    "batch_id": batch_id,
                    "product": batch.crop_type,
                    "quantity": batch.quantity,
                    "farmer": batch.farmer_name,
                    "verification_score": verification.get("verification_score"),
                    "certifications": batch.certifications,
                    "quality_score": batch.quality_score
                })
        return certified
