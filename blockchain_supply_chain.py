"""
Blockchain-Based Agricultural Supply Chain Traceability System
with transaction atomicity and rollback support.
"""

import hashlib
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import qrcode
import io
import copy as _copy
import base64


@dataclass
class BlockchainRecord:
    """Record stored in blockchain"""
    timestamp: str
    actor: str
    action: str
    location: str
    data: Dict
    previous_hash: str = ""

    def to_dict(self) -> Dict:
        """Serialize record to dict (previous_hash excluded — matches calculate_hash input)"""
        return {
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "location": self.location,
            "data": self.data,
        }

    def calculate_hash(self) -> str:
        """Calculate SHA256 hash of record (excludes previous_hash field)"""
        record_string = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(record_string.encode()).hexdigest()

    @staticmethod
    def from_dict(data: Dict) -> 'BlockchainRecord':
        """Reconstruct record from dict, then compute and verify hash"""
        record = BlockchainRecord(
            timestamp=data["timestamp"],
            actor=data["actor"],
            action=data["action"],
            location=data["location"],
            data=data.get("data", {}),
        )
        if "previous_hash" in data:
            record.previous_hash = data["previous_hash"]
        return record


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
    """Blockchain for agricultural supply chain with basic atomicity"""

    def __init__(self, repository=None):
        self.chain: List[BlockchainRecord] = []
        self.products: Dict[str, ProductBatch] = {}
        self.supply_chain_nodes: Dict[str, List[SupplyChainNode]] = {}
        self.smart_contracts: Dict[str, SmartContract] = {}
        self.verified_actors: Dict[str, Dict] = {}
        self._trace_batches: Dict[str, Dict] = {}
        self._repository = repository

    # ------------- Utilities for atomicity -------------
    def _snapshot_state(self):
        """Create snapshot of current state for rollback"""
        return {
            "chain_len": len(self.chain),
            "products_copy": _copy.deepcopy(self.products),
            "supply_chain_nodes_copy": {k: list(v) for k, v in self.supply_chain_nodes.items()},
            "smart_contracts_copy": {k: v.status for k, v in self.smart_contracts.items()},
            "trace_batches_copy": _copy.deepcopy(self._trace_batches),
        }

    def _rollback_to_snapshot(self, snap):
        """Rollback state to snapshot point"""
        self.chain = self.chain[: snap["chain_len"]]
        self.products = _copy.deepcopy(snap["products_copy"])
        self.supply_chain_nodes = {k: list(v) for k, v in snap["supply_chain_nodes_copy"].items()}
        for cid, status in snap["smart_contracts_copy"].items():
            if cid in self.smart_contracts:
                self.smart_contracts[cid].status = status
        self._trace_batches = _copy.deepcopy(snap["trace_batches_copy"])

    # ------------- Core operations -------------
    def register_actor(self, actor_id: str, name: str, actor_type: str, location: str) -> Dict:
        """Register supply chain participant"""
        actor_data = {
            "actor_id": actor_id,
            "name": name,
            "type": actor_type,
            "location": location,
            "registered_at": datetime.now().isoformat(),
            "verified": True,
            "transactions": 0,
            "rating": 5.0,
        }
        self.verified_actors[actor_id] = actor_data
        if self._repository is not None:
            self._repository.save_actor(actor_id, actor_data)
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
        """Create new product batch atomically"""
        snap = self._snapshot_state()
        try:
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

            record = BlockchainRecord(
                timestamp=datetime.now().isoformat(),
                actor=farmer_name,
                action="created_batch",
                location=farm_id,
                data=asdict(batch),
            )
            record.previous_hash = record.calculate_hash()

            # Commit changes atomically
            self.products[batch_id] = batch
            self.supply_chain_nodes[batch_id] = []
            self.chain.append(record)
            batch.blockchain_records.append(record.to_dict())

            return batch

        except Exception:
            self._rollback_to_snapshot(snap)
            raise

    def add_supply_chain_node(
        self,
        batch_id: str,
        node_type: str,
        actor_name: str,
        location: str,
        action: str,
        **kwargs,
    ) -> SupplyChainNode:
        """Add node to supply chain atomically"""
        if batch_id not in self.products:
            raise ValueError(f"Batch {batch_id} not found")

        snap = self._snapshot_state()
        try:
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
                notes=kwargs.get("notes", ""),
            )

            record = BlockchainRecord(
                timestamp=node.timestamp,
                actor=actor_name,
                action=action,
                location=location,
                data=asdict(node),
            )
            record.previous_hash = record.calculate_hash()

            # Commit
            self.supply_chain_nodes.setdefault(batch_id, []).append(node)
            self.chain.append(record)
            self.products[batch_id].blockchain_records.append(record.to_dict())

            if self._repository is not None:
                self._repository.create(asdict(node))

            return node

        except Exception:
            self._rollback_to_snapshot(snap)
            raise

    def create_smart_contract(
        self,
        batch_id: str,
        seller: str,
        buyer: str,
        price: float,
        terms: Optional[Dict] = None,
    ) -> SmartContract:
        """Create smart contract for transaction atomically"""
        if batch_id not in self.products:
            raise ValueError(f"Batch {batch_id} not found")

        snap = self._snapshot_state()
        try:
            contract_id = f"CONTRACT-{uuid.uuid4().hex[:12].upper()}"
            contract = SmartContract(
                contract_id=contract_id,
                batch_id=batch_id,
                seller=seller,
                buyer=buyer,
                price=price,
                terms=terms or {},
            )

            record = BlockchainRecord(
                timestamp=datetime.now().isoformat(),
                actor=seller,
                action="contract_created",
                location="contract",
                data=asdict(contract),
            )
            record.previous_hash = record.calculate_hash()

            # Commit
            self.smart_contracts[contract_id] = contract
            self.chain.append(record)

            return contract

        except Exception:
            self._rollback_to_snapshot(snap)
            raise

    def execute_smart_contract(self, contract_id: str) -> Dict:
        """Execute smart contract atomically with rollback on failure"""
        if contract_id not in self.smart_contracts:
            raise ValueError(f"Contract {contract_id} not found")

        snap = self._snapshot_state()
        contract = self.smart_contracts[contract_id]
        try:
            if contract.status != "pending":
                raise ValueError(f"Contract {contract_id} cannot be executed (status: {contract.status})")

            # Prepare execution record first (may raise)
            record = BlockchainRecord(
                timestamp=datetime.now().isoformat(),
                actor=contract.buyer,
                action="contract_executed",
                location="contract",
                data={
                    "contract_id": contract_id,
                    "batch_id": contract.batch_id,
                    "amount": contract.price,
                    "currency": contract.currency,
                },
            )
            record.previous_hash = record.calculate_hash()

            # Commit state updates atomically
            contract.status = "executed"
            contract.executed_at = datetime.now().isoformat()
            self.chain.append(record)

            return {
                "success": True,
                "contract_id": contract_id,
                "executed_at": contract.executed_at,
                "amount": contract.price,
            }

        except Exception:
            # rollback
            self._rollback_to_snapshot(snap)
            raise

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
            "verification_url": f"https://fasalsaathi.agri/verify/{batch_id}",
        }

        qr_code = qrcode.QRCode(version=1, box_size=10, border=5)
        qr_code.add_data(json.dumps(qr_data))
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

        verification_score = 80.0
        if len(records) >= 1:
            verification_score += 10

        quality_verifications = [r for r in records if r.quality_check == "passed"]
        if quality_verifications:
            verification_score += 5

        registered_count = 0
        for record in records:
            if record.actor_name in self.verified_actors:
                registered_count += 1

        if registered_count > 0:
            verification_score += 5

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
            "harvested_date": batch.harvesting_date,
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
            "nodes": [],
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
                "notes": node.notes,
            })

        return journey

    def get_supply_chain_analytics(self, batch_id: str) -> Dict:
        """Get analytics for supply chain"""
        if batch_id not in self.products:
            raise ValueError(f"Batch {batch_id} not found")

        batch = self.products[batch_id]
        nodes = self.supply_chain_nodes.get(batch_id, [])
        contracts = [c for c in self.smart_contracts.values() if c.batch_id == batch_id]

        total_journey_time = 0
        if len(nodes) >= 2:
            start_time = datetime.fromisoformat(nodes[0].timestamp)
            end_time = datetime.fromisoformat(nodes[-1].timestamp)
            total_journey_time = (end_time - start_time).total_seconds() / 3600

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
            "final_price": contracts[-1].price if contracts else None,
        }

    def _verify_blockchain_integrity(self) -> bool:
        """Verify blockchain hasn't been tampered with"""
        for record in self.chain:
            if record.previous_hash != record.calculate_hash():
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
                    "quality_score": batch.quality_score,
                })
        return certified

    # ------------- QR Traceability (farmer-facing) -------------

    def register_trace_batch(self, payload: Dict) -> Dict:
        """Store a QR-traceability batch submitted from the frontend.

        These batches are distinct from the supply-chain ProductBatch
        objects — they carry the farmer-entered journey data that
        consumers see when they scan a QR code.  Storing them here
        (server-side) means the data cannot be tampered with via
        DevTools or by clearing browser storage.
        """
        batch_id = payload.get("id")
        if not batch_id:
            raise ValueError("Batch ID is required")
        if batch_id in self._trace_batches:
            raise ValueError(f"Batch {batch_id} is already registered")

        entry = {
            "id": batch_id,
            "crop": payload.get("crop", ""),
            "variety": payload.get("variety", ""),
            "harvestDate": payload.get("harvestDate", ""),
            "farm": payload.get("farm", ""),
            "status": payload.get("status", "Pending Verification"),
            "registeredByUid": payload.get("registeredByUid", ""),
            "registeredAt": datetime.utcnow().isoformat() + "Z",
            "journey": payload.get("journey", []),
        }
        self._trace_batches[batch_id] = entry

        # Also record the registration on the blockchain for auditability.
        record = BlockchainRecord(
            timestamp=entry["registeredAt"],
            actor=entry["registeredByUid"] or "unknown",
            action="trace_batch_registered",
            location=entry["farm"],
            data={"batch_id": batch_id, "crop": entry["crop"]},
        )
        record.previous_hash = record.calculate_hash()
        self.chain.append(record)

        return entry

    def get_trace_batch(self, batch_id: str) -> Optional[Dict]:
        """Fetch a QR-traceability batch by ID.  Returns None if not found."""
        return self._trace_batches.get(batch_id)
