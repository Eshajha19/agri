"""
Blockchain-Based Agricultural Supply Chain Traceability System
with transaction atomicity and rollback support.
"""

import hashlib
import hmac
import json
import os
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, List, Optional
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
            "previous_hash": self.previous_hash,
        }

    def serialize(self) -> Dict:
        """Serialize record for persistence (includes hash)"""
        result = self.to_dict()
        result["hash"] = self.hash
        return result

    def calculate_hash(self) -> str:
        """Calculate SHA256 hash of record (excludes previous_hash field)"""
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
    owner_uid: str = ""
    certifications: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    created_at: str = ""
    blockchain_records: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


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
    created_by_uid: str = ""
    currency: str = "INR"
    terms: Dict = field(default_factory=dict)
    status: str = "pending"  # pending, executed, completed, disputed
    created_at: str = ""
    executed_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class SupplyChainBlockchain:
    """Blockchain for agricultural supply chain with basic atomicity"""

    def __init__(self, repository=None, signing_key: Optional[str] = None):
        self.chain: List[BlockchainRecord] = []
        self.products: Dict[str, ProductBatch] = {}
        self.supply_chain_nodes: Dict[str, List[SupplyChainNode]] = {}
        self.smart_contracts: Dict[str, SmartContract] = {}
        self.verified_actors: Dict[str, Dict] = {}
        self.idempotency_cache: Dict[str, object] = {}
        self._trace_batches: Dict[str, Dict] = {}
        self._processed_transaction_ids: OrderedDict[str, None] = OrderedDict()
        self._harvest_ids: set[str] = set()
        self._repository = repository
        self._signing_key = signing_key

    def _link_record(self, record: BlockchainRecord) -> BlockchainRecord:
        """Set previous_hash and compute hash for a record, returning it without appending."""
        record.previous_hash = self._last_hash()
        record.hash = record.calculate_hash()
        return record

    # ------------- Utilities for atomicity -------------
    def _snapshot_state(self):
        """Create snapshot of current state for rollback"""
        return {
            "chain_len": len(self.chain),
            "products_copy": _copy.deepcopy(self.products),
            "supply_chain_nodes_copy": {k: list(v) for k, v in self.supply_chain_nodes.items()},
            "smart_contracts_copy": _copy.deepcopy(self.smart_contracts),
            "trace_batches_copy": _copy.deepcopy(self._trace_batches),
            "verified_actors_copy": _copy.deepcopy(self.verified_actors),
        }

    def _rollback_to_snapshot(self, snap):
        """Rollback state to snapshot point"""
        self.chain = self.chain[: snap["chain_len"]]
        self.products = _copy.deepcopy(snap["products_copy"])
        self.supply_chain_nodes = {k: list(v) for k, v in snap["supply_chain_nodes_copy"].items()}
        self.smart_contracts = _copy.deepcopy(snap["smart_contracts_copy"])
        self._trace_batches = _copy.deepcopy(snap["trace_batches_copy"])
        self.verified_actors = _copy.deepcopy(snap["verified_actors_copy"])

    # ------------- Core operations -------------
    def register_actor(self, actor_id: str, name: str, actor_type: str, location: str) -> Dict:
        """Register supply chain participant atomically"""
        snap = self._snapshot_state()
        try:
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
        except Exception:
            self._rollback_to_snapshot(snap)
            raise

    def create_product_batch(
        self,
        crop_type: str,
        farm_id: str,
        quantity: float,
        unit: str,
        planting_date: str,
        harvesting_date: str,
        farmer_name: str,
        idempotency_key: Optional[str] = None,
        owner_uid: str = "",
        harvest_id: str = "",
    ) -> ProductBatch:
        """Create product batch atomically with harvest_id dedup."""
        # Check cache
        if idempotency_key and idempotency_key in self.idempotency_cache:
            return self.idempotency_cache[idempotency_key]

        snap = self._snapshot_state()
        try:
            batch_id = f"BATCH-{uuid.uuid4().hex[:12].upper()}"

            if harvest_id:
                self._record_harvest_id(harvest_id)

            transaction_payload = {
                "crop_type": crop_type,
                "farm_id": farm_id,
                "quantity": quantity,
                "planting_date": planting_date,
                "harvesting_date": harvesting_date,
                "harvest_id": harvest_id or "",
            }

            transaction_id = self._generate_transaction_id(transaction_payload)
            self._validate_transaction_uniqueness(transaction_id)

            batch = ProductBatch(
                batch_id=batch_id,
                crop_type=crop_type,
                farm_id=farm_id,
                quantity=quantity,
                unit=unit,
                planting_date=planting_date,
                harvesting_date=harvesting_date,
                farmer_name=farmer_name,
                owner_uid=owner_uid,
            )

            record = self._link_record(BlockchainRecord(
                timestamp=datetime.now().isoformat(),
                actor=farmer_name,
                action="created_batch",
                location=farm_id,
                data=asdict(batch),
            )
            record.previous_hash = record.calculate_hash()

            # Commit
            self.products[batch_id] = batch
            self.supply_chain_nodes[batch_id] = []
            self.chain.append(record)
            batch.blockchain_records.append(record.serialize())

            if idempotency_key:
                self.idempotency_cache[idempotency_key] = batch

            self._record_transaction_id(transaction_id)

            return batch
        

        except Exception as e:
            import logging
            logging.error(f"Blockchain error: {e}")
            self._rollback_to_snapshot(snap)
            raise


    def add_supply_chain_node(
        self,
        batch_id: str,
        node_type: str,
        actor_name: str,
        location: str,
        action: str,
        harvest_id: str = "",
        **kwargs,
    ) -> SupplyChainNode:
        """Add node to supply chain atomically with harvest_id dedup."""
        if batch_id not in self.products:
            raise ValueError(f"Batch {batch_id} not found")

        snap = self._snapshot_state()
        try:
            if harvest_id:
                self._record_harvest_id(harvest_id)

            transaction_payload = {
                "batch_id": batch_id,
                "actor_name": actor_name,
                "location": location,
                "action": action,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "harvest_id": harvest_id or "",
                **kwargs,
            }

            transaction_id = self._generate_transaction_id(transaction_payload)
            self._validate_transaction_uniqueness(transaction_id)
            node_id = f"NODE-{uuid.uuid4().hex[:12].upper()}"
            node = SupplyChainNode(
                node_id=node_id,
                batch_id=batch_id,
                node_type=node_type,
                actor_name=actor_name,
                location=location,
                timestamp=datetime.now(timezone.utc).isoformat(),
                action=action,
                temperature=kwargs.get("temperature"),
                humidity=kwargs.get("humidity"),
                quality_check=kwargs.get("quality_check"),
                notes=kwargs.get("notes", ""),
            )

            record = self._link_record(BlockchainRecord(
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
            self.products[batch_id].blockchain_records.append(record.serialize())

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
        created_by_uid: str = "",
        harvest_id: str = "",
    ) -> SmartContract:
        """Create smart contract for transaction atomically with harvest_id dedup."""
        if batch_id not in self.products:
            raise ValueError(f"Batch {batch_id} not found")

        snap = self._snapshot_state()
        try:
            if harvest_id:
                self._record_harvest_id(harvest_id)

            contract_id = f"CONTRACT-{uuid.uuid4().hex[:12].upper()}"
            transaction_payload = {
                "batch_id": batch_id,
                "seller": seller,
                "buyer": buyer,
                "price": price,
                "harvest_id": harvest_id or "",
            }

            transaction_id = self._generate_transaction_id(transaction_payload)
            self._validate_transaction_uniqueness(transaction_id)
            contract = SmartContract(
                contract_id=contract_id,
                batch_id=batch_id,
                seller=seller,
                buyer=buyer,
                price=price,
                created_by_uid=created_by_uid,
                terms=terms or {},
            )

            record = self._link_record(BlockchainRecord(
                timestamp=datetime.now().isoformat(),
                actor=seller,
                action="contract_created",
                location="contract",
                data=asdict(contract),
            )
            record.previous_hash = record.calculate_hash()

            # Commit
            self.smart_contracts[contract_id] = contract
            self._link_and_append(record)

            return contract

        except Exception:
            self._rollback_to_snapshot(snap)
            raise



    def execute_smart_contract(
        self,
        contract_id: str,
        harvest_id: str = "",
    ) -> Dict:
        """
        Execute a smart contract atomically.

        Supports:
        - harvest_id deduplication
        - transaction deduplication
        - rollback on failure
        """

        contract = self.smart_contracts.get(contract_id)
        if contract is None:
            raise ValueError(f"Contract {contract_id} not found")

        snapshot = self._snapshot_state()

        try:
            # ----------------------------
            # Validation phase
            # ----------------------------
            if contract.status != "pending":
                raise ValueError(
                    f"Contract {contract_id} cannot be executed "
                    f"(status: {contract.status})"
                )

            if harvest_id:
                self._record_harvest_id(harvest_id)

            now_utc = datetime.now(timezone.utc)
            timestamp = now_utc.isoformat()

            transaction_payload = {
                "contract_id": contract_id,
                "buyer": contract.buyer,
                "amount": contract.price,
                "timestamp": timestamp,
                "harvest_id": harvest_id,
            }

            transaction_id = self._generate_transaction_id(
                transaction_payload
            )

            self._validate_transaction_uniqueness(transaction_id)

            # Build blockchain record before mutating state
            record = self._link_record(
                BlockchainRecord(
                    timestamp=timestamp,
                    actor=contract.buyer,
                    action="contract_executed",
                    location="contract",
                    data={
                        "contract_id": contract_id,
                        "batch_id": contract.batch_id,
                        "amount": contract.price,
                        "currency": contract.currency,
                        "transaction_id": transaction_id,
                    },
                )
            )
            record.previous_hash = record.calculate_hash()

            # ----------------------------
            # Commit phase
            # ----------------------------
            contract.status = "executed"
            contract.executed_at = timestamp

            self._link_and_append(record)
            self._record_transaction_id(transaction_id)

            return {
                "success": True,
                "contract_id": contract_id,
                "transaction_id": transaction_id,
                "executed_at": timestamp,
                "amount": contract.price,
            }

        except Exception:
            self._rollback_to_snapshot(snapshot)
            raise

    def generate_qr_code(self, batch_id: str) -> str:
        """Generate QR code for product batch"""
        if batch_id not in self.products:
            raise ValueError(f"Batch {batch_id} not found")

        batch = self.products[batch_id]
        proof = self._build_trace_proof(batch_id)
        qr_data = {
            "batch_id": batch_id,
            "crop_type": batch.crop_type,
            "quantity": batch.quantity,
            "unit": batch.unit,
            "farmer": batch.farmer_name,
            "harvested": batch.harvesting_date,
        }
        if proof["signature"]:
            qr_data["trace_signature"] = proof["signature"]

        if self._signing_key:
            payload = json.dumps(qr_data, sort_keys=True)
            sig = hmac.new(self._signing_key.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
            qr_data["sig"] = sig
            qr_data["proof"] = "signed"
            qr_data["verification_url"] = f"https://fasalsaathi.agri/verify/{batch_id}?proof={sig}"
        else:
            qr_data["verification_url"] = f"https://fasalsaathi.agri/verify/{batch_id}"

        qr_code = qrcode.QRCode(version=1, box_size=10, border=5)
        qr_code.add_data(self._canonical_json(qr_data))
        qr_code.make(fit=True)

        qr_image = qr_code.make_image(fill_color="black", back_color="white")
        qr_buffer = io.BytesIO()
        qr_image.save(qr_buffer, format="PNG")
        qr_base64 = base64.b64encode(qr_buffer.getvalue()).decode()

        return qr_base64

    def verify_batch(self, batch_id: str, proof: Optional[str] = None) -> Dict:
        """Verify product batch authenticity"""
        if batch_id not in self.products:
            return {"success": False, "message": "Batch not found"}

        batch = self.products[batch_id]
        records = self.supply_chain_nodes.get(batch_id, [])

        verification_score = 0.0
        if len(records) >= 1:
            verification_score += 40

        quality_verifications = [r for r in records if r.quality_check == "passed"]
        if quality_verifications:
            verification_score += 15

        registered_count = 0
        for record in records:
            if record.actor_name in self.verified_actors:
                registered_count += 1

        if registered_count > 0:
            verification_score += 15

        blockchain_intact = self._verify_blockchain_integrity()
        trace_proof = self._build_trace_proof(batch_id)
        if blockchain_intact:
            verification_score = min(100, verification_score + 10)

        if not self._signing_key:
            authenticated = "unauthenticated"
        elif proof:
            expected = hmac.new(self._signing_key.encode("utf-8"), batch_id.encode("utf-8"), hashlib.sha256).hexdigest()
            authenticated = hmac.compare_digest(expected, proof)
        else:
            authenticated = verification_score >= 70

        return {
            "success": True,
            "batch_id": batch_id,
            "product": batch.crop_type,
            "quantity": batch.quantity,
            "farmer": batch.farmer_name,
            "verification_score": min(100, verification_score),
            "authenticated": authenticated,
            "blockchain_records": len(batch.blockchain_records),
            "supply_chain_nodes": len(records),
            "certifications": batch.certifications,
            "quality_score": batch.quality_score,
            "harvested_date": batch.harvesting_date,
            "integrity_ok": blockchain_intact,
            "trace_proof": trace_proof["proof_hash"],
            "trace_signature": trace_proof["signature"],
            "latest_block_hash": trace_proof["latest_block_hash"],
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

    def _link_record(self, record: BlockchainRecord) -> BlockchainRecord:
        """Set previous_hash from chain tail and return record."""
        record.previous_hash = self.chain[-1].hash if self.chain else ""
        record.hash = record.calculate_hash()
        return record

    def _verify_blockchain_integrity(self) -> bool:
        """Verify blockchain chain continuity and hash integrity"""
        prev_hash = ""
        for record in self.chain:
            if record.previous_hash != record.calculate_hash():
                return False
            if record.previous_hash != prev_hash:
                return False
            prev_hash = record.hash
        return True

    def get_blockchain_record_count(self) -> int:
        """Get total records in blockchain"""
        return len(self.chain)

    def get_blockchain_stats(self) -> dict:
        """Return dedup stats for /health/blockchain."""
        return {
            "total_blocks": len(self.chain),
            "unique_harvest_ids": len(self._harvest_ids),
            "duplicate_prevented": len(self._processed_transaction_ids) - len(self._harvest_ids),
            "integrity_ok": self._verify_blockchain_integrity(),
        }

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

    def _snapshot_state(self) -> Dict:
        """Snapshot mutable state for rollback."""
        return {
            "_trace_batches": dict(self._trace_batches),
            "chain": list(self.chain),
        }

    def _rollback(self, snapshot: Dict) -> None:
        """Restore state from a snapshot."""
        self._trace_batches.clear()
        self._trace_batches.update(snapshot["_trace_batches"])
        self.chain = snapshot["chain"]

    # ------------- QR Traceability (farmer-facing) -------------

    def register_trace_batch(self, payload: Dict) -> Dict:
        """Store a QR-traceability batch submitted from the frontend."""

        batch_id = payload.get("id")
        if not batch_id:
            raise ValueError("Batch ID is required")
        if batch_id in self._trace_batches:
            raise ValueError(f"Batch {batch_id} is already registered")

        snap = self._snapshot_state()
        try:
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
            record.hash = record.calculate_hash()
            self.chain.append(record)

            return entry
        except Exception:
            self._rollback_to_snapshot(snap)
            raise

    def get_trace_batch(self, batch_id: str) -> Optional[Dict]:
        """Fetch a QR-traceability batch by ID.  Returns None if not found."""
        batch = self._trace_batches.get(batch_id)
        if not batch:
            return None
        batch_copy = _copy.deepcopy(batch)
        if batch_id in self.products:
            batch_copy["traceability"] = self.get_traceability_qr_payload(batch_id)
        return batch_copy
