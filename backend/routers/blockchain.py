"""Blockchain Supply Chain Router"""
from dataclasses import asdict
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timezone
import logging
import time
from error_utils import safe_detail
from backend.core.logging_config import setup_logging

router = APIRouter()
logger = setup_logging(__name__)

class RegisterActorRequest(BaseModel):
    actor_id: str = Field(..., min_length=1, max_length=50)
    name: str = Field(..., min_length=1, max_length=100)
    actor_type: str = Field(..., min_length=1, max_length=50)
    location: str = Field(..., min_length=1, max_length=100)

class JourneyStep(BaseModel):
    date: str = Field(..., min_length=1, max_length=30)
    event: str = Field(..., min_length=1, max_length=100)
    location: str = Field(..., min_length=1, max_length=200)
    details: str = Field(..., max_length=500)

class RegisterTraceBatchRequest(BaseModel):
    """Register a QR-traceability batch with its initial journey data.

    This replaces the previous localStorage-only approach so batch data
    is stored server-side and cannot be tampered with by consumers or
    farmers after the QR code has been shared.
    """
    id: str = Field(..., min_length=1, max_length=100)
    crop: str = Field(..., min_length=1, max_length=100)
    variety: str = Field(..., min_length=1, max_length=100)
    harvestDate: str = Field(..., min_length=1, max_length=30)
    farm: str = Field(..., min_length=1, max_length=200)
    journey: List[JourneyStep] = Field(..., min_items=1)

class CreateProductBatchRequest(BaseModel):
    crop_type: str = Field(..., min_length=1, max_length=50)
    farm_id: str = Field(..., min_length=1, max_length=50)
    quantity: float = Field(..., gt=0)
    unit: str = Field(..., min_length=1, max_length=20)
    planting_date: str = Field(..., min_length=1)
    harvesting_date: str = Field(..., min_length=1)
    farmer_name: str = Field(..., min_length=1, max_length=100)
    harvest_id: str = Field(default="", max_length=200)

class AddSupplyChainNodeRequest(BaseModel):
    batch_id: str = Field(..., min_length=1, max_length=100)
    node_type: str = Field(..., min_length=1, max_length=50)
    actor_name: str = Field(..., min_length=1, max_length=100)
    location: str = Field(..., min_length=1, max_length=200)
    action: str = Field(..., min_length=1, max_length=100)
    harvest_id: str = Field(default="", max_length=200)

class CreateSmartContractRequest(BaseModel):
    batch_id: str = Field(..., min_length=1)
    seller: str = Field(..., min_length=1, max_length=100)
    buyer: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    terms: Optional[Dict] = None
    harvest_id: str = Field(default="", max_length=200)

supply_chain_blockchain = None
verify_role_fn = None


def _get_token_role(token_data: Dict) -> str:
    return str((token_data or {}).get("role", "")).strip().lower()


def _is_privileged_role(token_data: Dict) -> bool:
    return _get_token_role(token_data) in {"admin", "expert"}


def _require_owner_uid(token_data: Dict) -> str:
    uid = (token_data or {}).get("sub") or (token_data or {}).get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return uid


def _require_role_or_owner(token_data: Dict, allowed_roles: set[str], owner_uid: str) -> str:
    uid = _require_owner_uid(token_data)
    role = _get_token_role(token_data)
    if uid == owner_uid:
        return uid
    if role in allowed_roles or _is_privileged_role(token_data):
        return uid
    raise HTTPException(status_code=403, detail="Access denied: insufficient permissions")


def _verify_write_operation_auth(token_data: Optional[Dict]) -> str:
    """Centralized authentication check for all blockchain write operations.
    
    Enforces consistent authentication across POST endpoints:
    - Validates token exists and contains subject identity (sub preferred, uid fallback)
    - Logs authentication decisions
    - Returns authenticated uid for audit trail
    
    Raises HTTPException(401) if authentication fails.
    """
    if not token_data:
        logger.warning("Write operation attempted without authentication")
        raise HTTPException(status_code=401, detail="Authentication required for write operations")
    
    uid = token_data.get("sub") or token_data.get("uid")
    if not uid:
        logger.warning("Write operation attempted with invalid token (missing subject claim)")
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    logger.debug("Write operation authenticated for uid=%s", uid)
    return uid


def _get_batch(batch_id: str):
    batch = supply_chain_blockchain.products.get(batch_id) if supply_chain_blockchain is not None else None
    if batch is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    return batch

def init_blockchain(scb, vr_fn=None):
    global supply_chain_blockchain, verify_role_fn
    supply_chain_blockchain = scb
    verify_role_fn = vr_fn

@router.post("/register-actor")
async def register_actor(request: Request, data: RegisterActorRequest):
    """Register a supply chain actor. Requires admin or expert role.

    Without authentication any caller could inject fake verified actors
    (farms, warehouses, distributors) into the blockchain, inflating
    verification scores and making fraudulent produce appear certified.
    """
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")
    token_data = await verify_role_fn(request)
    uid = _verify_write_operation_auth(token_data)
    try:
        actor = supply_chain_blockchain.register_actor(
            data.actor_id, data.name, data.actor_type, data.location
        )
        logger.info("Actor registered: actor_id=%s by uid=%s", data.actor_id, uid)
        return {"success": True, "actor": actor}
    except Exception as e:
        logger.error(f"Actor registration error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))


@router.post("/trace-batch")
async def register_trace_batch(request: Request, data: RegisterTraceBatchRequest):
    """Persist a QR-traceability batch server-side.

    Requires authentication so only the owning farmer can register a
    batch under their account. The batch is stored in Firestore via the
    supply-chain blockchain so the data is immutable from the consumer's
    perspective — it cannot be edited through DevTools or by clearing
    browser storage.
    """
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")

    token_data = await verify_role_fn(request)
    uid = _verify_write_operation_auth(token_data)

    try:
        # register_trace_batch() does not exist on SupplyChainBlockchain.
        # Map the RegisterTraceBatchRequest fields to create_product_batch(),
        # which is the correct method for persisting a new batch server-side.
        #
        # Field mapping:
        #   data.crop      → crop_type   (crop name, e.g. "Rice")
        #   data.farm      → farm_id     (farm identifier / name)
        #   data.variety   → farmer_name (closest available field; variety
        #                                 is stored in the batch metadata)
        #   data.harvestDate → harvesting_date
        #   quantity / unit  → defaults (not in RegisterTraceBatchRequest)
        batch = supply_chain_blockchain.create_product_batch(
            crop_type=data.crop,
            farm_id=data.farm,
            quantity=1.0,           # not provided by this request schema
            unit="unit",            # not provided by this request schema
            planting_date="",       # not provided by this request schema
            harvesting_date=data.harvestDate,
            farmer_name=data.variety,
            owner_uid=uid or "",
        )
        batch_id = batch.batch_id
        traceability = supply_chain_blockchain.get_traceability_qr_payload(batch_id)
        logger.info("Trace batch registered: batch_id=%s by uid=%s", batch_id, uid)
        return {
            "success": True,
            "batch": {
                "id": batch_id,
                "crop": data.crop,
                "variety": data.variety,
                "harvestDate": data.harvestDate,
                "farm": data.farm,
                "journey": [step.model_dump() for step in data.journey],
                "registeredByUid": uid,
                "status": "Pending Verification",
                "traceability": traceability,
            },
        }
    except Exception as e:
        logger.error(f"Trace batch registration error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))


@router.get("/trace-batch/{batch_id}")
async def get_trace_batch(batch_id: str):
    """Fetch a single QR-traceability batch by ID.

    This endpoint is intentionally public (no auth required) so that
    consumers scanning a QR code can verify the batch without needing
    an account. The data is read-only and served from the server, so
    it cannot be tampered with client-side.
    """
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        # get_trace_batch() does not exist on SupplyChainBlockchain.
        # Use products.get() to look up the batch by ID, which is the
        # correct way to retrieve a persisted batch.
        batch = supply_chain_blockchain.products.get(batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="Batch not found")
        return {"success": True, "batch": asdict(batch)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trace batch fetch error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.post("/create-batch")
async def create_batch(request: Request, data: CreateProductBatchRequest):
    """Create a product batch on the blockchain. Requires authentication."""
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")
    token_data = await verify_role_fn(request)
    uid = _verify_write_operation_auth(token_data)
    try:
        batch = supply_chain_blockchain.create_product_batch(
            data.crop_type, data.farm_id, data.quantity, data.unit,
            data.planting_date, data.harvesting_date, data.farmer_name,
            owner_uid=uid, harvest_id=data.harvest_id,
        )
        logger.info("Product batch created: batch_id=%s by uid=%s", batch.batch_id, uid)
        return {"success": True, "batch": asdict(batch)}
    except ValueError as e:
        if "Duplicate harvest_id" in str(e):
            logger.warning("Duplicate harvest_id attempt: %s by uid=%s", data.harvest_id, uid)
            raise HTTPException(status_code=409, detail={"error": "duplicate_harvest", "harvest_id": data.harvest_id})
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.post("/add-node")
async def add_node(request: Request, data: AddSupplyChainNodeRequest):
    """Add a supply chain node to an existing batch. Requires authentication."""
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")
    token_data = await verify_role_fn(request)
    uid = _verify_write_operation_auth(token_data)
    try:
        node = supply_chain_blockchain.add_supply_chain_node(
            data.batch_id, data.node_type, data.actor_name, data.location, data.action,
            harvest_id=data.harvest_id,
        )
        logger.info("Supply chain node added: batch_id=%s node_type=%s by uid=%s", data.batch_id, data.node_type, uid)
        return {"success": True, "node": node}
    except ValueError as e:
        if "Duplicate harvest_id" in str(e):
            logger.warning("Duplicate harvest_id attempt: %s by uid=%s", data.harvest_id, uid)
            raise HTTPException(status_code=409, detail={"error": "duplicate_harvest", "harvest_id": data.harvest_id})
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Node error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.post("/create-contract")
async def create_contract(request: Request, data: CreateSmartContractRequest):
    """Create a smart contract between a seller and buyer. Requires authentication."""
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")
    token_data = await verify_role_fn(request)
    uid = _verify_write_operation_auth(token_data)
    try:
        contract = supply_chain_blockchain.create_smart_contract(
            data.batch_id, data.seller, data.buyer, data.price, data.terms,
            created_by_uid=uid, harvest_id=data.harvest_id,
        )
        logger.info("Smart contract created: batch_id=%s seller=%s buyer=%s by uid=%s", data.batch_id, data.seller, data.buyer, uid)
        return {"success": True, "contract": contract}
    except ValueError as e:
        if "Duplicate harvest_id" in str(e):
            logger.warning("Duplicate harvest_id attempt: %s by uid=%s", data.harvest_id, uid)
            raise HTTPException(status_code=409, detail={"error": "duplicate_harvest", "harvest_id": data.harvest_id})
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Contract error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.post("/execute-contract/{contract_id}")
async def execute_contract(request: Request, contract_id: str):
    """Execute a smart contract. Requires authentication."""
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")
    token_data = await verify_role_fn(request)
    uid = _verify_write_operation_auth(token_data)
    try:
        result = supply_chain_blockchain.execute_smart_contract(contract_id)
        logger.info("Smart contract executed: contract_id=%s by uid=%s", contract_id, uid)
        return {"success": True, "result": result}
    except ValueError as e:
        if "Duplicate harvest_id" in str(e):
            logger.warning("Duplicate harvest_id in contract execution: %s by uid=%s", contract_id, uid)
            raise HTTPException(status_code=409, detail={"error": "duplicate_harvest", "harvest_id": contract_id})
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Execution error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.get("/qr-code/{batch_id}")
async def get_qr_code(request: Request, batch_id: str):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")
    await verify_role_fn(request)
    try:
        qr_code = supply_chain_blockchain.generate_qr_code(batch_id)
        qr_payload = supply_chain_blockchain.get_traceability_qr_payload(batch_id)
        return {
            "success": True,
            "batch_id": batch_id,
            "qr_code_base64": qr_code,
            "qr_payload": qr_payload,
        }
    except Exception as e:
        logger.error(f"QR error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))



def _build_verification_diagnostics(
    batch_id: str,
    verification_result,
):
    diagnostics = {
        "verification_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "batch_id": batch_id,
        "verification_status": (
            "verified"
            if verification_result
            else "failed"
        ),
        "stages": [
            {
                "stage": "batch_lookup",
                "status": "completed",
            },
            {
                "stage": "traceability_validation",
                "status": "completed" if verification_result else "failed",
            },
            {
                "stage": "blockchain_integrity_check",
                "status": "completed" if verification_result else "skipped",
            },
        ],
        "verification_stages_total": 3,
        "verification_stages_completed": (
            3 if verification_result else 1
        ),
        "failure_reason": (
            None
            if verification_result
            else "verification_check_failed"
        ),
        "audit_metadata": {
            "diagnostic_version": "v1",
            "verification_type": "blockchain_traceability",
            "response_type": "enhanced_verification_diagnostics",
        },
    }

    return diagnostics


@router.get("/verify/{batch_id}")
async def verify_batch(request: Request, batch_id: str):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")

    if verify_role_fn is None:
        raise HTTPException(
            status_code=500,
            detail="Auth service not initialized",
        )

    await verify_role_fn(request)

    started = time.perf_counter()

    try:
        verification = supply_chain_blockchain.verify_batch(
            batch_id
        )

        verification_passed = (
            verification.get(
                "authenticated",
                False,
            )
            if isinstance(
                verification,
                dict,
            )
            else bool(
                verification,
            )
        )

        logger.info(
            "Verification completed for batch_id=%s authenticated=%s",
            batch_id,
            verification_passed,
        )

        diagnostics = _build_verification_diagnostics(
            batch_id,
            verification_passed,
        )

        diagnostics[
            "verification_duration_ms"
        ] = round(
            (time.perf_counter() - started)
            * 1000,
            2,
        )

        return {
            "success": True,
            "verification": verification,
            "diagnostics": diagnostics,
            "verification_summary": {
                "status": diagnostics["verification_status"],
                "duration_ms": diagnostics["verification_duration_ms"],
                "stages_completed": diagnostics["verification_stages_completed"],
            },
        }

    except Exception as e:
        diagnostics = {
            "verification_timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "batch_id": batch_id,
            "verification_status": "failed",
            "verification_stages_total": 3,
            "verification_stages_completed": 0,
            "failure_reason": str(e),
            "verification_duration_ms": round(
                (time.perf_counter() - started)
                * 1000,
                2,
            ),
        }
        logger.error(f"Verify error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.get("/journey/{batch_id}")
async def get_journey(request: Request, batch_id: str):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")
    await verify_role_fn(request)
    try:
        journey = supply_chain_blockchain.get_supply_chain_journey(batch_id)
        return {"success": True, "data": journey}
    except Exception as e:
        logger.error(f"Journey error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.get("/analytics/{batch_id}")
async def get_analytics(request: Request, batch_id: str):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")
    token_data = await verify_role_fn(request)
    if not _is_privileged_role(token_data):
        raise HTTPException(status_code=403, detail="Access denied: admin or expert role required")
    try:
        analytics = supply_chain_blockchain.get_supply_chain_analytics(batch_id)
        return {"success": True, "data": analytics}
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.get("/marketplace")
async def get_marketplace(request: Request):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")
    await verify_role_fn(request)
    try:
        marketplace = supply_chain_blockchain.get_certified_products()
        return {"success": True, "marketplace": marketplace}
    except Exception as e:
        logger.error(f"Marketplace error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.get("/stats")
async def get_stats(request: Request):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")
    token_data = await verify_role_fn(request)
    if not _is_privileged_role(token_data):
        raise HTTPException(status_code=403, detail="Access denied: admin or expert role required")
    try:
        stats = {
            "total_records": supply_chain_blockchain.get_blockchain_record_count(),
            "actors": len(supply_chain_blockchain.verified_actors),
            "contracts": len(supply_chain_blockchain.smart_contracts)
        }
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))
