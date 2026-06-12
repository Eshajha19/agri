"""Blockchain Supply Chain Router"""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
from error_utils import safe_detail

router = APIRouter()
logger = logging.getLogger(__name__)

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

class CreateSmartContractRequest(BaseModel):
    batch_id: str = Field(..., min_length=1)
    seller: str = Field(..., min_length=1, max_length=100)
    buyer: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    terms: Optional[Dict] = None

supply_chain_blockchain = None
verify_role_fn = None

def init_blockchain(scb, vr_fn=None):
    global supply_chain_blockchain, verify_role_fn
    supply_chain_blockchain = scb
    verify_role_fn = vr_fn

@router.post("/register-actor")
async def register_actor(request: Request, data: RegisterActorRequest):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        actor = supply_chain_blockchain.register_actor(
            data.actor_id, data.name, data.actor_type, data.location
        )
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
    uid = token_data["uid"]

    try:
        batch_payload = data.model_dump()
        batch_payload["registeredByUid"] = uid
        batch_payload["status"] = "Pending Verification"
        result = supply_chain_blockchain.register_trace_batch(batch_payload)
        return {"success": True, "batch": result}
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
        batch = supply_chain_blockchain.get_trace_batch(batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="Batch not found")
        return {"success": True, "batch": batch}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trace batch fetch error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.post("/create-batch")
async def create_batch(request: Request, data: CreateProductBatchRequest):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        batch = supply_chain_blockchain.create_product_batch(
            data.crop_type, data.farm_id, data.quantity, data.unit,
            data.planting_date, data.harvesting_date, data.farmer_name,
        )
        return {"success": True, "batch": asdict(batch) if hasattr(batch, '__dataclass_fields__') else batch}
    except Exception as e:
        logger.error(f"Batch error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.post("/add-node")
async def add_node(request: Request, batch_id: str, node_type: str, actor_name: str, location: str, action: str):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        node = supply_chain_blockchain.add_supply_chain_node(batch_id, node_type, actor_name, location, action)
        return {"success": True, "node": node}
    except Exception as e:
        logger.error(f"Node error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.post("/create-contract")
async def create_contract(request: Request, data: CreateSmartContractRequest):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        contract = supply_chain_blockchain.create_smart_contract(
            data.batch_id, data.seller, data.buyer, data.price, data.terms
        )
        return {"success": True, "contract": contract}
    except Exception as e:
        logger.error(f"Contract error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.post("/execute-contract")
async def execute_contract(request: Request, contract_id: str):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        result = supply_chain_blockchain.execute_smart_contract(contract_id)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Execution error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.get("/qr-code/{batch_id}")
async def get_qr_code(batch_id: str):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        qr_code = supply_chain_blockchain.generate_qr_code(batch_id)
        return {"success": True, "batch_id": batch_id, "qr_code_base64": qr_code}
    except Exception as e:
        logger.error(f"QR error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.get("/verify/{batch_id}")
async def verify_batch(batch_id: str):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        verification = supply_chain_blockchain.verify_batch(batch_id)
        return {"success": True, "verification": verification}
    except Exception as e:
        logger.error(f"Verify error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.get("/journey/{batch_id}")
async def get_journey(batch_id: str):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        journey = supply_chain_blockchain.get_supply_chain_journey(batch_id)
        return {"success": True, "data": journey}
    except Exception as e:
        logger.error(f"Journey error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.get("/analytics/{batch_id}")
async def get_analytics(batch_id: str):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        analytics = supply_chain_blockchain.get_supply_chain_analytics(batch_id)
        return {"success": True, "data": analytics}
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.get("/marketplace")
async def get_marketplace():
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        marketplace = supply_chain_blockchain.get_certified_products()
        return {"success": True, "marketplace": marketplace}
    except Exception as e:
        logger.error(f"Marketplace error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.get("/stats")
async def get_stats():
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        stats = {
            "total_records": supply_chain_blockchain.get_blockchain_record_count(),
            "actors": len(supply_chain_blockchain.verified_actors if hasattr(supply_chain_blockchain, 'verified_actors') else []),
            "contracts": len(supply_chain_blockchain.smart_contracts if hasattr(supply_chain_blockchain, 'smart_contracts') else [])
        }
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))
