"""Blockchain Supply Chain Router"""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class RegisterActorRequest(BaseModel):
    actor_id: str = Field(..., min_length=1, max_length=50)
    name: str = Field(..., min_length=1, max_length=100)
    actor_type: str = Field(..., min_length=1, max_length=50)
    location: str = Field(..., min_length=1, max_length=100)

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

def init_blockchain(scb):
    global supply_chain_blockchain
    supply_chain_blockchain = scb

@router.post("/register-actor")
async def register_actor(request: Request, data: RegisterActorRequest):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        actor = supply_chain_blockchain.register_actor(data.model_dump())
        return {"success": True, "actor": actor}
    except Exception as e:
        logger.error(f"Actor registration error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/create-batch")
async def create_batch(request: Request, data: CreateProductBatchRequest):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        batch = supply_chain_blockchain.create_product_batch(data.model_dump())
        return {"success": True, "batch": batch}
    except Exception as e:
        logger.error(f"Batch error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/add-node")
async def add_node(request: Request, batch_id: str, node_type: str, actor_name: str, location: str, action: str):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        node = supply_chain_blockchain.add_supply_chain_node(batch_id, node_type, actor_name, location, action)
        return {"success": True, "node": node}
    except Exception as e:
        logger.error(f"Node error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/create-contract")
async def create_contract(request: Request, data: CreateSmartContractRequest):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        contract = supply_chain_blockchain.create_smart_contract(data.model_dump())
        return {"success": True, "contract": contract}
    except Exception as e:
        logger.error(f"Contract error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/execute-contract")
async def execute_contract(request: Request, contract_id: str):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        result = supply_chain_blockchain.execute_smart_contract(contract_id)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/qr-code/{batch_id}")
async def get_qr_code(batch_id: str):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        qr_code = supply_chain_blockchain.generate_qr_code(batch_id)
        return {"success": True, "batch_id": batch_id, "qr_code_base64": qr_code}
    except Exception as e:
        logger.error(f"QR error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/verify/{batch_id}")
async def verify_batch(batch_id: str):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        verification = supply_chain_blockchain.verify_batch(batch_id)
        return {"success": True, "verification": verification}
    except Exception as e:
        logger.error(f"Verify error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/journey/{batch_id}")
async def get_journey(batch_id: str):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        journey = supply_chain_blockchain.get_supply_chain_journey(batch_id)
        return {"success": True, "data": journey}
    except Exception as e:
        logger.error(f"Journey error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/analytics/{batch_id}")
async def get_analytics(batch_id: str):
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        analytics = supply_chain_blockchain.get_supply_chain_analytics(batch_id)
        return {"success": True, "data": analytics}
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/marketplace")
async def get_marketplace():
    if supply_chain_blockchain is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        marketplace = supply_chain_blockchain.get_certified_products()
        return {"success": True, "marketplace": marketplace}
    except Exception as e:
        logger.error(f"Marketplace error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

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
        raise HTTPException(status_code=400, detail=str(e))
