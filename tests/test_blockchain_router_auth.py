import asyncio

import pytest

from backend.routers import blockchain as blockchain_router
from blockchain_supply_chain import SupplyChainBlockchain


def setup_function():
    blockchain_router.supply_chain_blockchain = None
    blockchain_router.verify_role_fn = None


def _init_router(blockchain, token_data):
    async def verify(_request, required_roles=None):
        if required_roles:
            role = token_data.get("role", "")
            if role not in required_roles and role not in {"admin", "expert"}:
                raise blockchain_router.HTTPException(status_code=403, detail="Forbidden")
        return token_data

    blockchain_router.init_blockchain(blockchain, verify)


def test_create_batch_binds_owner_uid():
    blockchain = SupplyChainBlockchain()
    _init_router(blockchain, {"uid": "farmer-1", "role": "farmer"})

    result = asyncio.run(
        blockchain_router.create_batch(
            request=object(),
            data=blockchain_router.CreateProductBatchRequest(
                crop_type="tomato",
                farm_id="FARM001",
                quantity=100.0,
                unit="kg",
                planting_date="2026-01-15",
                harvesting_date="2026-04-20",
                farmer_name="Raj Kumar",
            ),
        )
    )

    batch = result["batch"]
    assert batch["owner_uid"] == "farmer-1"


def test_add_node_rejects_non_owner():
    blockchain = SupplyChainBlockchain()
    batch = blockchain.create_product_batch(
        "tomato",
        "FARM001",
        100.0,
        "kg",
        "2026-01-15",
        "2026-04-20",
        "Raj Kumar",
        owner_uid="farmer-1",
    )
    _init_router(blockchain, {"uid": "farmer-2", "role": "farmer"})

    async def call():
        return await blockchain_router.add_node(
            request=object(),
            batch_id=batch.batch_id,
            node_type="warehouse",
            actor_name="Manager",
            location="Pune",
            action="stored",
        )

    with pytest.raises(blockchain_router.HTTPException) as exc:
        asyncio.run(call())
    assert exc.value.status_code == 403


def test_create_contract_requires_owner_and_persists_creator_uid():
    blockchain = SupplyChainBlockchain()
    batch = blockchain.create_product_batch(
        "tomato",
        "FARM001",
        100.0,
        "kg",
        "2026-01-15",
        "2026-04-20",
        "Raj Kumar",
        owner_uid="farmer-1",
    )
    _init_router(blockchain, {"uid": "farmer-1", "role": "farmer"})

    result = asyncio.run(
        blockchain_router.create_contract(
            request=object(),
            data=blockchain_router.CreateSmartContractRequest(
                batch_id=batch.batch_id,
                seller="Raj Kumar",
                buyer="Distributor Co",
                price=5000.0,
            ),
        )
    )

    contract = result["contract"]
    assert contract.created_by_uid == "farmer-1"


def test_execute_contract_rejects_non_creator():
    blockchain = SupplyChainBlockchain()
    batch = blockchain.create_product_batch(
        "tomato",
        "FARM001",
        100.0,
        "kg",
        "2026-01-15",
        "2026-04-20",
        "Raj Kumar",
        owner_uid="farmer-1",
    )
    contract = blockchain.create_smart_contract(
        batch.batch_id,
        "Raj Kumar",
        "Distributor Co",
        5000.0,
        created_by_uid="farmer-1",
    )
    _init_router(blockchain, {"uid": "farmer-2", "role": "farmer"})

    async def call():
        return await blockchain_router.execute_contract(request=object(), contract_id=contract.contract_id)

    with pytest.raises(blockchain_router.HTTPException) as exc:
        asyncio.run(call())
    assert exc.value.status_code == 403