from backend.blockchain_supply_chain import SupplyChainBlockchain

def test_idempotent_batch_creation():
    bc = SupplyChainBlockchain()
    key = "abc123"
    batch1 = bc.create_product_batch("wheat","farm1",100,"kg","2024-01-01","2024-02-01","Alice",idempotency_key=key)
    batch2 = bc.create_product_batch("wheat","farm1",100,"kg","2024-01-01","2024-02-01","Alice",idempotency_key=key)
    assert batch1.batch_id == batch2.batch_id  # same result

def test_idempotent_contract_creation():
    bc = SupplyChainBlockchain()
    batch = bc.create_product_batch("rice","farm2",50,"kg","2024-01-01","2024-02-01","Bob")
    key = "xyz789"
    contract1 = bc.create_smart_contract(batch.batch_id,"Bob","Charlie",100.0,idempotency_key=key)
    contract2 = bc.create_smart_contract(batch.batch_id,"Bob","Charlie",100.0,idempotency_key=key)
    assert contract1.contract_id == contract2.contract_id
