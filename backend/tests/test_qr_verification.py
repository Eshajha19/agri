def test_token_verification_success():
    proof = blockchain._build_trace_proof(batch_id)
    token = blockchain._create_verification_token(batch_id, proof)
    assert blockchain.verify_token(token)

def test_token_expiry():
    proof = blockchain._build_trace_proof(batch_id)
    token = blockchain._create_verification_token(batch_id, proof)
    blockchain._verification_tokens[token]["expires_at"] = time.time() - 1
    assert not blockchain.verify_token(token)
