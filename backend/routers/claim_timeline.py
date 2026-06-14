from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

# Mock data for now – later replace with DB queries
mock_claims = {
    1: {
        "claim_id": 1,
        "damage_event": "2026-06-10T08:00:00",
        "claim_filed": "2026-06-11T09:30:00",
        "survey": "2026-06-12T14:00:00",
        "verification": None,
        "approval": None,
        "payout": None,
        "current_stage": "Survey"
    }
}

@router.get("/api/claims/{claim_id}/timeline")
async def get_claim_timeline(claim_id: int):
    claim = mock_claims.get(claim_id)
    if not claim:
        return {"error": "Claim not found"}
    return claim
