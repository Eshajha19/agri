"""Knowledge Base Router - RAG, Climate Simulation, Seeds"""
import re
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field, validator
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class RAGQuery(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)

    @validator("query")
    def sanitize_and_normalize_query(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Query must be a non-empty string.")

        # Strip script tags entirely to prevent XSS / script injection
        v = re.sub(r'<script.*?>.*?</script>', '', v, flags=re.IGNORECASE | re.DOTALL)
        v = re.sub(r'</?script.*?>', '', v, flags=re.IGNORECASE)

        # Strip event handler attributes (onclick=, onload=, etc.)
        v = re.sub(r'on\w+\s*=', '', v, flags=re.IGNORECASE)

        # Strip dangerous URI schemes
        v = re.sub(r'javascript:', '', v, flags=re.IGNORECASE)
        v = re.sub(r'data:', '', v, flags=re.IGNORECASE)
        v = re.sub(r'vbscript:', '', v, flags=re.IGNORECASE)

        # Strip all remaining HTML tags to prevent HTML injection in prompts or UI
        v = re.sub(r'<[^>]*>', '', v)

        # Neutralize markdown links [text](url) -> text (url)
        v = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', v)

        # Neutralize markdown styling syntax (bold, italic, code, headers)
        v = re.sub(r'[*_~`#]', '', v)

        # Normalize whitespace
        v = v.strip()
        v = re.sub(r'\s+', ' ', v)

        # Prompt injection mitigation: reject known instruction-override phrases
        forbidden_patterns = [
            r"ignore\s+(?:all\s+)?previous\s+instructions",
            r"ignore\s+(?:the\s+)?system\s+prompt",
            r"override\s+system\s+constraints",
            r"developer\s+mode",
            r"bypass\s+safety\s+filter",
            r"disregard\s+(?:all\s+)?prior\s+instructions",
            r"act\s+as\s+(?:a\s+)?(?:different|unrestricted|unfiltered)\s+(?:ai|model|assistant)",
            r"pretend\s+(?:you\s+are|to\s+be)\s+(?:a\s+)?(?:different|unrestricted)",
            r"jailbreak",
            r"prompt\s+injection",
        ]
        v_lower = v.lower()
        for pattern in forbidden_patterns:
            if re.search(pattern, v_lower):
                raise ValueError("Query contains disallowed phrases or prompt injection attempts.")

        # Re-enforce minimum length after sanitization has stripped content
        if len(v) < 3:
            raise ValueError("Query must be at least 3 characters long after sanitization.")

        return v

class SimulationRequest(BaseModel):
    crop_type: str
    temp_delta: float = Field(..., ge=-5, le=5)
    rain_delta: float = Field(..., ge=-100, le=100)

class SeedVerifyRequest(BaseModel):
    code: str = Field(..., min_length=4, max_length=100)

rag_generate_fn = None
rbac_manager = None
Permission = None
seed_registry = None
verify_role_fn = None

def init_knowledge(rg_fn, rbac, perm, sr, vr_fn):
    global rag_generate_fn, rbac_manager, Permission, seed_registry, verify_role_fn
    rag_generate_fn = rg_fn
    rbac_manager = rbac
    Permission = perm
    seed_registry = sr
    verify_role_fn = vr_fn

@router.post("/rag/query")
async def rag_query(request: Request, body: RAGQuery):
    if rag_generate_fn is None:
        raise HTTPException(status_code=503, detail="RAG not available")
    try:
        result = rag_generate_fn(body.query, body.top_k)
        return {"success": True, "query": body.query, "results": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/simulate-climate")
async def simulate_climate(request: Request, data: SimulationRequest):
    try:
        new_temp = 28 + data.temp_delta
        new_rain = 100 + data.rain_delta
        impact_score = min(100, max(0, 50 + (data.temp_delta * 5) - (data.rain_delta * 0.1)))
        return {
            "success": True,
            "crop_type": data.crop_type,
            "simulated_temperature": new_temp,
            "simulated_rainfall": new_rain,
            "impact_score": impact_score,
            "recommendation": "Adjust irrigation" if impact_score < 40 else "Conditions favorable"
        }
    except Exception as e:
        logger.error(f"Climate error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/seeds/verify")
async def verify_seed(request: Request, data: SeedVerifyRequest):
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        await verify_role_fn(request)
        is_verified = seed_registry.get(data.code, {}).get("verified", False) if seed_registry else False
        seed_info = seed_registry.get(data.code, {}) if seed_registry else {}
        return {"success": True, "code": data.code, "verified": is_verified, "seed_info": seed_info}
    except Exception as e:
        logger.error(f"Seed error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
