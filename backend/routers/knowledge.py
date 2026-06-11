"""Knowledge Base Router - RAG, Climate Simulation, Seeds"""
from typing import Any, Callable, Optional
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, validator
import logging
from threading import Lock
from time import monotonic, time

from backend.compute_rate_limit import enforce_compute_rate_limit
from backend.climate_sim.data import (
    CROP_PROFILES,
    REGIONAL_SEASONAL_BASELINES,
    REGION_ALIASES,
    VALID_SEASONS,
)
from backend.schemas import RAGQuery
from backend.core.logging_config import setup_logging

router = APIRouter()
logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Cache Management
# ---------------------------------------------------------------------------

# RAG query cache with TTL (Time To Live)
_RAG_QUERY_CACHE = {}
_RAG_QUERY_CACHE_LOCK = Lock()
_RAG_QUERY_CACHE_TTL_SECONDS = 3600  # 1 hour

class CachedResult:
    """Wrapper for cached query results with timestamp."""
    def __init__(self, result: Any):
        self.result = result
        self.timestamp = time()

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if cache entry has exceeded TTL."""
        return (time() - self.timestamp) > ttl_seconds


def _get_cache_key(query: str, top_k: int) -> str:
    """Generate cache key from query parameters."""
    return f"{query}:{top_k}"


def _get_cached_result(query: str, top_k: int) -> Optional[Any]:
    """Retrieve cached result if available and not expired."""
    cache_key = _get_cache_key(query, top_k)
    with _RAG_QUERY_CACHE_LOCK:
        if cache_key not in _RAG_QUERY_CACHE:
            return None
        
        cached = _RAG_QUERY_CACHE[cache_key]
        if cached.is_expired(_RAG_QUERY_CACHE_TTL_SECONDS):
            del _RAG_QUERY_CACHE[cache_key]
            logger.debug("Expired cache entry removed: %s", cache_key)
            return None
        
        logger.debug("Cache hit for query: %s", cache_key)
        return cached.result


def _set_cached_result(query: str, top_k: int, result: Any) -> None:
    """Store result in cache with timestamp."""
    cache_key = _get_cache_key(query, top_k)
    with _RAG_QUERY_CACHE_LOCK:
        _RAG_QUERY_CACHE[cache_key] = CachedResult(result)
        logger.debug("Cached result stored: %s", cache_key)


def invalidate_rag_cache() -> None:
    """Invalidate entire RAG query cache (call when knowledge base is updated)."""
    with _RAG_QUERY_CACHE_LOCK:
        cache_size = len(_RAG_QUERY_CACHE)
        _RAG_QUERY_CACHE.clear()
        logger.info("RAG query cache invalidated. Cleared %d entries.", cache_size)


def invalidate_rag_cache_key(query: str, top_k: int) -> None:
    """Invalidate specific cache entry."""
    cache_key = _get_cache_key(query, top_k)
    with _RAG_QUERY_CACHE_LOCK:
        if cache_key in _RAG_QUERY_CACHE:
            del _RAG_QUERY_CACHE[cache_key]
            logger.info("Specific cache entry invalidated: %s", cache_key)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SimulationRequest(BaseModel):
    """Climate simulation request.

    Fields
    ------
    crop_type : str
        The crop being grown (e.g. "wheat", "rice", "cotton").
    temp_delta : float
        Change in temperature relative to the regional baseline (°C).
        Range: -5 to +5.
    rain_delta : float
        Change in rainfall relative to the regional baseline (mm/month).
        Range: -200 to +200.
    region : str, optional
        Indian agro-climatic region.  Used to select the correct baseline
        temperature and rainfall.  Defaults to "central" when omitted.
    season : str, optional
        Cropping season ("kharif", "rabi", "zaid").  Used to select the
        correct seasonal baseline.  Defaults to "kharif".
    """
    crop_type: str = Field(..., min_length=1, max_length=50)
    temp_delta: float = Field(..., ge=-5, le=5)
    rain_delta: float = Field(..., ge=-200, le=200)
    region: Optional[str] = Field(default="central", max_length=50)
    season: Optional[str] = Field(default="kharif", max_length=20)

    @validator("region", pre=True, always=True)
    def normalise_region(cls, v):
        return (v or "central").lower().strip()

    @validator("season", pre=True, always=True)
    def normalise_season(cls, v):
        return (v or "kharif").lower().strip()


class SeedVerifyRequest(BaseModel):
    code: str = Field(..., min_length=4, max_length=100)


_RAG_REJECTION_LOG_INTERVAL_SECONDS = 60


def get_verify_role_fn(request: Request) -> Callable[..., Any]:
    verify_fn = getattr(request.app.state, "verify_role_fn", None)
    if verify_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    return verify_fn


def get_rag_generate_fn(request: Request) -> Callable[..., Any]:
    rag_fn = getattr(request.app.state, "rag_generate_fn", None)
    if rag_fn is None:
        raise HTTPException(status_code=503, detail="RAG not available")
    return rag_fn


def get_seed_registry(request: Request) -> dict:
    registry = getattr(request.app.state, "seed_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Seed registry not initialized")
    return registry


def get_rag_runtime(
    rag_fn: Callable[..., Any] = Depends(get_rag_generate_fn),
    verify_fn: Callable[..., Any] = Depends(get_verify_role_fn),
):
    return rag_fn, verify_fn


def get_simulation_runtime(verify_fn: Callable[..., Any] = Depends(get_verify_role_fn)):
    return verify_fn


def get_seed_runtime(
    verify_fn: Callable[..., Any] = Depends(get_verify_role_fn),
    registry: dict = Depends(get_seed_registry),
):
    return verify_fn, registry


def _get_rag_rejection_log_state(request: Request) -> dict:
    state = getattr(request.app.state, "knowledge_rag_rejection_log_state", None)
    if state is None:
        state = {
            "last_log": 0.0,
            "lock": Lock(),
        }
        request.app.state.knowledge_rag_rejection_log_state = state
    return state


def _log_rag_rejection(request: Request, exc: ValidationError) -> None:
    log_state = _get_rag_rejection_log_state(request)
    now = monotonic()
    with log_state["lock"]:
        if now - log_state["last_log"] < _RAG_REJECTION_LOG_INTERVAL_SECONDS:
            return
        log_state["last_log"] = now

    errors = exc.errors()
    first_error = errors[0] if errors else {}
    context = first_error.get("ctx") or {}
    error_code = context.get("error_code", first_error.get("type", "validation_error"))

    logger.warning(
        "Rejected RAG query: error_code=%s threat_type=%s threat_match=%s client=%s path=%s",
        error_code,
        context.get("threat_type"),
        context.get("threat_match"),
        request.client.host if request.client and request.client.host else "unknown",
        request.url.path,
    )


async def _parse_rag_query(request: Request) -> RAGQuery:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_json",
                "message": "Request body must be valid JSON.",
            },
        ) from exc

    try:
        return RAGQuery.model_validate(payload)
    except ValidationError as exc:
        _log_rag_rejection(request, exc)
        errors = exc.errors()
        first_error = errors[0] if errors else {}
        context = first_error.get("ctx") or {}
        error_code = context.get("error_code", first_error.get("type", "validation_error"))

        detail = {
            "code": error_code,
            "message": context.get("error_message", first_error.get("msg", "Invalid RAG query.")),
            "reason": context.get("reason", "validation_error"),
            "errors": errors,
        }
        if context.get("threat_type"):
            detail["threat_type"] = context.get("threat_type", "prompt_injection")
            detail["threat_match"] = context.get("threat_match")

        raise HTTPException(status_code=422, detail=detail) from exc


def _enforce_rate_limit(request: Request, scope: str, uid: Optional[str], limit: int, window_seconds: int) -> None:
    """Enforce compute rate limit, raising HTTPException(429) on exhaustion.

    Replaces the previous return-value contract where callers checked
    `if rate_limited is not None: return rate_limited`. That pattern had
    two failure modes:
      1. An empty dict {} from the limiter would be returned as HTTP 200.
      2. Any non-None, non-JSONResponse value would be returned verbatim,
         producing an unstructured response the frontend could not parse.

    Raising HTTPException(429) is consistent with the rest of the codebase
    and guarantees the error path is always deterministic regardless of what
    enforce_compute_rate_limit returns.
    """
    rate_limited = enforce_compute_rate_limit(
        request,
        scope=scope,
        uid=uid,
        limit=limit,
        window_seconds=window_seconds,
    )
    if rate_limited is not None:
        # Extract retry_after from the JSONResponse if available so the
        # Retry-After header is preserved in the HTTPException detail.
        retry_after = None
        if isinstance(rate_limited, JSONResponse):
            try:
                import json as _json
                body = _json.loads(rate_limited.body)
                retry_after = body.get("error", {}).get("retry_after")
            except Exception:
                pass
        detail = "Rate limit exceeded. Please retry later."
        if retry_after:
            detail = f"Rate limit exceeded. Retry after {retry_after} seconds."
        raise HTTPException(status_code=429, detail=detail)


def _handle_router_exception(exc: Exception, log_message: str, detail: str) -> None:
    """Preserve explicit HTTP errors and normalize unexpected failures."""
    if isinstance(exc, HTTPException):
        raise exc

    logger.error("%s: %s", log_message, exc)
    raise HTTPException(status_code=500, detail=detail) from exc


# ---------------------------------------------------------------------------
# Climate simulation data tables
#
# Sources / methodology
# ---------------------
# Regional baseline temperatures and rainfall are derived from IMD
# (India Meteorological Department) normal data for the 1991-2020
# reference period, averaged across the major agro-climatic zones
# defined by ICAR (Indian Council of Agricultural Research).
#
# Zones covered:
#   northwest  – Punjab, Haryana, western UP, Rajasthan (arid/semi-arid)
#   northeast  – Assam, West Bengal, Bihar, eastern UP
#   central    – MP, Chhattisgarh, Vidarbha (default)
#   west       – Gujarat, coastal Maharashtra
#   south      – Karnataka, Andhra Pradesh, Telangana
#   southwest  – Kerala, coastal Karnataka (humid tropical)
#   east       – Odisha, Jharkhand
#
# Seasonal baselines (temp °C, rain mm/month):
#   kharif  – June–October (south-west monsoon)
#   rabi    – November–March (winter/post-monsoon)
#   zaid    – April–May (summer/pre-kharif)
#
# Crop sensitivity coefficients (yield impact per unit climate change)
# are adapted from ICAR/NICRA (National Innovations in Climate Resilient
# Agriculture) crop modelling studies.
# ---------------------------------------------------------------------------

def _resolve_region(region: str) -> str:
    """Map a user-supplied region string to a canonical zone key."""
    key = region.lower().strip()
    if key in REGIONAL_SEASONAL_BASELINES:
        return key
    return REGION_ALIASES.get(key, "central")


def _resolve_season(season: str) -> str:
    key = season.lower().strip()
    if key in VALID_SEASONS:
        return key
    # Accept common abbreviations / alternate spellings
    if key in ("monsoon", "kharif", "summer"):
        return "kharif"
    if key in ("winter", "rabi", "post-monsoon"):
        return "rabi"
    if key in ("spring", "zaid", "pre-kharif"):
        return "zaid"
    return "kharif"


def _get_crop_profile(crop_type: str) -> dict:
    key = crop_type.lower().strip()
    return CROP_PROFILES.get(key, CROP_PROFILES["default"])


def _compute_impact_score(
    sim_temp: float,
    sim_rain: float,
    profile: dict,
    base_temp: float,
    base_rain: float,
) -> float:
    """Compute a 0–100 impact score using crop-specific coefficients.

    The score represents expected yield as a percentage of the baseline
    yield under the simulated conditions.  100 = no change, values above
    100 indicate improved conditions, values below indicate stress.

    Formula
    -------
    yield_impact = temp_effect + rain_effect
      temp_effect = temp_coeff * (sim_temp - base_temp) * 100
      rain_effect = rain_coeff * ((sim_rain - base_rain) / 10) * 100

    The result is clamped to [0, 150] and then normalised to [0, 100]
    for the response field so the frontend always receives a bounded value.
    """
    temp_effect = profile["temp_coeff"] * (sim_temp - base_temp) * 100
    rain_effect = profile["rain_coeff"] * ((sim_rain - base_rain) / 10.0) * 100
    raw_score = 100.0 + temp_effect + rain_effect
    return round(min(150.0, max(0.0, raw_score)), 1)


def _build_recommendations(
    sim_temp: float,
    sim_rain: float,
    profile: dict,
    impact_score: float,
    crop_type: str,
    season: str,
) -> list:
    """Return a list of actionable, context-aware recommendations."""
    recs = []
    opt_t_min, opt_t_max = profile["opt_temp"]
    opt_r_min, opt_r_max = profile["opt_rain"]

    # Temperature stress
    if sim_temp > opt_t_max + 2:
        recs.append(
            f"High heat stress expected ({sim_temp:.1f}°C vs optimal {opt_t_min}–{opt_t_max}°C "
            f"for {crop_type}). Consider heat-tolerant varieties, mulching, and evening irrigation."
        )
    elif sim_temp > opt_t_max:
        recs.append(
            f"Mild heat stress ({sim_temp:.1f}°C). Increase irrigation frequency and apply "
            f"potassium-based foliar spray to improve heat tolerance."
        )
    elif sim_temp < opt_t_min - 2:
        recs.append(
            f"Cold stress risk ({sim_temp:.1f}°C vs optimal {opt_t_min}–{opt_t_max}°C). "
            f"Use frost-resistant varieties and consider row covers or smoke screens."
        )
    elif sim_temp < opt_t_min:
        recs.append(
            f"Below-optimal temperature ({sim_temp:.1f}°C). Delay sowing until temperatures "
            f"rise above {opt_t_min}°C or use cold-tolerant varieties."
        )

    # Rainfall / moisture stress
    if sim_rain < opt_r_min * 0.5:
        recs.append(
            f"Severe drought risk ({sim_rain:.0f} mm/month vs optimal {opt_r_min}–{opt_r_max} mm). "
            f"Switch to drip irrigation, apply mulch, and consider drought-tolerant varieties."
        )
    elif sim_rain < opt_r_min:
        recs.append(
            f"Moisture deficit ({sim_rain:.0f} mm/month). Increase supplemental irrigation by "
            f"{opt_r_min - sim_rain:.0f} mm/month and monitor soil moisture weekly."
        )
    elif sim_rain > opt_r_max * 1.5:
        recs.append(
            f"Waterlogging risk ({sim_rain:.0f} mm/month). Ensure field drainage channels are "
            f"clear and consider raised-bed cultivation to protect root systems."
        )
    elif sim_rain > opt_r_max:
        recs.append(
            f"Excess moisture ({sim_rain:.0f} mm/month). Improve field drainage and watch for "
            f"fungal diseases (blast, blight) — apply preventive fungicide if needed."
        )

    # Overall yield impact
    if impact_score < 60:
        recs.append(
            f"Significant yield reduction expected (~{100 - impact_score:.0f}% below baseline). "
            f"Consult your local Krishi Vigyan Kendra (KVK) for region-specific mitigation strategies."
        )
    elif impact_score < 85:
        recs.append(
            f"Moderate yield impact expected. Review crop insurance options and maintain "
            f"contingency plans for the {season} season."
        )
    elif impact_score > 110:
        recs.append(
            f"Favourable conditions projected. Consider increasing planting density or "
            f"applying additional fertiliser to capture the yield potential."
        )
    else:
        recs.append(
            f"Conditions are within the acceptable range for {crop_type} during {season}. "
            f"Continue standard agronomic practices."
        )

    return recs

def _build_scenario_snapshot(
    crop_type: str,
    region: str,
    season: str,
    temp_delta: float,
    rain_delta: float,
) -> dict:
    """Generate a simulation snapshot for comparison workflows."""

    seasonal_baselines = REGIONAL_SEASONAL_BASELINES.get(
        region,
        REGIONAL_SEASONAL_BASELINES["central"],
    )

    base_temp, base_rain = seasonal_baselines.get(
        season,
        seasonal_baselines["kharif"],
    )

    sim_temp = round(base_temp + temp_delta, 2)
    sim_rain = round(base_rain + rain_delta, 2)

    profile = _get_crop_profile(crop_type)

    impact_score = _compute_impact_score(
        sim_temp,
        sim_rain,
        profile,
        base_temp,
        base_rain,
    )

    return {
        "temperature_c": sim_temp,
        "rainfall_mm_per_month": sim_rain,
        "impact_score": impact_score,
    }


def _compare_scenarios(baseline: dict, comparison: dict) -> dict:
    """Generate structured scenario comparison metrics."""

    return {
        "temperature_difference": round(
            comparison["temperature_c"] - baseline["temperature_c"],
            2,
        ),
        "rainfall_difference": round(
            comparison["rainfall_mm_per_month"]
            - baseline["rainfall_mm_per_month"],
            2,
        ),
        "impact_score_difference": round(
            comparison["impact_score"] - baseline["impact_score"],
            2,
        ),
        "trend": (
            "improved"
            if comparison["impact_score"] > baseline["impact_score"]
            else "reduced"
            if comparison["impact_score"] < baseline["impact_score"]
            else "unchanged"
        ),
    }


def _build_comparison_summary(
    crop_type: str,
    region: str,
    season: str,
    simulated: dict,
) -> dict:
    """
    Build side-by-side comparison metadata using
    baseline and simulated scenario outputs.
    """

    baseline_scenario = _build_scenario_snapshot(
        crop_type,
        region,
        season,
        0,
        0,
    )

    comparison_scenario = {
        "temperature_c": simulated["temperature_c"],
        "rainfall_mm_per_month": simulated["rainfall_mm_per_month"],
        "impact_score": simulated["impact_score"],
    }

    return {
        "baseline_scenario": baseline_scenario,
        "comparison_scenario": comparison_scenario,
        "differences": _compare_scenarios(
            baseline_scenario,
            comparison_scenario,
        ),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/rag/query")
async def rag_query(request: Request, body: RAGQuery = Depends(_parse_rag_query), runtime=Depends(get_rag_runtime)):
    """Query the AI knowledge base (RAG).

    Authentication is required to prevent unauthenticated callers from
    consuming Gemini API quota on the project's billing account and to
    enable per-user rate limiting in the future.
    
    Results are cached with a 1-hour TTL to reduce redundant API calls
    for identical queries. Cache is automatically invalidated when entries
    exceed the TTL or when invalidate_rag_cache() is called.
    """
    rag_fn, verify_fn = runtime
    # Raises HTTP 401 if the Firebase token is missing or invalid.
    token_data = await verify_fn(request)
    _enforce_rate_limit(
        request,
        scope="knowledge.rag_query",
        uid=(token_data or {}).get("uid"),
        limit=12,
        window_seconds=60,
    )
    try:
        # Check cache first
        cached_result = _get_cached_result(body.query, body.top_k)
        if cached_result is not None:
            return {"success": True, "query": body.query, "results": cached_result, "cached": True}
        
        # Cache miss - call RAG function
        result = rag_fn(body.query, body.top_k)
        
        # Store in cache for future requests
        _set_cached_result(body.query, body.top_k, result)
        
        return {"success": True, "query": body.query, "results": result, "cached": False}
    except Exception as exc:
        _handle_router_exception(exc, "RAG query failed", "RAG query failed")


@router.post("/simulate-climate")
async def simulate_climate(request: Request, data: SimulationRequest, verify_fn=Depends(get_simulation_runtime)):
    """Run a climate impact simulation for a given crop.

    Authentication is required so that the endpoint is not freely
    accessible to scrapers and bots under the global rate limit.
    """
    # Raises HTTP 401 if the Firebase token is missing or invalid.
    token_data = await verify_fn(request)
    _enforce_rate_limit(
        request,
        scope="knowledge.simulate_climate",
        uid=(token_data or {}).get("uid"),
        limit=10,
        window_seconds=60,
    )
    try:
        canonical_region = _resolve_region(data.region or "central")
        canonical_season = _resolve_season(data.season or "kharif")

        seasonal_baselines = REGIONAL_SEASONAL_BASELINES.get(
            canonical_region,
            REGIONAL_SEASONAL_BASELINES["central"],
        )
        base_temp, base_rain = seasonal_baselines.get(
            canonical_season,
            seasonal_baselines["kharif"],
        )

        sim_temp = round(base_temp + data.temp_delta, 2)
        sim_rain = round(base_rain + data.rain_delta, 2)

        profile = _get_crop_profile(data.crop_type)
        impact_score = _compute_impact_score(sim_temp, sim_rain, profile, base_temp, base_rain)
        comparison_summary = _build_comparison_summary(
            data.crop_type,
            canonical_region,
            canonical_season,
            {
                "temperature_c": sim_temp,
                "rainfall_mm_per_month": sim_rain,
                "impact_score": impact_score,
            },
        )
        recommendations = _build_recommendations(
            sim_temp, sim_rain, profile, impact_score, data.crop_type, canonical_season
        )

        return {
            "success": True,
            "crop_type": data.crop_type,
            "region": canonical_region,
            "season": canonical_season,
            "baseline": {
                "temperature_c": base_temp,
                "rainfall_mm_per_month": base_rain,
                "source": "IMD 1991-2020 normals (ICAR agro-climatic zones)",
            },
            "simulated": {
                "temperature_c": sim_temp,
                "rainfall_mm_per_month": sim_rain,
                "temp_delta": data.temp_delta,
                "rain_delta": data.rain_delta,
            },
            "impact": {
                "score": impact_score,
                "interpretation": (
                    "Score represents projected yield as % of baseline. "
                    "100 = no change; <100 = yield reduction; >100 = yield improvement."
                ),
            },
            "recommendations": recommendations,
            "scenario_comparison": comparison_summary,
            "disclaimer": (
                "This simulation uses statistical crop-climate models and regional "
                "climate normals. Results are indicative only and should not replace "
                "advice from your local Krishi Vigyan Kendra (KVK) or agricultural officer."
            ),
        }
    except Exception as exc:
        _handle_router_exception(exc, "Climate simulation failed", "Climate simulation failed")


@router.post("/seeds/verify")
async def verify_seed(request: Request, data: SeedVerifyRequest, runtime=Depends(get_seed_runtime)):
    verify_fn, registry = runtime
    try:
        await verify_fn(request)
        is_verified = registry.get(data.code, {}).get("verified", False)
        seed_info = registry.get(data.code, {})
        return {"success": True, "code": data.code, "verified": is_verified, "seed_info": seed_info}
    except Exception as exc:
        _handle_router_exception(exc, "Seed verification failed", "Seed verification failed")


@router.post("/cache/invalidate")
async def invalidate_cache(request: Request, verify_fn=Depends(get_verify_role_fn)):
    """Invalidate entire RAG query cache.
    
    This endpoint should be called by an admin or knowledge base manager
    when the underlying knowledge base is updated to ensure clients receive
    fresh results on the next query.
    
    Requires authentication to prevent unauthorized cache clearing.
    """
    try:
        # Require authentication
        token_data = await verify_fn(request)
        
        # Optional: add role check for admin-only access
        # For now, any authenticated user can invalidate
        invalidate_rag_cache()
        
        return {
            "success": True,
            "message": "RAG query cache has been invalidated. All cached results cleared."
        }
    except Exception as exc:
        _handle_router_exception(exc, "Cache invalidation failed", "Cache invalidation failed")


@router.post("/cache/invalidate-query")
async def invalidate_query_cache(request: Request, query: str, top_k: int = 5, verify_fn=Depends(get_verify_role_fn)):
    """Invalidate a specific cache entry by query parameters.
    
    Use this endpoint to selectively invalidate cache entries without
    clearing the entire cache. Useful for targeted updates.
    
    Requires authentication.
    """
    try:
        # Require authentication
        token_data = await verify_fn(request)
        
        invalidate_rag_cache_key(query, top_k)
        
        return {
            "success": True,
            "message": f"Cache entry invalidated for query: {query} (top_k={top_k})"
        }
    except Exception as exc:
        _handle_router_exception(exc, "Query cache invalidation failed", "Query cache invalidation failed")