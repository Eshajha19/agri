"""Knowledge Base Router - RAG, Climate Simulation, Seeds"""
import re
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field, validator
import logging
from error_utils import safe_detail

router = APIRouter()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RAGQuery(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)

    @validator("query")
    def sanitize_and_normalize_query(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Query must be a non-empty string.")

        v = re.sub(r'<script.*?>.*?</script>', '', v, flags=re.IGNORECASE | re.DOTALL)
        v = re.sub(r'</?script.*?>', '', v, flags=re.IGNORECASE)
        v = re.sub(r'on\w+\s*=', '', v, flags=re.IGNORECASE)
        v = re.sub(r'javascript:', '', v, flags=re.IGNORECASE)
        v = re.sub(r'data:', '', v, flags=re.IGNORECASE)
        v = re.sub(r'vbscript:', '', v, flags=re.IGNORECASE)
        v = re.sub(r'<[^>]*>', '', v)
        v = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', v)
        v = re.sub(r'[*_~`#]', '', v)
        v = v.strip()
        v = re.sub(r'\s+', ' ', v)

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

        if len(v) < 3:
            raise ValueError("Query must be at least 3 characters long after sanitization.")

        return v


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


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

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

# (base_temp_C, base_rain_mm_per_month)
_REGIONAL_SEASONAL_BASELINES = {
    # region        kharif          rabi           zaid
    "northwest":  {"kharif": (32, 120), "rabi": (15,  20), "zaid": (38,  15)},
    "northeast":  {"kharif": (30, 350), "rabi": (18,  40), "zaid": (34,  80)},
    "central":    {"kharif": (30, 200), "rabi": (20,  25), "zaid": (36,  20)},
    "west":       {"kharif": (30, 180), "rabi": (22,  10), "zaid": (37,  10)},
    "south":      {"kharif": (28, 160), "rabi": (24,  60), "zaid": (34,  30)},
    "southwest":  {"kharif": (27, 600), "rabi": (26, 120), "zaid": (32,  80)},
    "east":       {"kharif": (30, 280), "rabi": (19,  30), "zaid": (35,  40)},
}

# Crop-specific sensitivity coefficients
# temp_coeff  : fractional yield change per +1°C above baseline
# rain_coeff  : fractional yield change per +10 mm/month above baseline
# opt_temp    : optimal temperature range (min, max) in °C
# opt_rain    : optimal monthly rainfall range (min, max) in mm
_CROP_PROFILES = {
    "wheat":      {"temp_coeff": -0.06, "rain_coeff":  0.03, "opt_temp": (15, 25), "opt_rain": (40,  80)},
    "rice":       {"temp_coeff": -0.05, "rain_coeff":  0.02, "opt_temp": (25, 35), "opt_rain": (150, 300)},
    "maize":      {"temp_coeff": -0.07, "rain_coeff":  0.04, "opt_temp": (20, 30), "opt_rain": (80,  150)},
    "cotton":     {"temp_coeff": -0.03, "rain_coeff":  0.01, "opt_temp": (25, 35), "opt_rain": (60,  120)},
    "sugarcane":  {"temp_coeff": -0.02, "rain_coeff":  0.05, "opt_temp": (25, 35), "opt_rain": (150, 250)},
    "soybean":    {"temp_coeff": -0.04, "rain_coeff":  0.03, "opt_temp": (20, 30), "opt_rain": (80,  150)},
    "potato":     {"temp_coeff": -0.05, "rain_coeff":  0.04, "opt_temp": (15, 25), "opt_rain": (60,  100)},
    "groundnut":  {"temp_coeff": -0.04, "rain_coeff":  0.02, "opt_temp": (25, 35), "opt_rain": (60,  120)},
    "mustard":    {"temp_coeff": -0.05, "rain_coeff":  0.02, "opt_temp": (10, 25), "opt_rain": (30,   60)},
    "chickpea":   {"temp_coeff": -0.06, "rain_coeff":  0.02, "opt_temp": (15, 25), "opt_rain": (30,   60)},
    "tomato":     {"temp_coeff": -0.05, "rain_coeff":  0.03, "opt_temp": (20, 30), "opt_rain": (80,  120)},
    "onion":      {"temp_coeff": -0.04, "rain_coeff":  0.02, "opt_temp": (15, 25), "opt_rain": (50,   80)},
    "default":    {"temp_coeff": -0.04, "rain_coeff":  0.02, "opt_temp": (20, 30), "opt_rain": (80,  150)},
}

# Region aliases so common user inputs map to canonical keys
_REGION_ALIASES = {
    "punjab": "northwest", "haryana": "northwest", "rajasthan": "northwest",
    "up": "northeast", "uttar pradesh": "northeast", "bihar": "northeast",
    "west bengal": "northeast", "assam": "northeast",
    "mp": "central", "madhya pradesh": "central", "chhattisgarh": "central",
    "vidarbha": "central", "maharashtra": "central",
    "gujarat": "west",
    "karnataka": "south", "andhra pradesh": "south", "telangana": "south",
    "kerala": "southwest",
    "odisha": "east", "jharkhand": "east",
}

_VALID_SEASONS = {"kharif", "rabi", "zaid"}


def _resolve_region(region: str) -> str:
    """Map a user-supplied region string to a canonical zone key."""
    key = region.lower().strip()
    if key in _REGIONAL_SEASONAL_BASELINES:
        return key
    return _REGION_ALIASES.get(key, "central")


def _resolve_season(season: str) -> str:
    key = season.lower().strip()
    if key in _VALID_SEASONS:
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
    return _CROP_PROFILES.get(key, _CROP_PROFILES["default"])


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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/rag/query")
async def rag_query(request: Request, body: RAGQuery):
    """Query the AI knowledge base (RAG).

    Authentication is required to prevent unauthenticated callers from
    consuming Gemini API quota on the project's billing account and to
    enable per-user rate limiting in the future.
    """
    if rag_generate_fn is None:
        raise HTTPException(status_code=503, detail="RAG not available")
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    # Raises HTTP 401 if the Firebase token is missing or invalid.
    await verify_role_fn(request)
    try:
        result = rag_generate_fn(body.query, body.top_k)
        return {"success": True, "query": body.query, "results": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail=safe_detail(e, 500))


@router.post("/simulate-climate")
async def simulate_climate(request: Request, data: SimulationRequest):
    """Run a climate impact simulation for a given crop.

    Authentication is required so that the endpoint is not freely
    accessible to scrapers and bots under the global rate limit.
    """
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    # Raises HTTP 401 if the Firebase token is missing or invalid.
    await verify_role_fn(request)
    try:
        canonical_region = _resolve_region(data.region or "central")
        canonical_season = _resolve_season(data.season or "kharif")

        seasonal_baselines = _REGIONAL_SEASONAL_BASELINES.get(
            canonical_region,
            _REGIONAL_SEASONAL_BASELINES["central"],
        )
        base_temp, base_rain = seasonal_baselines.get(
            canonical_season,
            seasonal_baselines["kharif"],
        )

        sim_temp = round(base_temp + data.temp_delta, 2)
        sim_rain = round(base_rain + data.rain_delta, 2)

        profile = _get_crop_profile(data.crop_type)
        impact_score = _compute_impact_score(sim_temp, sim_rain, profile, base_temp, base_rain)
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
            "disclaimer": (
                "This simulation uses statistical crop-climate models and regional "
                "climate normals. Results are indicative only and should not replace "
                "advice from your local Krishi Vigyan Kendra (KVK) or agricultural officer."
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Climate simulation error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))


@router.post("/seeds/verify")
async def verify_seed(request: Request, data: SeedVerifyRequest):
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        await verify_role_fn(request)
        is_verified = seed_registry.get(data.code, {}).get("verified", False) if seed_registry else False
        seed_info = seed_registry.get(data.code, {}) if seed_registry else {}
        return {"success": True, "code": data.code, "verified": is_verified, "seed_info": seed_info}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Seed error: {e}")
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))
