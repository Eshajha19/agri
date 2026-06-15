"""
Crop Recommendation API with Explanation Layer
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple, Callable, Coroutine, Any 
import logging
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import time
import asyncio
import threading

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/crop", tags=["crop"])

# ── Auth injection ────────────────────────────────────────────────────────────
_verify_role_fn = None

def init_auth(verify_role_fn):
    global _verify_role_fn
    _verify_role_fn = verify_role_fn
    logger.info("Crop recommendation router: auth initialised")

async def _require_authenticated(request):
    if _verify_role_fn is None:
        raise HTTPException(status_code=503, detail="Authorization service not initialised")
    return await _verify_role_fn(request, required_roles=None)


# ── Validation Configuration ─────────────────────────────────────────────────────
@dataclass
class ValidationRules:
    """Validation rules for soil and climate parameters"""
    # Soil parameters
    SOIL_PH_MIN: float = 4.5
    SOIL_PH_MAX: float = 8.5
    NITROGEN_MIN: float = 0
    NITROGEN_MAX: float = 500
    PHOSPHORUS_MIN: float = 0
    PHOSPHORUS_MAX: float = 100
    POTASSIUM_MIN: float = 0
    POTASSIUM_MAX: float = 500
    MOISTURE_MIN: float = 0
    MOISTURE_MAX: float = 100

    # Climate parameters
    TEMP_MIN: float = -50
    TEMP_MAX: float = 50
    RAINFALL_MIN: float = 0
    RAINFALL_MAX: float = 10000
    HUMIDITY_MIN: float = 0
    HUMIDITY_MAX: float = 100
    ALTITUDE_MIN: float = 0
    ALTITUDE_MAX: float = 10000


class ValidationError:
    """Container for validation errors and warnings"""
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_error(self, field: str, message: str):
        self.errors.append(f"{field}: {message}")

    def add_warning(self, field: str, message: str):
        self.warnings.append(f"{field}: {message}")

    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> Dict:
        return {"errors": self.errors, "warnings": self.warnings}


def validate_soil_parameters(ph: float, nitrogen: float, phosphorus: float,
                            potassium: float, moisture: Optional[float] = None) -> ValidationError:
    """Validate soil parameters"""
    errors = ValidationError()
    rules = ValidationRules()

    # pH validation
    if ph < rules.SOIL_PH_MIN or ph > rules.SOIL_PH_MAX:
        errors.add_error("soil_ph", f"Must be between {rules.SOIL_PH_MIN} and {rules.SOIL_PH_MAX}")

    # Nitrogen validation
    if nitrogen < rules.NITROGEN_MIN or nitrogen > rules.NITROGEN_MAX:
        errors.add_error("nitrogen", f"Must be between {rules.NITROGEN_MIN} and {rules.NITROGEN_MAX} ppm")
    if nitrogen < 10:
        errors.add_warning("nitrogen", "Low nitrogen levels may limit crop growth")

    # Phosphorus validation
    if phosphorus < rules.PHOSPHORUS_MIN or phosphorus > rules.PHOSPHORUS_MAX:
        errors.add_error("phosphorus", f"Must be between {rules.PHOSPHORUS_MIN} and {rules.PHOSPHORUS_MAX} ppm")
    if phosphorus < 5:
        errors.add_warning("phosphorus", "Low phosphorus may affect root development")

    # Potassium validation
    if potassium < rules.POTASSIUM_MIN or potassium > rules.POTASSIUM_MAX:
        errors.add_error("potassium", f"Must be between {rules.POTASSIUM_MIN} and {rules.POTASSIUM_MAX} ppm")
    if potassium < 30:
        errors.add_warning("potassium", "Low potassium may reduce disease resistance")

    # Moisture validation (optional)
    if moisture is not None:
        if moisture < rules.MOISTURE_MIN or moisture > rules.MOISTURE_MAX:
            errors.add_error("moisture", f"Must be between {rules.MOISTURE_MIN} and {rules.MOISTURE_MAX} %")

    return errors


def validate_climate_parameters(temperature: Optional[float] = None,
                               rainfall: Optional[float] = None,
                               humidity: Optional[float] = None) -> ValidationError:
    """Validate climate parameters"""
    errors = ValidationError()
    rules = ValidationRules()

    if temperature is not None:
        if temperature < rules.TEMP_MIN or temperature > rules.TEMP_MAX:
            errors.add_error("temperature", f"Must be between {rules.TEMP_MIN} and {rules.TEMP_MAX}°C")

    if rainfall is not None:
        if rainfall < rules.RAINFALL_MIN or rainfall > rules.RAINFALL_MAX:
            errors.add_error("rainfall", f"Must be between {rules.RAINFALL_MIN} and {rules.RAINFALL_MAX}mm")

    if humidity is not None:
        if humidity < rules.HUMIDITY_MIN or humidity > rules.HUMIDITY_MAX:
            errors.add_error("humidity", f"Must be between {rules.HUMIDITY_MIN} and {rules.HUMIDITY_MAX}%")

    return errors


def calculate_confidence_score(data_quality: float, climate_match: float,
                              soil_match: float) -> float:
    """Calculate recommendation confidence score (0-100)

    Factors:
    - Data quality (40%): How complete the input data is
    - Climate match (35%): How well crop suits the climate
    - Soil match (25%): How well crop suits the soil
    """
    confidence = (data_quality * 0.4) + (climate_match * 0.35) + (soil_match * 0.25)
    return min(100, max(0, confidence))


class RecommendationCache:
    """TTL-bounded cache for recommendation results.

    Entries older than ttl_hours are treated as expired and evicted on
    the next access. A size cap (max_size) prevents unbounded growth when
    many unique parameter combinations are queried over time.
    """
    _MAX_SIZE = 1000

    def __init__(self, ttl_hours: int = 24):
        # Stores (result, inserted_at_seconds) tuples keyed by cache key
        self.cache: Dict[str, Tuple] = {}
        self.ttl_seconds = ttl_hours * 3600
        self._lock = threading.Lock()

    def _generate_key(self, ph: float, nitrogen: float, phosphorus: float,
                     potassium: float, season: str, area_size: Optional[float]) -> str:
        """Generate cache key from parameters"""
        key_data = f"{ph}:{nitrogen}:{phosphorus}:{potassium}:{season}:{area_size}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, ph: float, nitrogen: float, phosphorus: float,
            potassium: float, season: str, area_size: Optional[float] = None) -> Optional[Dict]:
        """Get cached recommendation, or None if absent or expired."""
        key = self._generate_key(ph, nitrogen, phosphorus, potassium, season, area_size)
        async with self._lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            result, inserted_at = entry
            if time.monotonic() - inserted_at > self.ttl_seconds:
                del self.cache[key]
                return None
            return result

    def set(self, ph: float, nitrogen: float, phosphorus: float,
            potassium: float, season: str, area_size: Optional[float], result: Dict):
        """Cache recommendation result with the current timestamp."""
        key = self._generate_key(ph, nitrogen, phosphorus, potassium, season, area_size)
        async with self._lock:
            if len(self.cache) >= self._MAX_SIZE:
                # Evict the oldest entry to maintain the size cap
                oldest_key = min(self.cache, key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            self.cache[key] = (result, time.monotonic())


recommendation_cache = RecommendationCache()



# ── Request Model ─────────────────────────────────────────────────────────────
class CropRecommendationRequest(BaseModel):
    soil_ph: float = Field(..., ge=4.5, le=8.5)
    nitrogen: float = Field(..., ge=0, le=500)
    phosphorus: float = Field(..., ge=0, le=100)
    potassium: float = Field(..., ge=0, le=500)
    location: str
    season: str = "kharif"
    area_size: Optional[float] = None


# ── Crop Knowledge Base ───────────────────────────────────────────────────────
CROP_DATABASE = {
    "Rice": {
        "seasons": ["kharif"],
        "ph_range": (5.5, 7.0),
        "nitrogen_min": 20,
        "phosphorus_min": 10,
        "potassium_min": 50,
        "fertilizer": "Apply Urea 100kg/ha, DAP 50kg/ha at transplanting. Top-dress with Urea at tillering stage.",
        "description": "Staple cereal crop ideal for waterlogged conditions"
    },
    "Wheat": {
        "seasons": ["rabi"],
        "ph_range": (6.0, 7.5),
        "nitrogen_min": 25,
        "phosphorus_min": 12,
        "potassium_min": 60,
        "fertilizer": "Apply NPK 120:60:40 kg/ha. Split nitrogen into 3 doses at sowing, tillering, and jointing.",
        "description": "Cool-season cereal with high nutrition demand"
    },
    "Maize": {
        "seasons": ["kharif", "rabi", "summer"],
        "ph_range": (5.8, 7.0),
        "nitrogen_min": 30,
        "phosphorus_min": 15,
        "potassium_min": 80,
        "fertilizer": "Apply NPK 150:75:50 kg/ha. Apply 1/3 N at sowing, 1/3 at knee-high, 1/3 at tasseling.",
        "description": "Versatile cereal crop suitable for multiple seasons"
    },
    "Cotton": {
        "seasons": ["kharif"],
        "ph_range": (6.0, 8.0),
        "nitrogen_min": 20,
        "phosphorus_min": 10,
        "potassium_min": 70,
        "fertilizer": "Apply NPK 100:50:50 kg/ha. Apply potassium at boll formation stage.",
        "description": "Cash crop requiring well-drained soil"
    },
    "Sugarcane": {
        "seasons": ["summer", "kharif"],
        "ph_range": (6.0, 7.5),
        "nitrogen_min": 35,
        "phosphorus_min": 15,
        "potassium_min": 100,
        "fertilizer": "Apply NPK 250:100:120 kg/ha split over 3 applications across the growing season.",
        "description": "Long-duration cash crop with high nutrient demand"
    },
    "Soybean": {
        "seasons": ["kharif"],
        "ph_range": (6.0, 7.0),
        "nitrogen_min": 10,
        "phosphorus_min": 15,
        "potassium_min": 40,
        "fertilizer": "Apply starter nitrogen 20kg/ha + Rhizobium inoculant. Add phosphorus 60kg/ha at sowing.",
        "description": "Protein-rich legume that fixes atmospheric nitrogen"
    },
    "Chickpea": {
        "seasons": ["rabi"],
        "ph_range": (6.0, 8.0),
        "nitrogen_min": 10,
        "phosphorus_min": 12,
        "potassium_min": 30,
        "fertilizer": "Apply starter NPK 20:60:20 kg/ha. Rhizobium inoculation recommended.",
        "description": "Drought-tolerant pulse crop for rabi season"
    },
    "Mustard": {
        "seasons": ["rabi"],
        "ph_range": (6.0, 7.5),
        "nitrogen_min": 20,
        "phosphorus_min": 10,
        "potassium_min": 40,
        "fertilizer": "Apply NPK 80:40:40 kg/ha. Apply sulphur 20kg/ha for better oil content.",
        "description": "Oilseed crop well-suited for cool dry conditions"
    },
    "Sunflower": {
        "seasons": ["summer", "kharif"],
        "ph_range": (6.0, 7.5),
        "nitrogen_min": 20,
        "phosphorus_min": 12,
        "potassium_min": 50,
        "fertilizer": "Apply NPK 80:60:60 kg/ha. Boron 1.5kg/ha improves seed setting.",
        "description": "Short-duration oilseed crop with good drought tolerance"
    },
    "Groundnut": {
        "seasons": ["kharif", "summer"],
        "ph_range": (5.5, 7.0),
        "nitrogen_min": 10,
        "phosphorus_min": 15,
        "potassium_min": 40,
        "fertilizer": "Apply NPK 25:50:75 kg/ha + Gypsum 200kg/ha at pegging stage.",
        "description": "Leguminous oilseed crop fixing atmospheric nitrogen"
    }
}


# ── Helper Functions ──────────────────────────────────────────────────────────

def analyze_soil(ph: float, nitrogen: float,
                 phosphorus: float, potassium: float) -> Dict:
    """Analyze soil parameters and return classification."""

    # pH classification
    if ph < 5.5:
        ph_level = "Strongly Acidic"
    elif ph < 6.5:
        ph_level = "Moderately Acidic"
    elif ph <= 7.5:
        ph_level = "Neutral"
    elif ph <= 8.5:
        ph_level = "Moderately Alkaline"
    else:
        ph_level = "Strongly Alkaline"

    # Nitrogen classification
    if nitrogen < 15:
        n_level = "Low"
    elif nitrogen < 40:
        n_level = "Medium"
    else:
        n_level = "High"

    # Phosphorus classification
    if phosphorus < 10:
        p_level = "Low"
    elif phosphorus < 25:
        p_level = "Medium"
    else:
        p_level = "High"

    # Potassium classification
    if potassium < 50:
        k_level = "Low"
    elif potassium < 150:
        k_level = "Medium"
    else:
        k_level = "High"

    return {
        "ph_value": round(ph, 1),
        "ph_level": ph_level,
        "nitrogen_value": round(nitrogen, 1),
        "nitrogen_level": n_level,
        "phosphorus_value": round(phosphorus, 1),
        "phosphorus_level": p_level,
        "potassium_value": round(potassium, 1),
        "potassium_level": k_level
    }


def build_explanation(crop_name: str, crop_data: Dict,
                      req: CropRecommendationRequest) -> List[str]:
    """Build human-readable reasons explaining why a crop is recommended."""
    reasons = []
    ph_min, ph_max = crop_data["ph_range"]

    # pH explanation
    if ph_min <= req.soil_ph <= ph_max:
        reasons.append(
            f"Soil pH {req.soil_ph} is within the ideal range "
            f"({ph_min}–{ph_max}) for {crop_name}"
        )
    elif req.soil_ph < ph_min:
        reasons.append(
            f"Soil pH {req.soil_ph} is slightly below ideal "
            f"({ph_min}–{ph_max}) — lime application recommended"
        )
    else:
        reasons.append(
            f"Soil pH {req.soil_ph} is slightly above ideal range "
            f"({ph_min}–{ph_max}) — sulphur amendment may help"
        )

    # Nitrogen explanation
    if req.nitrogen >= crop_data["nitrogen_min"]:
        reasons.append(
            f"Nitrogen level {req.nitrogen} ppm meets the minimum "
            f"requirement of {crop_data['nitrogen_min']} ppm"
        )
    else:
        reasons.append(
            f"Nitrogen level {req.nitrogen} ppm is below the ideal "
            f"{crop_data['nitrogen_min']} ppm — additional fertilization needed"
        )

    # Phosphorus explanation
    if req.phosphorus >= crop_data["phosphorus_min"]:
        reasons.append(
            f"Phosphorus {req.phosphorus} ppm supports strong root "
            f"development for {crop_name}"
        )
    else:
        reasons.append(
            f"Phosphorus {req.phosphorus} ppm is low — DAP application "
            f"recommended before sowing"
        )

    # Potassium explanation
    if req.potassium >= crop_data["potassium_min"]:
        reasons.append(
            f"Potassium {req.potassium} ppm provides adequate disease "
            f"resistance and yield support"
        )
    else:
        reasons.append(
            f"Potassium {req.potassium} ppm is below optimum — "
            f"MOP application recommended"
        )

    # Season explanation
    reasons.append(
        f"{crop_name} is well-suited for the "
        f"{req.season.capitalize()} season"
    )

    return reasons


def calculate_compatibility_score(crop_data: Dict,
                                   req: CropRecommendationRequest) -> float:
    """Calculate 0-100 compatibility score with explanation weights."""
    score = 0.0
    ph_min, ph_max = crop_data["ph_range"]

    # pH score (30 points)
    if ph_min <= req.soil_ph <= ph_max:
        score += 30
    elif abs(req.soil_ph - ph_min) <= 0.5 or abs(req.soil_ph - ph_max) <= 0.5:
        score += 15
    else:
        score += 5

    # Nitrogen score (25 points)
    if req.nitrogen >= crop_data["nitrogen_min"]:
        score += 25
    elif req.nitrogen >= crop_data["nitrogen_min"] * 0.7:
        score += 12
    else:
        score += 3

    # Phosphorus score (20 points)
    if req.phosphorus >= crop_data["phosphorus_min"]:
        score += 20
    elif req.phosphorus >= crop_data["phosphorus_min"] * 0.7:
        score += 10
    else:
        score += 2

    # Potassium score (25 points)
    if req.potassium >= crop_data["potassium_min"]:
        score += 25
    elif req.potassium >= crop_data["potassium_min"] * 0.7:
        score += 12
    else:
        score += 3

    return round(score, 1)


def get_warnings(req: CropRecommendationRequest) -> List[str]:
    """Generate soil health warnings based on parameters."""
    warnings = []

    if req.soil_ph < 5.5:
        warnings.append(
            "Soil is strongly acidic (pH < 5.5). Apply agricultural lime "
            "to raise pH before planting."
        )
    elif req.soil_ph > 8.0:
        warnings.append(
            "Soil is highly alkaline (pH > 8.0). Consider sulphur or "
            "acidifying fertilizers."
        )

    if req.nitrogen < 10:
        warnings.append(
            "Very low nitrogen detected. Apply organic manure or "
            "nitrogen fertilizer before sowing."
        )

    if req.phosphorus < 5:
        warnings.append(
            "Critically low phosphorus. Apply DAP or SSP fertilizer "
            "to improve soil phosphorus levels."
        )

    if req.potassium < 30:
        warnings.append(
            "Low potassium levels detected. Apply Muriate of Potash "
            "(MOP) to prevent yield loss."
        )

    return warnings


def get_confidence_level(score: float) -> Dict:
    """Return confidence label and note based on compatibility score."""
    if score >= 80:
        return {
            "level": "High",
            "note": "Strong match — soil conditions are well-suited for this crop",
            "color": "green"
        }
    elif score >= 60:
        return {
            "level": "Moderate",
            "note": "Good match — minor soil amendments may improve yield",
            "color": "orange"
        }
    else:
        return {
            "level": "Low",
            "note": "Possible but challenging — significant soil preparation needed",
            "color": "red"
        }


# ── Main Endpoint ─────────────────────────────────────────────────────────────

@router.post("/recommend")
async def recommend_crops(req: CropRecommendationRequest, request: Request):
    """
    Recommend crops based on soil parameters with full explanation layer.
    Requires authentication. Returns compatibility scores, reasons, soil analysis, and warnings.
    """
    await _require_authenticated(request)
    try:
        # 0. Validate input parameters
        soil_validation = validate_soil_parameters(
            req.soil_ph, req.nitrogen, req.phosphorus, req.potassium
        )

        if not soil_validation.is_valid():
            logger.error(f"Soil validation failed: {soil_validation.errors}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Soil parameter validation failed",
                    "validation": soil_validation.to_dict()
                }
            )

        # Log validation warnings
        if soil_validation.warnings:
            logger.warning(f"Soil validation warnings: {soil_validation.warnings}")

        # Check cache for existing recommendation
        cached_result = recommendation_cache.get(
            req.soil_ph, req.nitrogen, req.phosphorus, req.potassium, req.season, req.area_size
        )
        if cached_result:
            logger.info("Returning cached recommendation")
            return cached_result

        # 1. Analyze soil
        soil_analysis = analyze_soil(
            req.soil_ph, req.nitrogen,
            req.phosphorus, req.potassium
        )

        # 2. Generate warnings
        warnings = get_warnings(req)

        # 3. Score and rank all crops
        results = []
        for crop_name, crop_data in CROP_DATABASE.items():

            # Season filter
            if req.season not in crop_data["seasons"]:
                continue

            # Calculate compatibility score
            score = calculate_compatibility_score(crop_data, req)

            # Only include crops with score > 20
            if score < 20:
                continue

            # Build explanation reasons
            reasons = build_explanation(crop_name, crop_data, req)

            # Get confidence level
            confidence = get_confidence_level(score)

            # Fertilizer recommendation adjusted for area
            fertilizer = crop_data["fertilizer"]
            if req.area_size:
                fertilizer = (
                    f"For {req.area_size} hectares: " + fertilizer
                )

            results.append({
                "crop": crop_name,
                "compatibility_score": score,
                "confidence": confidence,
                "reasons": reasons,
                "recommended_fertilizer": fertilizer,
                "description": crop_data["description"],
                "optimal_season": req.season,
                "score": score  # for SmartCropRecommendation.jsx compatibility
            })

        # 4. Sort by score descending
        results.sort(key=lambda x: x["compatibility_score"], reverse=True)

        # 5. Return top 5
        top_results = results[:5]

        if not top_results:
            return {
                "success": False,
                "error": (
                    f"No suitable crops found for {req.season} season "
                    f"with current soil parameters. "
                    f"Consider soil amendment or try a different season."
                ),
                "recommendations": [],
                "soil_analysis": soil_analysis,
                "warnings": warnings
            }

        result = {
            "success": True,
            "recommendations": top_results,
            "soil_analysis": soil_analysis,
            "warnings": warnings,
            "total_analyzed": len(CROP_DATABASE),
            "location": req.location,
            "season": req.season
        }

        # Cache the result for future queries
        recommendation_cache.set(
            req.soil_ph, req.nitrogen, req.phosphorus, req.potassium, req.season, req.area_size, result
        )
        logger.info(f"Cached recommendation for {req.location}/{req.season}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Crop recommendation error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while generating recommendations. Please try again.")
