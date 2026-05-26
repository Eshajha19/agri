"""
Crop Recommendation API with Explanation Layer
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/crop", tags=["crop"])


# ── Request Model ─────────────────────────────────────────────────────────────
class CropRecommendationRequest(BaseModel):
    soil_ph: float = Field(..., ge=4.0, le=9.0)
    nitrogen: float = Field(..., ge=0, le=100)
    phosphorus: float = Field(..., ge=0, le=50)
    potassium: float = Field(..., ge=0, le=300)
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
async def recommend_crops(req: CropRecommendationRequest):
    """
    Recommend crops based on soil parameters with full explanation layer.
    Returns compatibility scores, reasons, soil analysis, and warnings.
    """
    try:
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

        return {
            "success": True,
            "recommendations": top_results,
            "soil_analysis": soil_analysis,
            "warnings": warnings,
            "total_analyzed": len(CROP_DATABASE),
            "location": req.location,
            "season": req.season
        }

    except Exception as e:
        logger.error(f"Crop recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))# Crop recommendation validation enhanced
