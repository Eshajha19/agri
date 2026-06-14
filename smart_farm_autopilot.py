"""
Smart Farm Autopilot — End-to-End Seasonal Farming Planner
Generates a complete season plan: crop selection, sowing schedule,
irrigation, fertilizer/pesticide timeline, and yield/profit projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Any, Dict, List, Optional
import calendar


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SowingSchedule:
    crop: str
    sowing_start: str
    sowing_end: str
    germination_days: int
    transplant_date: Optional[str]
    harvest_date: str
    total_days: int


@dataclass
class IrrigationEvent:
    week: int
    stage: str
    method: str
    frequency_per_week: int
    water_mm: float
    notes: str


@dataclass
class AgrochemicalEvent:
    week: int
    type: str          # fertilizer | pesticide | fungicide
    product: str
    dose_per_acre: str
    application_method: str
    notes: str


@dataclass
class YieldProjection:
    crop: str
    area_acres: float
    expected_yield_kg: float
    min_yield_kg: float
    max_yield_kg: float
    market_price_per_kg: float
    gross_revenue_inr: float
    input_cost_inr: float
    net_profit_inr: float
    roi_percent: float
    break_even_yield_kg: float


@dataclass
class SeasonPlan:
    plan_id: str
    generated_at: str
    farm_details: Dict[str, Any]
    recommended_crops: List[str]
    primary_crop: str
    season: str
    sowing_schedule: List[Dict]
    irrigation_plan: List[Dict]
    agrochemical_timeline: List[Dict]
    yield_projection: Dict
    risk_factors: List[str]
    advisory_notes: List[str]
    summary: str


# ---------------------------------------------------------------------------
# Crop knowledge base
# ---------------------------------------------------------------------------

CROP_DB: Dict[str, Dict] = {
    # Kharif crops
    "Rice": {
        "seasons": ["Kharif"],
        "soil_types": ["Alluvial", "Clay", "Loamy"],
        "states": ["West Bengal", "Uttar Pradesh", "Punjab", "Andhra Pradesh",
                   "Tamil Nadu", "Bihar", "Telangana", "Karnataka"],
        "sowing_month": 6, "harvest_month": 10, "duration_days": 120,
        "water_need": "high", "germination_days": 7, "transplant_days": 25,
        "yield_kg_per_acre": 1800, "price_per_kg": 22,
        "input_cost_per_acre": 18000,
        "fertilizer_schedule": [
            (1, "Basal DAP", "fertilizer", "50 kg/acre", "Broadcast"),
            (3, "Urea (1st split)", "fertilizer", "25 kg/acre", "Top-dress"),
            (6, "Urea (2nd split)", "fertilizer", "25 kg/acre", "Top-dress"),
            (8, "MOP", "fertilizer", "20 kg/acre", "Top-dress"),
        ],
        "pesticide_schedule": [
            (4, "Chlorpyrifos (stem borer)", "pesticide", "2 ml/L water", "Spray"),
            (7, "Tricyclazole (blast)", "fungicide", "1 g/L water", "Spray"),
            (10, "Carbendazim (sheath blight)", "fungicide", "1 g/L water", "Spray"),
        ],
        "irrigation": [
            (1, "Land prep", "Flood", 3, 80, "Maintain 5 cm standing water"),
            (3, "Tillering", "Flood", 3, 60, "Alternate wetting & drying"),
            (6, "Panicle init.", "Flood", 4, 70, "Critical stage — no stress"),
            (9, "Grain fill", "Flood", 3, 50, "Reduce water 2 weeks before harvest"),
        ],
        "risks": ["Blast disease in humid conditions", "Stem borer infestation",
                  "Flood damage in low-lying fields"],
    },
    "Cotton": {
        "seasons": ["Kharif"],
        "soil_types": ["Black", "Alluvial", "Loamy"],
        "states": ["Maharashtra", "Gujarat", "Telangana", "Andhra Pradesh",
                   "Punjab", "Haryana", "Madhya Pradesh", "Rajasthan"],
        "sowing_month": 5, "harvest_month": 11, "duration_days": 180,
        "water_need": "medium", "germination_days": 5, "transplant_days": None,
        "yield_kg_per_acre": 600, "price_per_kg": 65,
        "input_cost_per_acre": 22000,
        "fertilizer_schedule": [
            (1, "Basal NPK 10:26:26", "fertilizer", "50 kg/acre", "Broadcast"),
            (4, "Urea (1st split)", "fertilizer", "30 kg/acre", "Top-dress"),
            (8, "Urea (2nd split)", "fertilizer", "30 kg/acre", "Top-dress"),
            (12, "Potash (boll dev.)", "fertilizer", "20 kg/acre", "Top-dress"),
        ],
        "pesticide_schedule": [
            (3, "Imidacloprid (sucking pests)", "pesticide", "0.5 ml/L", "Spray"),
            (6, "Profenofos (bollworm)", "pesticide", "2 ml/L", "Spray"),
            (9, "Spinosad (pink bollworm)", "pesticide", "0.3 ml/L", "Spray"),
        ],
        "irrigation": [
            (1, "Germination", "Drip/Furrow", 2, 40, "Light irrigation at sowing"),
            (4, "Squaring", "Drip/Furrow", 2, 50, "Avoid waterlogging"),
            (8, "Flowering", "Drip/Furrow", 3, 60, "Critical — maintain moisture"),
            (12, "Boll dev.", "Drip/Furrow", 2, 45, "Reduce before harvest"),
        ],
        "risks": ["Pink bollworm resistance", "Whitefly-transmitted CLCuV",
                  "Waterlogging in heavy soils"],
    },
    "Soybean": {
        "seasons": ["Kharif"],
        "soil_types": ["Black", "Loamy", "Alluvial"],
        "states": ["Madhya Pradesh", "Maharashtra", "Rajasthan", "Karnataka"],
        "sowing_month": 6, "harvest_month": 10, "duration_days": 100,
        "water_need": "medium", "germination_days": 5, "transplant_days": None,
        "yield_kg_per_acre": 800, "price_per_kg": 45,
        "input_cost_per_acre": 12000,
        "fertilizer_schedule": [
            (1, "Basal SSP + Rhizobium", "fertilizer", "100 kg/acre", "Broadcast"),
            (3, "Urea (light)", "fertilizer", "10 kg/acre", "Top-dress"),
            (6, "MOP", "fertilizer", "15 kg/acre", "Top-dress"),
        ],
        "pesticide_schedule": [
            (3, "Thiamethoxam (aphids)", "pesticide", "0.3 g/L", "Spray"),
            (6, "Chlorantraniliprole (pod borer)", "pesticide", "0.4 ml/L", "Spray"),
        ],
        "irrigation": [
            (1, "Germination", "Sprinkler", 2, 30, "Light irrigation"),
            (4, "Flowering", "Sprinkler", 2, 40, "Critical stage"),
            (7, "Pod fill", "Sprinkler", 2, 35, "Maintain moisture"),
        ],
        "risks": ["Yellow mosaic virus", "Pod borer", "Waterlogging sensitivity"],
    },
    # Rabi crops
    "Wheat": {
        "seasons": ["Rabi"],
        "soil_types": ["Alluvial", "Loamy", "Clay"],
        "states": ["Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh",
                   "Rajasthan", "Bihar", "Gujarat"],
        "sowing_month": 11, "harvest_month": 4, "duration_days": 150,
        "water_need": "medium", "germination_days": 7, "transplant_days": None,
        "yield_kg_per_acre": 2000, "price_per_kg": 22,
        "input_cost_per_acre": 16000,
        "fertilizer_schedule": [
            (1, "Basal DAP", "fertilizer", "50 kg/acre", "Broadcast"),
            (3, "Urea (CRI stage)", "fertilizer", "33 kg/acre", "Top-dress"),
            (5, "Urea (tillering)", "fertilizer", "33 kg/acre", "Top-dress"),
            (8, "Urea (flag leaf)", "fertilizer", "17 kg/acre", "Top-dress"),
        ],
        "pesticide_schedule": [
            (4, "Clodinafop (narrow weeds)", "pesticide", "60 g/acre", "Spray"),
            (6, "Propiconazole (rust)", "fungicide", "1 ml/L", "Spray"),
            (9, "Dimethoate (aphids)", "pesticide", "1.5 ml/L", "Spray"),
        ],
        "irrigation": [
            (1, "Crown root init.", "Flood/Sprinkler", 1, 60, "First critical irrigation"),
            (3, "Tillering", "Flood/Sprinkler", 1, 55, "Second irrigation"),
            (5, "Jointing", "Flood/Sprinkler", 1, 50, "Third irrigation"),
            (7, "Flowering", "Flood/Sprinkler", 1, 50, "Fourth — critical"),
            (9, "Grain fill", "Flood/Sprinkler", 1, 45, "Fifth irrigation"),
            (11, "Dough stage", "Flood/Sprinkler", 1, 40, "Sixth — last irrigation"),
        ],
        "risks": ["Yellow rust in cool humid weather", "Aphid infestation",
                  "Terminal heat stress if sown late"],
    },
    "Mustard": {
        "seasons": ["Rabi"],
        "soil_types": ["Alluvial", "Sandy", "Loamy"],
        "states": ["Rajasthan", "Haryana", "Uttar Pradesh", "Madhya Pradesh",
                   "Punjab", "West Bengal"],
        "sowing_month": 10, "harvest_month": 2, "duration_days": 120,
        "water_need": "low", "germination_days": 5, "transplant_days": None,
        "yield_kg_per_acre": 500, "price_per_kg": 55,
        "input_cost_per_acre": 10000,
        "fertilizer_schedule": [
            (1, "Basal NPK 12:32:16", "fertilizer", "40 kg/acre", "Broadcast"),
            (3, "Urea", "fertilizer", "25 kg/acre", "Top-dress"),
            (5, "Sulphur (flowering)", "fertilizer", "10 kg/acre", "Top-dress"),
        ],
        "pesticide_schedule": [
            (3, "Dimethoate (aphids)", "pesticide", "1.5 ml/L", "Spray"),
            (5, "Mancozeb (Alternaria)", "fungicide", "2 g/L", "Spray"),
        ],
        "irrigation": [
            (1, "Germination", "Flood/Sprinkler", 1, 40, "Pre-sowing moisture"),
            (4, "Flowering", "Flood/Sprinkler", 1, 45, "Critical stage"),
            (7, "Pod fill", "Flood/Sprinkler", 1, 35, "Last irrigation"),
        ],
        "risks": ["Aphid attack at flowering", "Alternaria blight",
                  "Frost damage in December"],
    },
    "Chickpea": {
        "seasons": ["Rabi"],
        "soil_types": ["Black", "Loamy", "Sandy", "Alluvial"],
        "states": ["Madhya Pradesh", "Rajasthan", "Maharashtra", "Uttar Pradesh",
                   "Andhra Pradesh", "Karnataka"],
        "sowing_month": 10, "harvest_month": 2, "duration_days": 110,
        "water_need": "low", "germination_days": 6, "transplant_days": None,
        "yield_kg_per_acre": 700, "price_per_kg": 60,
        "input_cost_per_acre": 9000,
        "fertilizer_schedule": [
            (1, "Basal DAP + Rhizobium", "fertilizer", "25 kg/acre", "Broadcast"),
            (4, "Sulphur", "fertilizer", "10 kg/acre", "Top-dress"),
        ],
        "pesticide_schedule": [
            (4, "Helicoverpa NPV (pod borer)", "pesticide", "250 ml/acre", "Spray"),
            (6, "Chlorpyrifos (cut worm)", "pesticide", "2 ml/L", "Spray"),
        ],
        "irrigation": [
            (1, "Germination", "Sprinkler", 1, 30, "Pre-sowing only if dry"),
            (5, "Flowering", "Sprinkler", 1, 35, "Critical — avoid excess"),
            (8, "Pod fill", "Sprinkler", 1, 30, "Light irrigation"),
        ],
        "risks": ["Pod borer (Helicoverpa)", "Wilt disease",
                  "Excess moisture causes root rot"],
    },
    # Zaid crops
    "Watermelon": {
        "seasons": ["Zaid"],
        "soil_types": ["Sandy", "Loamy", "Alluvial"],
        "states": ["Uttar Pradesh", "Andhra Pradesh", "Karnataka", "Maharashtra",
                   "Rajasthan", "Gujarat"],
        "sowing_month": 2, "harvest_month": 5, "duration_days": 90,
        "water_need": "medium", "germination_days": 5, "transplant_days": None,
        "yield_kg_per_acre": 8000, "price_per_kg": 8,
        "input_cost_per_acre": 20000,
        "fertilizer_schedule": [
            (1, "Basal FYM + NPK", "fertilizer", "2 tons FYM + 30 kg NPK", "Broadcast"),
            (3, "Urea (vine dev.)", "fertilizer", "20 kg/acre", "Top-dress"),
            (5, "Potash (fruit set)", "fertilizer", "20 kg/acre", "Fertigation"),
        ],
        "pesticide_schedule": [
            (2, "Imidacloprid (aphids)", "pesticide", "0.5 ml/L", "Spray"),
            (4, "Mancozeb (downy mildew)", "fungicide", "2 g/L", "Spray"),
        ],
        "irrigation": [
            (1, "Germination", "Drip", 3, 25, "Light frequent irrigation"),
            (3, "Vine dev.", "Drip", 4, 35, "Increase frequency"),
            (5, "Fruit set", "Drip", 5, 45, "Critical — maintain moisture"),
            (7, "Ripening", "Drip", 2, 20, "Reduce water for sweetness"),
        ],
        "risks": ["Downy mildew in humid conditions", "Fruit fly",
                  "Cracking due to irregular irrigation"],
    },
    "Maize": {
        "seasons": ["Kharif", "Zaid"],
        "soil_types": ["Alluvial", "Loamy", "Sandy", "Red"],
        "states": ["Karnataka", "Andhra Pradesh", "Telangana", "Maharashtra",
                   "Bihar", "Uttar Pradesh", "Rajasthan"],
        "sowing_month": 6, "harvest_month": 9, "duration_days": 90,
        "water_need": "medium", "germination_days": 5, "transplant_days": None,
        "yield_kg_per_acre": 1500, "price_per_kg": 20,
        "input_cost_per_acre": 14000,
        "fertilizer_schedule": [
            (1, "Basal NPK 10:26:26", "fertilizer", "50 kg/acre", "Broadcast"),
            (3, "Urea (knee-high)", "fertilizer", "33 kg/acre", "Top-dress"),
            (5, "Urea (tasseling)", "fertilizer", "33 kg/acre", "Top-dress"),
        ],
        "pesticide_schedule": [
            (3, "Chlorpyrifos (FAW)", "pesticide", "2 ml/L", "Spray"),
            (5, "Emamectin (FAW)", "pesticide", "0.4 g/L", "Spray"),
        ],
        "irrigation": [
            (1, "Germination", "Furrow", 2, 40, "Ensure good germination"),
            (3, "Knee-high", "Furrow", 2, 45, "Vegetative growth"),
            (5, "Tasseling", "Furrow", 3, 55, "Critical — no stress"),
            (7, "Grain fill", "Furrow", 2, 40, "Maintain moisture"),
        ],
        "risks": ["Fall Armyworm (FAW)", "Stem borer", "Drought at tasseling"],
    },
}

# State → season mapping
STATE_SEASON_MAP: Dict[str, List[str]] = {
    "Punjab": ["Kharif", "Rabi"],
    "Haryana": ["Kharif", "Rabi"],
    "Uttar Pradesh": ["Kharif", "Rabi", "Zaid"],
    "Madhya Pradesh": ["Kharif", "Rabi"],
    "Rajasthan": ["Kharif", "Rabi", "Zaid"],
    "Maharashtra": ["Kharif", "Rabi"],
    "Gujarat": ["Kharif", "Rabi", "Zaid"],
    "Karnataka": ["Kharif", "Rabi", "Zaid"],
    "Andhra Pradesh": ["Kharif", "Zaid"],
    "Telangana": ["Kharif", "Zaid"],
    "Tamil Nadu": ["Kharif", "Rabi"],
    "West Bengal": ["Kharif", "Rabi"],
    "Bihar": ["Kharif", "Rabi", "Zaid"],
}

SEASON_MONTHS: Dict[str, Dict[str, int]] = {
    "Kharif": {"start": 6, "end": 10},
    "Rabi":   {"start": 11, "end": 4},
    "Zaid":   {"start": 2, "end": 5},
}


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class SmartFarmAutopilot:
    """Generates a complete seasonal farming plan from farm inputs."""

    def generate_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point. Returns a full season plan dict.

        Parameters
        ----------
        payload : dict
            farm_name, state, district, area_acres, soil_type,
            season, water_source, budget_inr (optional)
        """
        state      = payload.get("state", "Maharashtra")
        soil_type  = payload.get("soil_type", "Black")
        season     = payload.get("season", "Kharif")
        area_acres = float(payload.get("area_acres", 1.0))
        water_src  = payload.get("water_source", "Canal")
        budget     = float(payload.get("budget_inr", 0) or 0)

        # 1. Select suitable crops
        suitable = self._select_crops(state, soil_type, season)
        if not suitable:
            suitable = self._fallback_crops(season)

        primary = suitable[0]
        crop_data = CROP_DB[primary]

        # 2. Build sowing schedule
        sowing = self._build_sowing_schedule(primary, crop_data, season)

        # 3. Build irrigation plan
        irrigation = self._build_irrigation_plan(crop_data, water_src)

        # 4. Build agrochemical timeline
        agrochem = self._build_agrochem_timeline(crop_data)

        # 5. Yield & profit projection
        projection = self._build_yield_projection(primary, crop_data, area_acres)

        # 6. Risk factors
        risks = list(crop_data.get("risks", []))
        if budget and projection.input_cost_inr > budget:
            risks.append(
                f"Estimated input cost ₹{projection.input_cost_inr:,.0f} "
                f"exceeds your budget of ₹{budget:,.0f}. Consider reducing area or crop."
            )

        # 7. Advisory notes
        notes = self._build_advisory_notes(primary, crop_data, water_src, season)

        import uuid
        from datetime import datetime as dt

        plan = {
            "plan_id": str(uuid.uuid4())[:8].upper(),
            "generated_at": dt.now().isoformat(),
            "farm_details": {
                "farm_name": payload.get("farm_name", "My Farm"),
                "state": state,
                "district": payload.get("district", ""),
                "area_acres": area_acres,
                "soil_type": soil_type,
                "season": season,
                "water_source": water_src,
            },
            "recommended_crops": suitable[:4],
            "primary_crop": primary,
            "season": season,
            "sowing_schedule": [asdict(s) for s in sowing],
            "irrigation_plan": [asdict(i) for i in irrigation],
            "agrochemical_timeline": [asdict(a) for a in agrochem],
            "yield_projection": asdict(projection),
            "risk_factors": risks,
            "advisory_notes": notes,
            "summary": self._build_summary(primary, season, projection, area_acres),
        }
        return plan

    # ── Private helpers ──────────────────────────────────────────────────────

    def _select_crops(self, state: str, soil_type: str, season: str) -> List[str]:
        matches = []
        for crop, data in CROP_DB.items():
            if (season in data["seasons"]
                    and soil_type in data["soil_types"]
                    and state in data["states"]):
                matches.append(crop)
        # Sort by yield value (yield_kg * price)
        matches.sort(
            key=lambda c: CROP_DB[c]["yield_kg_per_acre"] * CROP_DB[c]["price_per_kg"],
            reverse=True,
        )
        return matches

    def _fallback_crops(self, season: str) -> List[str]:
        """Return default crops when no exact match found."""
        defaults = {"Kharif": ["Rice", "Maize"], "Rabi": ["Wheat", "Mustard"], "Zaid": ["Watermelon", "Maize"]}
        return defaults.get(season, ["Wheat"])

    def _build_sowing_schedule(
        self, crop: str, data: Dict, season: str
    ) -> List[SowingSchedule]:
        today = date.today()
        year = today.year
        sow_month = data["sowing_month"]
        # Adjust year: if the sowing month has already passed this year,
        # advance to the next calendar year.
        months_passed = today.month - sow_month
        if months_passed <= 2:
            # Sowing month is current, upcoming, or just passed (within 2-month buffer)
            sow_year = year
        else:
            # Sowing month has clearly passed
            sow_year = year + 1
        sow_start = date(sow_year, sow_month, 1)
        sow_end   = date(sow_year, sow_month, min(20, calendar.monthrange(sow_year, sow_month)[1]))

        transplant = None
        if data.get("transplant_days"):
            t = sow_start + timedelta(days=data["transplant_days"])
            transplant = t.isoformat()

        harvest_days = data["duration_days"]
        harvest = sow_start + timedelta(days=harvest_days)

        return [SowingSchedule(
            crop=crop,
            sowing_start=sow_start.isoformat(),
            sowing_end=sow_end.isoformat(),
            germination_days=data["germination_days"],
            transplant_date=transplant,
            harvest_date=harvest.isoformat(),
            total_days=harvest_days,
        )]

    def _build_irrigation_plan(
        self, data: Dict, water_source: str
    ) -> List[IrrigationEvent]:
        events = []
        for week, stage, method, freq, mm, notes in data.get("irrigation", []):
            # Adjust method based on water source
            if water_source in ("Borewell", "Drip") and method == "Flood":
                method = "Drip/Sprinkler"
            events.append(IrrigationEvent(
                week=week, stage=stage, method=method,
                frequency_per_week=freq, water_mm=mm, notes=notes,
            ))
        return events

    def _build_agrochem_timeline(self, data: Dict) -> List[AgrochemicalEvent]:
        events = []
        for week, product, atype, dose, method in data.get("fertilizer_schedule", []):
            events.append(AgrochemicalEvent(
                week=week, type=atype, product=product,
                dose_per_acre=dose, application_method=method,
                notes="Apply in the morning or evening. Avoid application before rain.",
            ))
        for week, product, atype, dose, method in data.get("pesticide_schedule", []):
            events.append(AgrochemicalEvent(
                week=week, type=atype, product=product,
                dose_per_acre=dose, application_method=method,
                notes="Wear protective gear. Follow label instructions.",
            ))
        events.sort(key=lambda e: e.week)
        return events

    def _build_yield_projection(
        self, crop: str, data: Dict, area_acres: float
    ) -> YieldProjection:
        base_yield = data["yield_kg_per_acre"] * area_acres
        price      = data["price_per_kg"]
        input_cost = data["input_cost_per_acre"] * area_acres
        revenue    = base_yield * price
        profit     = revenue - input_cost
        roi        = (profit / input_cost * 100) if input_cost else 0
        break_even = input_cost / price if price else 0

        return YieldProjection(
            crop=crop,
            area_acres=area_acres,
            expected_yield_kg=round(base_yield),
            min_yield_kg=round(base_yield * 0.75),
            max_yield_kg=round(base_yield * 1.25),
            market_price_per_kg=price,
            gross_revenue_inr=round(revenue),
            input_cost_inr=round(input_cost),
            net_profit_inr=round(profit),
            roi_percent=round(roi, 1),
            break_even_yield_kg=round(break_even),
        )

    def _build_advisory_notes(
        self, crop: str, data: Dict, water_source: str, season: str
    ) -> List[str]:
        notes = [
            f"Obtain certified {crop} seeds from a registered dealer before sowing.",
            "Conduct soil testing before applying fertilizers for accurate dosing.",
            f"Water requirement is {data['water_need']} — plan {water_source} usage accordingly.",
            "Keep field records (spray diary) for traceability and insurance claims.",
            "Register on PM-FASAL BIMA YOJANA before the cut-off date for crop insurance.",
        ]
        if season == "Kharif":
            notes.append("Monitor weather forecasts during monsoon for timely spray decisions.")
        elif season == "Rabi":
            notes.append("Protect crop from frost in December–January using smoke or irrigation.")
        elif season == "Zaid":
            notes.append("Use mulching to conserve soil moisture during summer heat.")
        return notes

    def _build_summary(
        self, crop: str, season: str, proj: YieldProjection, area: float
    ) -> str:
        return (
            f"Your {season} season plan recommends growing {crop} on {area} acre(s). "
            f"Expected yield: {proj.expected_yield_kg:,} kg. "
            f"Projected net profit: ₹{proj.net_profit_inr:,} "
            f"(ROI: {proj.roi_percent}%). "
            f"Break-even yield: {proj.break_even_yield_kg:,} kg."
        )


# Module-level singleton
_autopilot = SmartFarmAutopilot()


def generate_season_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Public function called from main.py."""
    return _autopilot.generate_plan(payload)
