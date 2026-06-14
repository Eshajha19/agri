"""
Crop sustainability analytics — LCA-style water footprint & carbon emission estimates.

Coefficients are configurable via CROP_COEFFICIENTS and EMISSION_FACTORS.
Values are illustrative regional averages (India-centric) for advisory use, not
certified carbon accounting.
"""
from __future__ import annotations
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# kg CO2e per kg of nutrient applied (cradle-to-field, simplified)
EMISSION_FACTORS = {
    "nitrogen_kg_co2e_per_kg": 5.2,
    "phosphorus_kg_co2e_per_kg": 1.1,
    "potassium_kg_co2e_per_kg": 0.58,
    "diesel_kg_co2e_per_liter": 2.68,
    "electricity_kg_co2e_per_kwh": 0.82,
    "machinery_kg_co2e_per_hour": 3.5,
}

# Irrigation efficiency (fraction of applied water that reaches crop)
IRRIGATION_EFFICIENCY = {
    "rainfed": 1.0,
    "drip": 0.92,
    "sprinkler": 0.75,
    "flood": 0.45,
}

# Season length in days
SEASON_DAYS = {
    "kharif": 120,
    "rabi": 150,
    "zaid": 90,
}

# Per-acre reference values; scaled by crop water_need_factor
CROP_COEFFICIENTS: Dict[str, Dict[str, float]] = {
    "rice": {"water_m3_per_acre_season": 4500, "base_fertilizer_n": 80, "base_fertilizer_p": 40, "base_fertilizer_k": 30, "machinery_hours": 14},
    "wheat": {"water_m3_per_acre_season": 1800, "base_fertilizer_n": 60, "base_fertilizer_p": 30, "base_fertilizer_k": 20, "machinery_hours": 10},
    "maize": {"water_m3_per_acre_season": 2200, "base_fertilizer_n": 70, "base_fertilizer_p": 35, "base_fertilizer_k": 25, "machinery_hours": 12},
    "cotton": {"water_m3_per_acre_season": 3200, "base_fertilizer_n": 65, "base_fertilizer_p": 32, "base_fertilizer_k": 22, "machinery_hours": 16},
    "sugarcane": {"water_m3_per_acre_season": 5500, "base_fertilizer_n": 90, "base_fertilizer_p": 45, "base_fertilizer_k": 35, "machinery_hours": 20},
    "pulses": {"water_m3_per_acre_season": 1200, "base_fertilizer_n": 25, "base_fertilizer_p": 20, "base_fertilizer_k": 15, "machinery_hours": 8},
    "vegetables": {"water_m3_per_acre_season": 2800, "base_fertilizer_n": 55, "base_fertilizer_p": 28, "base_fertilizer_k": 18, "machinery_hours": 12},
    "default": {"water_m3_per_acre_season": 2000, "base_fertilizer_n": 50, "base_fertilizer_p": 25, "base_fertilizer_k": 20, "machinery_hours": 10},
}


def _normalize_crop(crop: str) -> str:
    key = (crop or "").strip().lower()
    aliases = {
        "paddy": "rice",
        "chawal": "rice",
        "gehu": "wheat",
        "makka": "maize",
        "corn": "maize",
        "kapas": "cotton",
        "ganna": "sugarcane",
        "dal": "pulses",
        "tur": "pulses",
        "arhar": "pulses",
    }
    return aliases.get(key, key if key in CROP_COEFFICIENTS else "default")


def _normalize_season(season: str) -> str:
    s = (season or "kharif").strip().lower()
    if s in SEASON_DAYS:
        return s
    return "kharif"


@dataclass
class SustainabilityRecord:
    record_id: str
    user_id: str
    crop_type: str
    season: str
    acreage: float
    created_at: str
    water_footprint_m3: float
    carbon_emissions_kg: float
    sustainability_score: int
    breakdown: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class SustainabilityAnalytics:
    """LCA-style sustainability engine with in-memory history."""

    def __init__(self) -> None:
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._history_lock = threading.Lock()
        import sys
        self.is_testing = "pytest" in sys.modules or "unittest" in sys.modules

    def _get_db(self):
        try:
            import firebase_admin
            from firebase_admin import firestore
            if firebase_admin._apps:
                return firestore.client()
        except Exception:
            pass
        return None

    def _get_local_file_path(self) -> str:
        if getattr(self, "is_testing", False):
            return "sustainability_history_test.json"
        return "sustainability_history.json"

    def _load_local_history(self) -> Dict[str, List[Dict[str, Any]]]:
        import json
        import os
        from filelock import FileLock
        path = self._get_local_file_path()
        lock_path = path + ".lock"
        if os.path.exists(path):
            try:
                with FileLock(lock_path, timeout=5):
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception as e:
                logger.error("Failed to read local history: %s", e)
        return {}

    def _save_local_history(self, history: Dict[str, List[Dict[str, Any]]]) -> None:
        import json
        import os
        from filelock import FileLock
        path = self._get_local_file_path()
        tmp_path = path + ".tmp"
        lock_path = path + ".lock"
        try:
            with FileLock(lock_path, timeout=5):
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
        except Exception as e:
            logger.error("Failed to save local history atomically: %s", e)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def get_formula_config(self) -> Dict[str, Any]:
        return {
            "emission_factors": EMISSION_FACTORS,
            "irrigation_efficiency": IRRIGATION_EFFICIENCY,
            "season_days": SEASON_DAYS,
            "crop_coefficients": CROP_COEFFICIENTS,
        }

    def analyze(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = self._normalize_payload(payload)
        crop_key = _normalize_crop(data["crop_type"])
        season_key = _normalize_season(data["season"])
        coeffs = CROP_COEFFICIENTS.get(crop_key, CROP_COEFFICIENTS["default"])
        acreage = data["acreage"]
        irr_type = data["irrigation_type"]
        irr_eff = IRRIGATION_EFFICIENCY.get(irr_type, 0.7)
        season_days = SEASON_DAYS[season_key]

        # --- Water footprint (m³) ---
        base_water = coeffs["water_m3_per_acre_season"] * acreage
        if irr_type == "rainfed":
            green_water = base_water * 0.85
            blue_water = base_water * 0.15
        else:
            applied = base_water / max(irr_eff, 0.3)
            blue_water = applied * 0.9
            green_water = base_water * 0.1

        irrigation_events = data["irrigation_events"]
        event_factor = 1.0 + min(irrigation_events, 40) * 0.01
        total_water_m3 = round((blue_water + green_water) * event_factor, 2)
        water_per_acre = round(total_water_m3 / max(acreage, 0.1), 2)

        # --- Carbon emissions (kg CO2e) ---
        n_kg = data["fertilizer_n_kg"] or coeffs["base_fertilizer_n"] * acreage
        p_kg = data["fertilizer_p_kg"] or coeffs["base_fertilizer_p"] * acreage
        k_kg = data["fertilizer_k_kg"] or coeffs["base_fertilizer_k"] * acreage

        ef = EMISSION_FACTORS
        fert_co2 = (
            n_kg * ef["nitrogen_kg_co2e_per_kg"]
            + p_kg * ef["phosphorus_kg_co2e_per_kg"]
            + k_kg * ef["potassium_kg_co2e_per_kg"]
        )

        machinery_hours = data["machinery_hours"] or coeffs["machinery_hours"] * acreage
        diesel_liters = data["diesel_liters"]
        if diesel_liters is None:
            diesel_liters = machinery_hours * 2.8
        machinery_co2 = machinery_hours * ef["machinery_kg_co2e_per_hour"]
        fuel_co2 = diesel_liters * ef["diesel_kg_co2e_per_liter"]

        pump_kwh = 0.0
        if irr_type in ("drip", "sprinkler", "flood"):
            pump_kwh = irrigation_events * 4.5 * acreage * (1.1 if irr_type == "flood" else 0.7)
        irrigation_energy_co2 = pump_kwh * ef["electricity_kg_co2e_per_kwh"]

        organic_reduction = 0.12 if data["organic_practices"] else 0.0
        total_carbon_kg = round(
            (fert_co2 + machinery_co2 + fuel_co2 + irrigation_energy_co2) * (1.0 - organic_reduction),
            2,
        )
        carbon_per_acre = round(total_carbon_kg / max(acreage, 0.1), 2)

        # Benchmark vs crop average (simplified)
        benchmark_water = coeffs["water_m3_per_acre_season"] * acreage
        benchmark_carbon = (
            (coeffs["base_fertilizer_n"] * acreage * ef["nitrogen_kg_co2e_per_kg"])
            + (coeffs["machinery_hours"] * acreage * ef["machinery_kg_co2e_per_hour"])
        )

        water_index = min(100, max(0, int(100 - ((total_water_m3 - benchmark_water) / max(benchmark_water, 1)) * 40)))
        carbon_index = min(100, max(0, int(100 - ((total_carbon_kg - benchmark_carbon) / max(benchmark_carbon, 1)) * 35)))
        sustainability_score = int(round((water_index + carbon_index) / 2))

        recommendations = self._build_recommendations(
            crop_key=crop_key,
            irr_type=irr_type,
            total_water_m3=total_water_m3,
            total_carbon_kg=total_carbon_kg,
            organic=data["organic_practices"],
            sustainability_score=sustainability_score,
        )

        breakdown = {
            "water": {
                "total_m3": total_water_m3,
                "per_acre_m3": water_per_acre,
                "blue_water_m3": round(blue_water * event_factor, 2),
                "green_water_m3": round(green_water * event_factor, 2),
                "irrigation_type": irr_type,
                "irrigation_efficiency": irr_eff,
            },
            "carbon": {
                "total_kg_co2e": total_carbon_kg,
                "per_acre_kg_co2e": carbon_per_acre,
                "fertilizer_kg_co2e": round(fert_co2 * (1.0 - organic_reduction), 2),
                "machinery_kg_co2e": round(machinery_co2, 2),
                "fuel_kg_co2e": round(fuel_co2 * (1.0 - organic_reduction), 2),
                "irrigation_energy_kg_co2e": round(irrigation_energy_co2, 2),
            },
            "inputs": {
                "fertilizer_n_kg": round(n_kg, 2),
                "fertilizer_p_kg": round(p_kg, 2),
                "fertilizer_k_kg": round(k_kg, 2),
                "machinery_hours": round(machinery_hours, 2),
                "diesel_liters": round(diesel_liters, 2),
                "irrigation_events": irrigation_events,
                "season_days": season_days,
            },
        }

        chart_comparison = [
            {"metric": "Water (m³)", "current": total_water_m3, "benchmark": round(benchmark_water, 2)},
            {"metric": "Carbon (kg)", "current": total_carbon_kg, "benchmark": round(benchmark_carbon, 2)},
        ]

        record_id = str(uuid4())
        created_at = datetime.utcnow().isoformat() + "Z"
        user_id = data["user_id"] or "anonymous"

        result = {
            "record_id": record_id,
            "crop_type": data["crop_type"],
            "season": season_key.title(),
            "acreage": acreage,
            "created_at": created_at,
            "water_footprint_m3": total_water_m3,
            "carbon_emissions_kg_co2e": total_carbon_kg,
            "sustainability_score": sustainability_score,
            "water_index": water_index,
            "carbon_index": carbon_index,
            "breakdown": breakdown,
            "comparison_chart": chart_comparison,
            "recommendations": recommendations,
            "formula_version": "lca-v1",
        }

        self._append_history(user_id, result)
        return result

    def get_history(self, user_id: str, limit: int = 12) -> List[Dict[str, Any]]:
        key = user_id or "anonymous"
        db = self._get_db()
        if db is not None:
            try:
                docs = db.collection("sustainability_history").where("user_id", "==", key).limit(limit).get()
                entries = [d.to_dict() for d in docs]
                # Sort entries by created_at descending (newest first)
                entries.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                return entries[:limit]
            except Exception:
                pass

        # Fallback to local history
        local_hist = self._load_local_history()
        entries = local_hist.get(key, [])
        with self._history_lock:
            self._history[key] = entries
        return list(reversed(entries[-limit:]))

    def _append_history(self, user_id: str, result: Dict[str, Any]) -> None:
        key = user_id or "anonymous"
        record = {
            "record_id": result["record_id"],
            "user_id": key,
            "crop_type": result["crop_type"],
            "season": result["season"],
            "acreage": result["acreage"],
            "created_at": result["created_at"],
            "water_footprint_m3": result["water_footprint_m3"],
            "carbon_emissions_kg_co2e": result["carbon_emissions_kg_co2e"],
            "sustainability_score": result["sustainability_score"],
        }

        # Save to memory cache
        with self._history_lock:
            if key not in self._history:
                self._history[key] = []
            self._history[key].append(record)
            if len(self._history[key]) > 50:
                self._history[key] = self._history[key][-50:]

        # Save to Firestore primarilly
        db = self._get_db()
        if db is not None:
            try:
                db.collection("sustainability_history").document(record["record_id"]).set(record)
            except Exception:
                pass

        # Save to local persistent file as fallback
        try:
            local_hist = self._load_local_history()
            if key not in local_hist:
                local_hist[key] = []
            local_hist[key].append(record)
            if len(local_hist[key]) > 50:
                local_hist[key] = local_hist[key][-50:]
            self._save_local_history(local_hist)
        except Exception:
            pass

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "crop_type": str(payload.get("crop_type", "Rice")).strip() or "Rice",
            "season": str(payload.get("season", "Kharif")).strip() or "Kharif",
            "acreage": max(float(payload.get("acreage", 1) or 1), 0.1),
            "irrigation_type": str(payload.get("irrigation_type", "drip")).lower(),
            "irrigation_events": max(int(payload.get("irrigation_events", 10) or 0), 0),
            "fertilizer_n_kg": float(payload.get("fertilizer_n_kg")) if payload.get("fertilizer_n_kg") is not None else None,
            "fertilizer_p_kg": float(payload.get("fertilizer_p_kg")) if payload.get("fertilizer_p_kg") is not None else None,
            "fertilizer_k_kg": float(payload.get("fertilizer_k_kg")) if payload.get("fertilizer_k_kg") is not None else None,
            "machinery_hours": float(payload.get("machinery_hours")) if payload.get("machinery_hours") is not None else None,
            "diesel_liters": float(payload.get("diesel_liters")) if payload.get("diesel_liters") is not None else None,
            "organic_practices": bool(payload.get("organic_practices", False)),
            "user_id": str(payload.get("user_id", "")).strip() or None,
        }

    def _build_recommendations(
        self,
        *,
        crop_key: str,
        irr_type: str,
        total_water_m3: float,
        total_carbon_kg: float,
        organic: bool,
        sustainability_score: int,
    ) -> List[str]:
        tips: List[str] = []
        if irr_type == "flood":
            tips.append("Switch from flood irrigation to drip or sprinkler to cut blue-water use by 30–50%.")
        if irr_type == "rainfed" and crop_key == "rice":
            tips.append("Consider SRI or alternate wetting for paddy to lower seasonal water demand.")
        if total_carbon_kg > 500:
            tips.append("Split nitrogen applications and use soil tests to avoid over-fertilization.")
        if not organic:
            tips.append("Integrate compost or green manure to reduce synthetic fertilizer dependency.")
        if sustainability_score < 50:
            tips.append("Plan crop rotation with legumes next season to improve soil carbon and lower N inputs.")
        else:
            tips.append("Your footprint is near regional benchmarks — maintain current efficient practices.")
        tips.append("Schedule machinery in fewer passes to reduce diesel use and field compaction.")
        return tips[:5]
