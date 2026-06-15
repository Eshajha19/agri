"""
Crop sustainability analytics — LCA-style water footprint & carbon emission estimates.
"""
from __future__ import annotations
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4
import os

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


EMISSION_FACTORS = {
    "nitrogen_kg_co2e_per_kg": 5.2,
    "phosphorus_kg_co2e_per_kg": 1.1,
    "potassium_kg_co2e_per_kg": 0.58,
    "diesel_kg_co2e_per_liter": 2.68,
    "electricity_kg_co2e_per_kwh": 0.82,
    "machinery_kg_co2e_per_hour": 3.5,
}

IRRIGATION_EFFICIENCY = {
    "rainfed": 1.0,
    "drip": 0.92,
    "sprinkler": 0.75,
    "flood": 0.45,
}
DEFAULT_IRRIGATION_EFFICIENCY = 0.5

SEASON_DAYS = {
    "kharif": 120,
    "rabi": 150,
    "zaid": 90,
}

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

_MAX_HISTORY_USERS = 500
_MAX_HISTORY_PER_USER = 50


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


def _normalize_irrigation_type(irr_type: str) -> str:
    t = (irr_type or "drip").strip().lower()
    return t if t in IRRIGATION_EFFICIENCY else "drip"


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
        # OrderedDict with LRU eviction capped at _MAX_HISTORY_USERS keys
        self._history: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
        self._history_lock = threading.Lock()   # FIX: initialize lock

        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._history_lock = threading.Lock()
        import sys
        self.is_testing = "pytest" in sys.modules or "unittest" in sys.modules

    def _touch_user(self, key: str) -> None:
        """Move key to MRU position, evict LRU entry if cap exceeded."""
        if key in self._history:
            self._history.move_to_end(key)
        else:
            if len(self._history) >= _MAX_HISTORY_USERS:
                self._history.popitem(last=False)
            self._history[key] = []

    def _get_db(self):
        try:
            import firebase_admin
            from firebase_admin import firestore
            if firebase_admin._apps:
                return firestore.client()
            else:
                if not getattr(self, "is_testing", False):
                    logger.warning("Firestore client requested but Firebase Admin SDK has not been initialized.")
        except ImportError as exc:
            if not getattr(self, "is_testing", False):
                logger.warning("Firebase Admin SDK is not installed: %s. Using local fallback.", exc)
        except Exception as exc:
            logger.error("Firestore initialization failed: %s", exc, exc_info=True)
        return None

    def _get_local_file_path(self) -> str:
        """Return the path of the append-only JSONL history file.

        Each line in the file is a JSON-encoded sustainability record.
        The .jsonl extension makes the format explicit and distinguishes
        the new layout from any legacy .json files.
        """
        if getattr(self, "is_testing", False):
            return "sustainability_history_test.jsonl"
        return "sustainability_history.jsonl"

    def _load_local_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Read the append-only JSONL file and rebuild the user-keyed dict.

        Each line is one JSON record containing at minimum a ``user_id`` key.
        Lines that cannot be decoded are skipped with a warning so a single
        corrupt entry never blocks the entire history from loading.
        """
        import json
        import os

        path = self._get_local_file_path()
        history: Dict[str, List[Dict[str, Any]]] = {}
        if not os.path.exists(path):
            return history
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for lineno, raw in enumerate(fh, start=1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                        uid = record.get("user_id", "anonymous")
                        history.setdefault(uid, []).append(record)
                    except json.JSONDecodeError:
                        import logging as _log
                        _log.getLogger(__name__).warning(
                            "Skipping malformed JSONL record at line %d in %s", lineno, path
                        )
        except Exception:
            pass
        return history

    def _append_record_to_file(self, record: Dict[str, Any]) -> None:
        """Append a single JSON record as one line to the JSONL file.

        Opens the file in append mode (``'a'``), so only the new record is
        written.  This is O(1) regardless of how large the history file has
        grown — no read or full-file rewrite is needed.
        """
        import json
        import os

        path = self._get_local_file_path()
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False))
                fh.write("\n")
        except Exception as exc:
            import logging as _log
            _log.getLogger(__name__).warning(
                "Failed to append sustainability record to %s: %s", path, exc
            )

    def _save_local_history(self, history: Dict[str, List[Dict[str, Any]]]) -> None:
        """Full-rewrite used only for compaction / migration.

        Under normal operation, prefer ``_append_record_to_file`` which is
        O(1).  This method rewrites the entire JSONL file from the supplied
        dict and should only be called when explicitly compacting.
        """
        import json

        path = self._get_local_file_path()
        tmp_path = path + ".tmp"
        lock_path = path + ".lock"
        try:
            with open(path, "w", encoding="utf-8") as fh:
                for records in history.values():
                    for record in records:
                        fh.write(json.dumps(record, ensure_ascii=False))
                        fh.write("\n")
        except Exception as exc:
            import logging as _log
            _log.getLogger(__name__).warning(
                "Failed to rewrite sustainability history file %s: %s", path, exc
            )

    def compact_local_history(self, max_records_per_user: int = 50) -> int:
        """Compact the JSONL file by keeping only the most recent records.

        Reads all lines, trims each user's list to *max_records_per_user*
        newest entries, and rewrites the file.  Returns the number of records
        removed.  Suitable for a periodic maintenance job — not called on
        every request.
        """
        history = self._load_local_history()
        total_before = sum(len(v) for v in history.values())
        trimmed = {uid: recs[-max_records_per_user:] for uid, recs in history.items()}
        total_after = sum(len(v) for v in trimmed.values())
        self._save_local_history(trimmed)
        return total_before - total_after

    def get_formula_config(self) -> Dict[str, Any]:
        return {
            "emission_factors": EMISSION_FACTORS,
            "irrigation_efficiency": IRRIGATION_EFFICIENCY,
            "default_irrigation_efficiency": DEFAULT_IRRIGATION_EFFICIENCY,
            "season_days": SEASON_DAYS,
            "crop_coefficients": CROP_COEFFICIENTS,
        }

    def analyze(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = self._normalize_payload(payload)
        crop_key = _normalize_crop(data["crop_type"])
        season_key = _normalize_season(data["season"])
        coeffs = CROP_COEFFICIENTS.get(crop_key, CROP_COEFFICIENTS["default"])
        MAX_ACREAGE = int(os.getenv("MAX_ACREAGE", "10000"))  # configurable limit

        acreage = float(data.get("acreage", 0.1))
        acreage = max(min(acreage, MAX_ACREAGE), 0.1)
        irr_type = data["irrigation_type"]
        irr_eff = IRRIGATION_EFFICIENCY.get(irr_type, DEFAULT_IRRIGATION_EFFICIENCY)
        season_days = SEASON_DAYS[season_key]

        rainfall_mm = data.get("rainfall_mm", 0.0)
        effective_rainfall_mm = data.get("effective_rainfall_mm", rainfall_mm * 0.8)
        rainfall_m3_per_acre = effective_rainfall_mm * 4046.86 / 1000.0
        rainfall_m3_total = rainfall_m3_per_acre * acreage

        base_water = coeffs["water_m3_per_acre_season"] * acreage
        net_irrigation_water = max(base_water - rainfall_m3_total, 0.0)

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

        # Apply reduction ONLY to fertilizer emissions
        fert_co2 *= (1.0 - organic_reduction)

        # Then sum everything normally
        total_carbon_kg = round(
            fert_co2 + machinery_co2 + fuel_co2 + irrigation_energy_co2,
            2,
        )

        carbon_per_acre = round(total_carbon_kg / max(acreage, 0.1), 2)

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
                "fertilizer_kg_co2e": round(fert_co2, 2),
                "machinery_kg_co2e": round(machinery_co2, 2),
                "fuel_kg_co2e": round(fuel_co2, 2),
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
        created_at = datetime.now(timezone.utc).isoformat()
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
                from firebase_admin import firestore
                docs = (
                    db.collection("sustainability_history")
                    .where("user_id", "==", key)
                    .order_by("created_at", direction=firestore.Query.DESCENDING)
                    .limit(limit)
                    .get()
                )
                entries = [d.to_dict() for d in docs]
                entries.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                return entries[:limit]
            except Exception as exc:
                logger.exception("Firestore operation failed: %s", exc)
                return {}


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
            "water_index": result.get("water_index"),
            "carbon_index": result.get("carbon_index"),
            "breakdown": result.get("breakdown", {}),
            "recommendations": result.get("recommendations", []),
            "comparison_chart": result.get("comparison_chart", []),
            "formula_version": result.get("formula_version", "lca-v1"),
        }

        # Save to memory cache
        with self._history_lock:
            if key not in self._history:
                self._history[key] = []
            self._history[key].append(record)
            if len(self._history[key]) > 50:
                self._history[key] = self._history[key][-50:]

        # Save to Firestore primarily
        db = self._get_db()
        if db is not None:
            try:
                db.collection("sustainability_history").document(record["record_id"]).set(record)
            except Exception as exc:
                logger.warning("Failed to persist sustainability record to Firestore: %s", exc)

        # Save to local persistent file as fallback (serialized with lock)
        with self._local_file_lock:
            try:
                local_hist = self._load_local_history()
                if key not in local_hist:
                    local_hist[key] = []
                local_hist[key].append(record)
                if len(local_hist[key]) > 50:
                    local_hist[key] = local_hist[key][-50:]
                self._save_local_history(local_hist)
            except Exception:
                logger.exception("Failed to persist sustainability history to local file")

        # O(1) append to the local JSONL file — no full read/rewrite needed.
        self._append_record_to_file(record)

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "crop_type": str(payload.get("crop_type", "Rice")).strip() or "Rice",
            "season": str(payload.get("season", "Kharif")).strip() or "Kharif",
            "acreage": max(float(payload.get("acreage", 1) or 1), 0.1),
            "irrigation_type": _normalize_irrigation_type(payload.get("irrigation_type", "drip")),
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