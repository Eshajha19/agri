from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from backend.core.constants import CROP_THRESHOLDS


def validate_inputs(
    crop: Optional[str] = None,
    temperature: Optional[float] = None,
    soil_moisture: Optional[float] = None,
    humidity: Optional[float] = None,
    ph: Optional[float] = None,
    phosphorus: Optional[float] = None,
    irrigation_count: Optional[int] = None,
    water_coverage: Optional[int] = None,
    season: Optional[str] = None
) -> Dict[str, Any]:
    errors = []

    if crop and crop.strip().lower() not in CROP_THRESHOLDS:
        errors.append(f"Invalid crop. Supported: {', '.join(CROP_THRESHOLDS.keys())}")

    if temperature is not None and (temperature < -50 or temperature > 60):
        errors.append("Temperature must be between -50°C and 60°C")

    if soil_moisture is not None and (soil_moisture < 0 or soil_moisture > 100):
        errors.append("Soil moisture must be between 0% and 100%")

    if humidity is not None and (humidity < 0 or humidity > 100):
        errors.append("Humidity must be between 0% and 100%")

    if ph is not None and (ph < 3 or ph > 10):
        errors.append("pH must be between 3.0 and 10.0")

    if phosphorus is not None and phosphorus < 0:
        errors.append("Phosphorus level cannot be negative")

    if irrigation_count is not None and irrigation_count < 0:
        errors.append("Irrigation count cannot be negative")

    if water_coverage is not None and (water_coverage < 0 or water_coverage > 100):
        errors.append("Water coverage must be between 0% and 100%")

    if season and season.strip().lower() not in {"kharif", "rabi", "zaid"}:
        errors.append("Season must be one of: kharif, rabi, zaid")

    if errors:
        logger.error(f"Validation errors: {', '.join(errors)}")
        return {
            "valid": False,
            "errors": errors
        }

    return {"valid": True, "errors": []}


def get_season_from_month(month: int) -> str:
    if month < 1 or month > 12:
        raise ValueError(f"Invalid month: {month}")
    if month in [6, 7, 8, 9]:
        return "kharif"
    elif month in [11, 12, 1, 2]:
        return "rabi"
    return "zaid"


def generate_weather_advisories(
    temperature: Optional[float] = None,
    humidity: Optional[float] = None,
    alerts: List[Dict] = None
) -> List[Dict]:
    if alerts is None:
        alerts = []

    try:
        if temperature is not None:
            if temperature >= 40:
                alerts.append({
                    "type": "critical",
                    "message": f"Extreme heat stress detected ({temperature}°C). Increase irrigation frequency immediately.",
                    "category": "weather"
                })
            elif temperature >= 38:
                alerts.append({
                    "type": "warning",
                    "message": f"High temperature stress ({temperature}°C). Monitor crops for wilting.",
                    "category": "weather"
                })
            elif temperature >= 34:
                alerts.append({
                    "type": "info",
                    "message": f"Warm day ahead ({temperature}°C). Ensure adequate water availability.",
                    "category": "weather"
                })

        if humidity is not None:
            if humidity >= 85:
                alerts.append({
                    "type": "critical",
                    "message": f"High humidity ({humidity}%) - fungal disease risk imminent. Apply fungicides.",
                    "category": "weather"
                })
            elif humidity < 30:
                alerts.append({
                    "type": "info",
                    "message": f"Low humidity ({humidity}%) - drought risk. Increase irrigation.",
                    "category": "weather"
                })
    except Exception as e:
        logger.error(f"Error generating weather advisories: {e}")
        alerts.append({
            "type": "error",
            "message": "Error processing weather data",
            "category": "weather"
        })

    return alerts


def generate_soil_advisories(
    soil_moisture: Optional[float] = None,
    ph: Optional[float] = None,
    phosphorus: Optional[float] = None,
    alerts: List[Dict] = None
) -> List[Dict]:
    if alerts is None:
        alerts = []

    try:
        if soil_moisture is not None:
            if soil_moisture < 15:
                alerts.append({
                    "type": "critical",
                    "message": f"Severe soil moisture deficit ({soil_moisture}%). Crop stress imminent.",
                    "category": "soil"
                })
            elif soil_moisture < 25:
                alerts.append({
                    "type": "warning",
                    "message": f"Low soil moisture ({soil_moisture}%). Plan irrigation soon.",
                    "category": "soil"
                })

        if ph is not None:
            if ph < 5.5:
                alerts.append({
                    "type": "critical",
                    "message": f"Soil too acidic (pH {ph}). Apply lime to raise pH.",
                    "category": "soil"
                })
            elif ph < 5.8 or ph > 8.0:
                alerts.append({
                    "type": "warning",
                    "message": f"Soil moderately acidic/alkaline (pH {ph}). Consider adjustment.",
                    "category": "soil"
                })
            elif 5.8 <= ph <= 8.0:
                alerts.append({
                    "type": "info",
                    "message": f"Soil pH acceptable (pH {ph}).",
                    "category": "soil"
                })

        if phosphorus is not None and phosphorus > 50:
            alerts.append({
                "type": "warning",
                "message": f"High phosphorus level ({phosphorus} ppm). Avoid excessive P fertilization.",
                "category": "soil"
            })
    except Exception as e:
        logger.error(f"Error generating soil advisories: {e}")
        alerts.append({
            "type": "error",
            "message": "Error processing soil data",
            "category": "soil"
        })

    return alerts


    Returns:
        List of unique, non-duplicate alerts with time and type.
    """
    alerts: list[dict] = []
    now = datetime.now()
def generate_crop_advisories(
    crop: Optional[str] = None,
    temperature: Optional[float] = None,
    soil_moisture: Optional[float] = None,
    alerts: List[Dict] = None
) -> List[Dict]:
    if alerts is None:
        alerts = []

    try:
        if not crop:
            return alerts

        crop_lower = crop.strip().lower()
        if crop_lower not in CROP_THRESHOLDS:
            return alerts

        thresholds = CROP_THRESHOLDS[crop_lower]

        if crop_lower == "rice":
            if soil_moisture is not None and soil_moisture < 30:
                alerts.append({
                    "type": "warning",
                    "message": "Rice water management: Maintain 2-5 cm standing water during tillering.",
                    "category": "crop"
                })
            alerts.append({
                "type": "info",
                "message": "Rice advisory: Monitor for drought stress during critical stages.",
                "category": "crop"
            })

        elif crop_lower == "wheat":
            alerts.append({
                "type": "info",
                "message": "Wheat advisory: First irrigation at crown root initiation (21 days) is critical.",
                "category": "crop"
            })

        elif crop_lower == "cotton":
            alerts.append({
                "type": "info",
                "message": "Cotton advisory: Scout for pests during flowering and boll formation.",
                "category": "crop"
            })
            if phosphorus is not None and phosphorus > 40:
                alerts.append({
                    "type": "warning",
                    "message": "Cotton: Excess nitrogen can increase vegetative growth. Monitor balance.",
                    "category": "crop"
                })

        elif crop_lower == "maize":
            if temperature is not None and temperature >= 38:
                alerts.append({
                    "type": "warning",
                    "message": "Maize: High heat during tasseling/silking causes pollen sterility.",
                    "category": "crop"
                })
            alerts.append({
                "type": "info",
                "message": "Maize: Ensure irrigation at tasseling and silking stages.",
                "category": "crop"
            })

        elif crop_lower == "sugarcane":
            alerts.append({
                "type": "info",
                "message": "Sugarcane: Demands high water and nutrient availability during growth.",
                "category": "crop"
            })
    except Exception as e:
        logger.error(f"Error generating crop advisories: {e}")
        alerts.append({
            "type": "error",
            "message": f"Error processing crop data for {crop}",
            "category": "crop"
        })

    return alerts


def generate_alerts(
    crop: Optional[str] = None,
    temperature: Optional[float] = None,
    irrigation_count: Optional[int] = None,
    water_coverage: Optional[int] = None,
    soil_moisture: Optional[float] = None,
    humidity: Optional[float] = None,
    ph: Optional[float] = None,
    phosphorus: Optional[float] = None,
    season: Optional[str] = None
) -> list:
    now = datetime.now()

    validation = validate_inputs(
        crop=crop, temperature=temperature, soil_moisture=soil_moisture,
        humidity=humidity, ph=ph, phosphorus=phosphorus,
        irrigation_count=irrigation_count, water_coverage=water_coverage, season=season
    )

    if not validation["valid"]:
        logger.warning(f"Invalid input: {validation['errors']}")
        return [{
            "id": 1,
            "type": "error",
            "message": f"Input validation failed: {'; '.join(validation['errors'])}",
            "time": now.isoformat()
        }]

    alerts = []

    crop_lower = crop.strip().lower() if crop else None
    normalized_season = season.strip().lower() if season else None
    current_season = (
        normalized_season
        if normalized_season in {"kharif", "rabi", "zaid"}
        else get_season_from_month(now.month)
    )

    alerts = generate_weather_advisories(temperature=temperature, humidity=humidity, alerts=alerts)
    alerts = generate_soil_advisories(soil_moisture=soil_moisture, ph=ph, phosphorus=phosphorus, alerts=alerts)
    alerts = generate_crop_advisories(crop=crop, temperature=temperature, soil_moisture=soil_moisture, alerts=alerts)

    if water_coverage is not None and water_coverage < 40:
        alerts.append({
            "type": "warning",
            "message": f"Water coverage is only {water_coverage}%. Consider increasing irrigation.",
            "category": "irrigation"
        })

    if irrigation_count is not None and irrigation_count > 6:
        alerts.append({
            "type": "warning",
            "message": f"High irrigation count ({irrigation_count}). Excess may cause waterlogging.",
            "category": "irrigation"
        })

    if current_season == "kharif":
        alerts.append({
            "type": "recommendation",
            "message": "Kharif season active. Ideal: Rice, Maize, Soybean, Cotton. Ensure drainage.",
            "category": "season"
        })
    elif current_season == "rabi":
        alerts.append({
            "type": "recommendation",
            "message": "Rabi season active. Ideal: Wheat, Mustard, Chickpea. Monitor frost.",
            "category": "season"
        })
    elif current_season == "zaid":
        alerts.append({
            "type": "recommendation",
            "message": "Zaid season active. Suitable: Moong, Watermelon, Cucumber. Watch heat.",
            "category": "season"
        })

    for idx, alert in enumerate(alerts, 1):
        alert["id"] = idx
        alert["time"] = now.isoformat()

    logger.info(f"Generated {len(alerts)} advisories for crop={crop}")
    return alerts
