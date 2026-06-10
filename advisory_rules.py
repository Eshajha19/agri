from datetime import datetime, timezone
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

LOW_LEVELS = {"verylow", "very low", "low", "deficient"}
HIGH_LEVELS = {"veryhigh", "very high", "high", "excess"}

CROP_THRESHOLDS = {
    "rice": {"temp_ideal": (20, 30), "moisture_min": 50, "ph_range": (6.0, 7.5)},
    "paddy": {"temp_ideal": (20, 30), "moisture_min": 50, "ph_range": (6.0, 7.5)},
    "wheat": {"temp_ideal": (10, 25), "moisture_min": 30, "ph_range": (6.5, 7.5)},
    "cotton": {"temp_ideal": (21, 27), "moisture_min": 35, "ph_range": (6.0, 7.5)},
    "maize": {"temp_ideal": (18, 26), "moisture_min": 40, "ph_range": (6.0, 7.0)},
}

NUTRIENT_THRESHOLDS = {
    "nitrogen": {"low": 140, "high": 360},
    "phosphorus": {"low": 10, "high": 40},
    "potassium": {"low": 110, "high": 420},
}


def _as_number(value: Any) -> Optional[float]:
    """Convert value to float, handling None and non-numeric types safely."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def _as_level(value: Any, low_below: float, high_above: float) -> str:
    """Classify nutrient levels into low/ok/high categories."""
    number = _as_number(value)
    if number is not None:
        if number < low_below:
            return "low"
        if number > high_above:
            return "high"
        return "ok"

    label = str(value or "").strip().lower()
    if label in LOW_LEVELS:
        return "low"
    if label in HIGH_LEVELS:
        return "high"
    return "ok"


def _validate_input_data(weather: dict, soil: dict, crop_type: str) -> tuple[bool, str]:
    """
    Validate input data for advisory generation.
    Returns (is_valid, error_message).
    """
    if weather is not None and not isinstance(weather, dict):
        return False, "Weather data must be a dictionary"

    if soil is not None and not isinstance(soil, dict):
        return False, "Soil data must be a dictionary"

    if crop_type is not None and not isinstance(crop_type, str):
        return False, "Crop type must be a string"

    return True, ""


def _add_alert(
    alerts: list[dict[str, Any]],
    severity: str,
    category: str,
    title: str,
    message: str,
    action: str,
) -> None:
    """
    Add alert to advisory list if not duplicate.
    Prevents duplicate alerts with same title and action.
    """
    if any(alert["title"] == title and alert["action"] == action for alert in alerts):
        logger.debug("Skipping duplicate alert: %s", title)
        return

    alerts.append(
        {
            "id": len(alerts) + 1,
            "severity": severity,
            "type": severity,
            "category": category,
            "title": title,
            "message": message,
            "action": action,
            "source": "rule-based",
            "time": datetime.now(timezone.utc).isoformat(),
        }
    )
    logger.debug("Added advisory: %s (severity: %s)", title, severity)


def _generate_weather_advisories(alerts: list[dict[str, Any]], weather: dict) -> None:
    """Generate weather-related advisories."""
    temperature = (
        _as_number(weather.get("temperature"))
        or _as_number(weather.get("temperature_c"))
        or _as_number(weather.get("max_temperature"))
        or _as_number(weather.get("max_temperature_c"))
    )
    rainfall = (
        _as_number(weather.get("rainfall_next_24h"))
        or _as_number(weather.get("precipitation_next_24h"))
        or _as_number(weather.get("rainfall"))
        or 0
    )
    rain_probability = _as_number(weather.get("rain_probability")) or 0
    humidity = _as_number(weather.get("humidity")) or _as_number(weather.get("relative_humidity"))

    if rainfall >= 10 or rain_probability >= 70:
        _add_alert(
            alerts,
            "critical",
            "weather",
            "Heavy rain expected",
            "High rainfall probability may cause waterlogging and disease spread.",
            "Ensure field drainage is clear and avoid pesticide applications.",
        )
    elif rainfall >= 5 or rain_probability >= 60:
        _add_alert(
            alerts,
            "warning",
            "weather",
            "Rain expected in next 24 hours",
            "Rain is likely soon, so extra irrigation can waste water and increase waterlogging risk.",
            "Avoid irrigation today and keep drainage channels clear.",
        )

    if temperature is not None:
        if temperature >= 40:
            _add_alert(
                alerts,
                "critical",
                "weather",
                "Extreme heat stress",
                f"Temperature is around {round(temperature)}°C, which causes severe crop stress.",
                "Ensure irrigation is available and consider shade management for sensitive crops.",
            )
        elif temperature >= 38:
            _add_alert(
                alerts,
                "critical",
                "weather",
                "High temperature stress",
                f"Temperature is around {round(temperature)}°C, which can stress crops and dry soil quickly.",
                "Water early morning or late evening and use mulch where possible.",
            )
        elif temperature >= 34:
            _add_alert(
                alerts,
                "warning",
                "weather",
                "Warm day ahead",
                f"Temperature is around {round(temperature)}°C, so soil moisture may fall faster than usual.",
                "Check soil moisture and avoid spraying during peak afternoon heat.",
            )

    if humidity is not None:
        if humidity >= 85:
            _add_alert(
                alerts,
                "warning",
                "weather",
                "High humidity - disease risk",
                f"Humidity is {round(humidity)}%, creating favorable conditions for fungal diseases.",
                "Scout for disease symptoms and consider preventive spraying if needed.",
            )
        elif humidity < 30:
            _add_alert(
                alerts,
                "info",
                "weather",
                "Low humidity - drought risk",
                f"Humidity is {round(humidity)}%, indicating low moisture in the air.",
                "Monitor soil moisture closely and increase irrigation frequency if needed.",
            )


def _generate_soil_advisories(alerts: list[dict[str, Any]], soil: dict) -> None:
    """Generate soil-related advisories."""
    moisture = _as_number(soil.get("moisture") or soil.get("soil_moisture"))
    if moisture is not None:
        if moisture < 15:
            _add_alert(
                alerts,
                "critical",
                "soil",
                "Critical soil moisture",
                f"Soil moisture is about {round(moisture)}%, which severely limits crop growth.",
                "Irrigate immediately and increase irrigation frequency.",
            )
        elif moisture < 25:
            _add_alert(
                alerts,
                "warning",
                "soil",
                "Low soil moisture",
                f"Soil moisture is about {round(moisture)}%, which can reduce crop growth.",
                "Irrigate lightly and recheck the field after water settles.",
            )

    nitrogen_level = _as_level(soil.get("nitrogen"), **NUTRIENT_THRESHOLDS["nitrogen"])
    phosphorus_level = _as_level(soil.get("phosphorus"), **NUTRIENT_THRESHOLDS["phosphorus"])
    potassium_level = _as_level(soil.get("potassium"), **NUTRIENT_THRESHOLDS["potassium"])

    if nitrogen_level == "low":
        _add_alert(
            alerts,
            "warning",
            "soil",
            "Low nitrogen detected",
            "Nitrogen is below the preferred range for strong vegetative growth.",
            "Apply nitrogen fertilizer in split doses and irrigate after application if rain is not expected.",
        )
    elif nitrogen_level == "high":
        _add_alert(
            alerts,
            "warning",
            "soil",
            "Nitrogen is high",
            "Excess nitrogen can increase pest pressure and weak stem growth.",
            "Avoid additional urea for now and balance with potassium and organic matter.",
        )

    if phosphorus_level == "low":
        _add_alert(
            alerts,
            "info",
            "soil",
            "Low phosphorus detected",
            "Phosphorus supports root growth, flowering, and early crop establishment.",
            "Use DAP or single super phosphate near the root zone as per local dosage guidance.",
        )
    elif phosphorus_level == "high":
        _add_alert(
            alerts,
            "info",
            "soil",
            "Phosphorus is high",
            "Excess phosphorus can reduce zinc and iron availability.",
            "Avoid additional phosphate fertilizers and monitor micronutrient status.",
        )

    if potassium_level == "low":
        _add_alert(
            alerts,
            "info",
            "soil",
            "Low potassium detected",
            "Potassium helps crops handle heat, drought, and disease pressure.",
            "Apply potash fertilizer or composted crop residue after checking crop stage.",
        )

    ph = _as_number(soil.get("ph") or soil.get("soil_ph"))
    if ph is not None:
        if ph < 5.5:
            _add_alert(
                alerts,
                "warning",
                "soil",
                "Soil is too acidic",
                f"Soil pH is {ph:g}, which severely reduces nutrient availability.",
                "Apply lime urgently and discuss with a local agriculture officer.",
            )
        elif ph < 5.8:
            _add_alert(
                alerts,
                "info",
                "soil",
                "Soil is acidic",
                f"Soil pH is {ph:g}, which can reduce nutrient availability.",
                "Discuss lime application with a local agriculture officer before the next sowing.",
            )
        elif ph > 8.0:
            _add_alert(
                alerts,
                "warning",
                "soil",
                "Soil is too alkaline",
                f"Soil pH is {ph:g}, which severely limits nutrient uptake.",
                "Add sulfur or acidifying organic matter and consider professional soil amendment.",
            )
        elif ph > 7.8:
            _add_alert(
                alerts,
                "info",
                "soil",
                "Soil is alkaline",
                f"Soil pH is {ph:g}, which can limit nutrient uptake.",
                "Add organic matter and consider gypsum based on a soil test recommendation.",
            )


def _generate_crop_advisories(alerts: list[dict[str, Any]], crop: str, weather: dict, soil: dict) -> None:
    """Generate crop-specific advisories based on crop type and conditions."""
    if crop in {"rice", "paddy"}:
        _add_alert(
            alerts,
            "info",
            "crop",
            "Rice water management",
            "Rice performs best when water is managed carefully during active growth.",
            "Maintain shallow standing water (2-5cm), but drain excess water after heavy rain.",
        )
        if _as_number(soil.get("moisture")) and _as_number(soil.get("moisture")) < 40:
            _add_alert(
                alerts,
                "warning",
                "crop",
                "Rice drought stress",
                "Rice requires consistent water availability for optimal growth.",
                "Ensure adequate irrigation supply and monitor field water levels daily.",
            )

    elif crop == "wheat":
        _add_alert(
            alerts,
            "info",
            "crop",
            "Wheat irrigation timing",
            "Wheat yield depends strongly on timely irrigation at key growth stages.",
            "Prioritize irrigation at crown root, tillering, and grain filling stages.",
        )

    elif crop == "cotton":
        _add_alert(
            alerts,
            "info",
            "crop",
            "Cotton pest scouting",
            "Cotton can attract sucking pests, especially after humid or rainy weather.",
            "Scout leaves twice a week, avoid excess nitrogen, and maintain field hygiene.",
        )

    elif crop == "maize":
        _add_alert(
            alerts,
            "info",
            "crop",
            "Maize critical stages",
            "Maize is sensitive to stress during tasseling and silking (V8-R1 stages).",
            "Keep moisture steady and top dress nitrogen before tasseling if soil nitrogen is low.",
        )

    elif crop == "sugarcane":
        _add_alert(
            alerts,
            "info",
            "crop",
            "Sugarcane nutrient demand",
            "Sugarcane is a heavy feeder requiring consistent nutrient availability.",
            "Split fertilizer applications and ensure regular irrigation during growth season.",
        )


def generate_advisories(
    weather: Optional[dict[str, Any]] = None,
    soil: Optional[dict[str, Any]] = None,
    crop_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Generate actionable farmer advisories from farm conditions.

    Validates inputs, generates weather/soil/crop advisories, and returns
    a prioritized list of recommendations.

    Args:
        weather: Dictionary with weather data (temperature, rainfall, humidity, etc.)
        soil: Dictionary with soil data (moisture, NPK, pH, etc.)
        crop_type: String identifier for the crop being grown

    Returns:
        List of advisory dictionaries with severity, title, message, and action.
    """
    alerts: list[dict[str, Any]] = []

    is_valid, error_msg = _validate_input_data(weather, soil, crop_type)
    if not is_valid:
        logger.error("Invalid input data: %s", error_msg)
        return [{
            "id": 1,
            "severity": "error",
            "type": "error",
            "category": "general",
            "title": "Invalid advisory data",
            "message": error_msg,
            "action": "Check input parameters",
            "source": "validation",
            "time": datetime.now(timezone.utc).isoformat(),
        }]

    weather = weather or {}
    soil = soil or {}
    crop = str(crop_type or "").strip().lower()

    try:
        _generate_weather_advisories(alerts, weather)
    except Exception as exc:
        logger.error("Error generating weather advisories: %s", exc, exc_info=True)

    try:
        _generate_soil_advisories(alerts, soil)
    except Exception as exc:
        logger.error("Error generating soil advisories: %s", exc, exc_info=True)

    try:
        if crop:
            _generate_crop_advisories(alerts, crop, weather, soil)
    except Exception as exc:
        logger.error("Error generating crop advisories: %s", exc, exc_info=True)

    if not weather:
        _add_alert(
            alerts,
            "info",
            "weather",
            "Add weather data",
            "Live weather improves irrigation, spraying, and heat-stress advice.",
            "Open the Weather page once so the dashboard can use your latest local forecast.",
        )

    if not soil:
        _add_alert(
            alerts,
            "info",
            "soil",
            "Add soil readings",
            "Soil nutrients help generate fertilizer and pH correction actions.",
            "Run Soil Analysis or add recent NPK values for more precise advisories.",
        )

    if not alerts:
        _add_alert(
            alerts,
            "success",
            "general",
            "Conditions look stable",
            "No urgent weather or soil risk was detected from the submitted data.",
            "Continue regular field scouting and update readings after major weather changes.",
        )

    logger.info("Generated %d advisories for crop=%s", len(alerts), crop or "unknown")
    return alerts
