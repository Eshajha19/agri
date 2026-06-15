import re
from datetime import datetime
from typing import Any, Optional


LOW_LEVELS = {"verylow", "very low", "low", "deficient"}
HIGH_LEVELS = {"veryhigh", "very high", "high", "excess"}


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
    try:
        number = float(value)
    except (TypeError, ValueError):
        match = re.match(r"[-+]?\d*\.?\d+", str(value).strip())
        if match:
            try:
                number = float(match.group())
            except (TypeError, ValueError):
                return None
        else:
            return None
    return number if number == number else None


def _as_level(value: Any, low_below: float, high_above: float) -> str:
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


def _add_alert(alerts: list[dict[str, Any]], severity: str, category: str, title: str, message: str, action: str) -> None:
    if any(alert["title"] == title and alert["action"] == action for alert in alerts):
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
            "time": datetime.now().isoformat(),
        }
    )


def generate_advisories(
    weather: Optional[dict[str, Any]] = None,
    soil: Optional[dict[str, Any]] = None,
    crop_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Generate simple, actionable farmer advisories from farm conditions."""
    alerts: list[dict[str, Any]] = []
    weather = weather or {}
    soil = soil or {}
    crop = str(crop_type or "").strip().lower()

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

    if rainfall >= 5 or rain_probability >= 60:
        _add_alert(
            alerts,
            "warning",
            "weather",
            "Rain expected in next 24 hours",
            "Rain is likely soon, so extra irrigation can waste water and increase waterlogging risk.",
            "Avoid irrigation today and keep drainage channels clear.",
        )

    if temperature is not None and temperature >= 38:
        _add_alert(
            alerts,
            "critical",
            "weather",
            "High temperature stress",
            f"Temperature is around {round(temperature)} C, which can stress crops and dry soil quickly.",
            "Water early morning or late evening and use mulch where possible.",
        )
    elif temperature is not None and temperature >= 34:
        _add_alert(
            alerts,
            "warning",
            "weather",
            "Warm day ahead",
            f"Temperature is around {round(temperature)} C, so soil moisture may fall faster than usual.",
            "Check soil moisture and avoid spraying during peak afternoon heat.",
        )

    moisture = _as_number(soil.get("moisture") or soil.get("soil_moisture"))
    if moisture is not None and moisture < 25 and rainfall < 5 and rain_probability < 60:
        _add_alert(
            alerts,
            "warning",
            "soil",
            "Low soil moisture",
            f"Soil moisture is about {round(moisture)}%, which can reduce crop growth.",
            "Irrigate lightly and recheck the field after water settles.",
        )

    nitrogen_level = _as_level(soil.get("nitrogen"), low_below=140, high_above=360)
    phosphorus_level = _as_level(soil.get("phosphorus"), low_below=10, high_above=40)
    potassium_level = _as_level(soil.get("potassium"), low_below=110, high_above=420)

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
            "info",
            "soil",
            "Nitrogen is high",
            "Excess nitrogen can increase pest pressure and weak growth.",
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
    if ph is not None and ph < 5.8:
        _add_alert(
            alerts,
            "info",
            "soil",
            "Soil is acidic",
            f"Soil pH is {ph:g}, which can reduce nutrient availability.",
            "Discuss lime application with a local agriculture officer before the next sowing.",
        )
    elif ph is not None and ph > 7.8:
        _add_alert(
            alerts,
            "info",
            "soil",
            "Soil is alkaline",
            f"Soil pH is {ph:g}, which can limit nutrient uptake.",
            "Add organic matter and consider gypsum based on a soil test recommendation.",
        )

    if crop in {"rice", "paddy"}:
        _add_alert(
            alerts,
            "info",
            "crop",
            "Rice water management",
            "Rice performs best when water is managed carefully during active growth.",
            "Maintain shallow standing water, but drain excess water after heavy rain.",
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
            "Scout leaves twice a week and avoid excess nitrogen.",
        )
    elif crop == "maize":
        _add_alert(
            alerts,
            "info",
            "crop",
            "Maize critical stages",
            "Maize is sensitive to stress during tasseling and silking.",
            "Keep moisture steady and top dress nitrogen before tasseling if soil nitrogen is low.",
        )

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

    return alerts
