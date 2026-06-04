from datetime import datetime, timezone
from typing import Optional
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

SEASON_MONTHS = {
    "kharif": {6, 7, 8, 9},
    "rabi": {11, 12, 1, 2},
    "zaid": {3, 4, 5, 10}
}

CROP_SEASON_MAP = {
    "rice": {"kharif", "rabi"},
    "maize": {"kharif", "zaid"},
    "soybean": {"kharif"},
    "cotton": {"kharif"},
    "wheat": {"rabi"},
    "mustard": {"rabi"},
    "chickpea": {"rabi"},
    "moong": {"zaid"},
    "watermelon": {"zaid"},
    "cucumber": {"zaid"},
}

CROP_ADVISORIES = {
    "rice": {
        "message": "Rice advisory: Maintain 2-5 cm standing water during tillering stage for best yield.",
        "critical_stages": ["tillering", "panicle initiation", "flowering"]
    },
    "wheat": {
        "message": "Wheat advisory: First irrigation at crown root initiation (21 days after sowing) is critical.",
        "critical_stages": ["crown root", "tillering", "grain filling"]
    },
    "maize": {
        "message": "Maize advisory: Ensure irrigation at tasseling and silking stages to prevent yield loss.",
        "critical_stages": ["tasseling", "silking", "grain fill"]
    },
}

SEASON_ADVISORIES = {
    "kharif": "Kharif season active. Ideal crops: Rice, Maize, Soybean, Cotton. Ensure adequate drainage.",
    "rabi": "Rabi season active. Ideal crops: Wheat, Mustard, Chickpea. Monitor for frost risk.",
    "zaid": "Zaid season active. Suitable for Moong, Watermelon, Cucumber. Watch for heat stress.",
}


@lru_cache(maxsize=12)
def get_season_from_month(month: int) -> str:
    """Get agricultural season from month number (cached)."""
    if month < 1 or month > 12:
        raise ValueError(f"Invalid month: {month}")
    for season, months in SEASON_MONTHS.items():
        if month in months:
            return season
    raise ValueError(f"No season found for month: {month}")


def _is_duplicate_alert(alerts: list[dict], new_type: str, new_message: str) -> bool:
    """Check if alert already exists to prevent duplicates."""
    return any(
        alert.get("type") == new_type and alert.get("message") == new_message
        for alert in alerts
    )


def _add_alert(
    alerts: list[dict],
    alert_type: str,
    message: str,
    now: datetime,
    severity: Optional[str] = None
) -> bool:
    """Add alert if not duplicate. Returns True if added."""
    if _is_duplicate_alert(alerts, alert_type, message):
        logger.debug("Skipped duplicate alert: %s", message[:50])
        return False

    alert_id = len(alerts) + 1
    alert = {
        "id": alert_id,
        "type": alert_type,
        "message": message,
        "time": now.isoformat(),
    }
    if severity:
        alert["severity"] = severity

    alerts.append(alert)
    logger.debug("Added alert #%d: %s (type=%s)", alert_id, message[:50], alert_type)
    return True


def _generate_water_alerts(
    alerts: list[dict],
    irrigation_count: Optional[int],
    water_coverage: Optional[int],
    now: datetime
) -> None:
    """Generate water management related alerts."""
    if water_coverage is not None:
        if water_coverage < 25:
            _add_alert(
                alerts,
                "critical",
                f"Water coverage is critically low at {water_coverage}%. Urgent irrigation needed to prevent crop failure.",
                now,
                "critical"
            )
        elif water_coverage < 40:
            _add_alert(
                alerts,
                "warning",
                f"Water coverage is only {water_coverage}%. Consider increasing irrigation to avoid crop stress.",
                now,
                "warning"
            )

    if irrigation_count is not None:
        if irrigation_count > 8:
            _add_alert(
                alerts,
                "critical",
                f"Excessive irrigation count ({irrigation_count}). Risk of waterlogging and soil degradation.",
                now,
                "critical"
            )
        elif irrigation_count > 6:
            _add_alert(
                alerts,
                "warning",
                f"High irrigation count ({irrigation_count}). Excess irrigation may cause waterlogging.",
                now,
                "warning"
            )
        elif irrigation_count < 2:
            _add_alert(
                alerts,
                "warning",
                f"Very low irrigation count ({irrigation_count}). Consider increasing frequency based on season.",
                now,
                "warning"
            )


def _generate_season_alerts(
    alerts: list[dict],
    current_season: str,
    now: datetime
) -> None:
    """Generate season-specific recommendations."""
    if current_season in SEASON_ADVISORIES:
        _add_alert(
            alerts,
            "recommendation",
            SEASON_ADVISORIES[current_season],
            now
        )


def _generate_crop_alerts(
    alerts: list[dict],
    crop: Optional[str],
    current_season: str,
    now: datetime
) -> None:
    """Generate crop-specific advisories."""
    if not crop:
        return

    if crop in CROP_ADVISORIES:
        _add_alert(
            alerts,
            "info",
            CROP_ADVISORIES[crop]["message"],
            now
        )

    if crop in CROP_SEASON_MAP:
        suitable_seasons = CROP_SEASON_MAP[crop]
        if current_season not in suitable_seasons:
            logger.warning(
                "Crop %s not ideal for season %s. Suitable: %s",
                crop, current_season, suitable_seasons
            )


def generate_alerts(
    crop: Optional[str] = None,
    irrigation_count: Optional[int] = None,
    water_coverage: Optional[int] = None,
    season: Optional[str] = None
) -> list[dict]:
    """
    Generate deduped, actionable agricultural alerts.

    Implements deduplication to prevent duplicate alerts, categorizes
    by severity, and provides crop/season-specific recommendations.

    Args:
        crop: Crop type (rice, wheat, maize, etc.)
        irrigation_count: Number of irrigations applied
        water_coverage: Percentage of field irrigated
        season: Season name (kharif, rabi, zaid) or auto-detected

    Returns:
        List of unique, non-duplicate alerts with time and type.
    """
    alerts: list[dict] = []
    now = datetime.now(timezone.utc)

    try:
        crop = crop.strip().lower() if crop else None
        normalized_season = season.strip().lower() if season else None

        if normalized_season not in SEASON_MONTHS:
            current_season = get_season_from_month(now.month)
        else:
            current_season = normalized_season
    except (ValueError, AttributeError) as exc:
        logger.error("Error normalizing inputs: %s", exc)
        current_season = get_season_from_month(now.month)

    try:
        _generate_water_alerts(alerts, irrigation_count, water_coverage, now)
    except Exception as exc:
        logger.error("Error generating water alerts: %s", exc, exc_info=True)

    try:
        _generate_season_alerts(alerts, current_season, now)
    except Exception as exc:
        logger.error("Error generating season alerts: %s", exc, exc_info=True)

    try:
        _generate_crop_alerts(alerts, crop, current_season, now)
    except Exception as exc:
        logger.error("Error generating crop alerts: %s", exc, exc_info=True)

    logger.info("Generated %d unique alerts for crop=%s, season=%s", len(alerts), crop or "unknown", current_season)
    return alerts
