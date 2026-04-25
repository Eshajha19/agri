from datetime import datetime


def get_season_from_month(month: int) -> str:
    if month in [6, 7, 8, 9]:
        return "kharif"
    elif month in [11, 12, 1, 2]:
        return "rabi"
    return "zaid"


def generate_alerts(
    crop: str = None,
    irrigation_count: int = None,
    water_coverage: int = None,
    season: str = None
) -> list:
    alerts = []
    now = datetime.now()
    current_season = season or get_season_from_month(now.month)

    # Rule 1 — low water coverage
    if water_coverage is not None and water_coverage < 40:
        alerts.append({
            "id": len(alerts) + 1,
            "type": "warning",
            "message": (
                f"⚠️ Water coverage is only {water_coverage}%. "
                "Consider increasing irrigation to avoid crop stress."
            ),
            "time": now.isoformat()
        })

    # Rule 2 — over irrigation
    if irrigation_count is not None and irrigation_count > 6:
        alerts.append({
            "id": len(alerts) + 1,
            "type": "warning",
            "message": (
                f"⚠️ High irrigation count ({irrigation_count}). "
                "Excess irrigation may cause waterlogging and root damage."
            ),
            "time": now.isoformat()
        })

    # Rule 3 — seasonal advisory
    if current_season == "kharif":
        alerts.append({
            "id": len(alerts) + 1,
            "type": "recommendation",
            "message": (
                "🌱 Kharif season active. Ideal crops: Rice, Maize, "
                "Soybean, Cotton. Ensure adequate drainage."
            ),
            "time": now.isoformat()
        })
    elif current_season == "rabi":
        alerts.append({
            "id": len(alerts) + 1,
            "type": "recommendation",
            "message": (
                "🌾 Rabi season active. Ideal crops: Wheat, Mustard, "
                "Chickpea. Monitor for frost risk."
            ),
            "time": now.isoformat()
        })
    elif current_season == "zaid":
        alerts.append({
            "id": len(alerts) + 1,
            "type": "recommendation",
            "message": (
                "☀️ Zaid season active. Suitable for Moong, Watermelon, "
                "Cucumber. Watch for heat stress."
            ),
            "time": now.isoformat()
        })

    # Rule 4 — crop-specific
    if crop:
        crop_lower = crop.lower()
        if crop_lower == "rice":
            alerts.append({
                "id": len(alerts) + 1,
                "type": "info",
                "message": (
                    "🌾 Rice advisory: Maintain 2–5 cm standing water "
                    "during tillering stage for best yield."
                ),
                "time": now.isoformat()
            })
        elif crop_lower == "wheat":
            alerts.append({
                "id": len(alerts) + 1,
                "type": "info",
                "message": (
                    "🌿 Wheat advisory: First irrigation at crown root "
                    "initiation (21 days after sowing) is critical."
                ),
                "time": now.isoformat()
            })
        elif crop_lower == "maize":
            alerts.append({
                "id": len(alerts) + 1,
                "type": "info",
                "message": (
                    "🌽 Maize advisory: Ensure irrigation at tasseling "
                    "and silking stages to prevent yield loss."
                ),
                "time": now.isoformat()
            })

    # Rule 5 — fallback
    if not alerts:
        alerts.append({
            "id": 1,
            "type": "info",
            "message": "✅ No critical advisories. Crop conditions look stable.",
            "time": now.isoformat()
        })

    return alerts