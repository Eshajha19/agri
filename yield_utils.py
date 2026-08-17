def predict_yield_rule_based(data: dict) -> float:
    crop = str(data.get("Crop") or data.get("crop") or "").strip()
    season = str(data.get("Season") or data.get("season") or "").strip()
    crop_height = int(data.get("CHeight") or data.get("cHeight") or data.get("cropHeight") or 0)
    transplanting_method = str(data.get("CTransp") or data.get("cTransp") or data.get("transplantingMethod") or "").strip()
    irrigation_type = str(data.get("IrriType") or data.get("irriType") or data.get("irrigationType") or "").strip()
    irrigation_source = str(data.get("IrriSource") or data.get("irriSource") or data.get("irrigationSource") or "").strip()
    irrigation_count = int(data.get("IrriCount") or data.get("irriCount") or data.get("irrigationCount") or 0)
    water_coverage = int(data.get("WaterCov") or data.get("waterCov") or data.get("waterCoverage") or 0)

    yield_value = 18.0

    crop_adjustment = {
        "Paddy": 0,
        "Cotton": -2,
        "Maize": 2,
        "Bengal Gram": -3,
        "Groundnut": -1,
        "Chillies": -2,
        "Red Gram": -3,
    }

    yield_value += crop_adjustment.get(crop, 0)

    if season == "Kharif":
        yield_value += 1
    elif season == "Rabi":
        yield_value += 0.5

    if crop_height >= 100:
        yield_value += 1.5
    elif crop_height >= 80:
        yield_value += 1
    elif crop_height >= 60:
        yield_value += 0.5
    else:
        yield_value -= 1

    if transplanting_method == "Transplanting":
        yield_value += 1

    irrigation_adjustment = {
        "Flood": 1,
        "Sprinkler": 1.5,
        "Drip": 2,
        "Surface": 0.5,
    }

    yield_value += irrigation_adjustment.get(irrigation_type, 0)

    source_adjustment = {
        "Groundwater": 0.5,
        "Canal": 1,
        "Rainfed": -2,
        "Well": 0.5,
        "Tubewell": 1,
    }

    yield_value += source_adjustment.get(irrigation_source, 0)

    if irrigation_count >= 8:
        yield_value += 0.8
    elif irrigation_count >= 5:
        yield_value += 0.4
    elif irrigation_count < 3:
        yield_value -= 1

    if water_coverage >= 85:
        yield_value += 1
    elif water_coverage >= 70:
        yield_value += 0.5
    elif water_coverage < 50:
        yield_value -= 2

    yield_value = max(5.0, min(yield_value, 40.0))

    return round(float(yield_value), 2)
