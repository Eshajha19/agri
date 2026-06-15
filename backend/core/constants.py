"""
Shared configuration constants for the Fasal Saathi application.
"""

# Consolidated crop thresholds for advisory and alert engines
CROP_THRESHOLDS = {
    "rice": {
        "temp_min": 20,
        "temp_max": 30,
        "moisture_min": 50,
        "ph_min": 6.0,
        "ph_max": 7.5,
        "temp_ideal": (20, 30),
        "ph_range": (6.0, 7.5),
        "critical_stages": ["tillering", "flowering", "grain-filling"]
    },
    "paddy": {
        "temp_min": 20,
        "temp_max": 30,
        "moisture_min": 50,
        "ph_min": 6.0,
        "ph_max": 7.5,
        "temp_ideal": (20, 30),
        "ph_range": (6.0, 7.5),
        "critical_stages": ["tillering", "flowering", "grain-filling"]
    },
    "wheat": {
        "temp_min": 10,
        "temp_max": 25,
        "moisture_min": 30,
        "ph_min": 6.5,
        "ph_max": 7.5,
        "temp_ideal": (10, 25),
        "ph_range": (6.5, 7.5),
        "critical_stages": ["crown-root-initiation", "tillering", "flowering"]
    },
    "cotton": {
        "temp_min": 21,
        "temp_max": 27,
        "moisture_min": 35,
        "ph_min": 6.0,
        "ph_max": 7.5,
        "temp_ideal": (21, 27),
        "ph_range": (6.0, 7.5),
        "critical_stages": ["flowering", "boll-formation"]
    },
    "maize": {
        "temp_min": 18,
        "temp_max": 26,
        "moisture_min": 40,
        "ph_min": 6.0,
        "ph_max": 7.0,
        "temp_ideal": (18, 26),
        "ph_range": (6.0, 7.0),
        "critical_stages": ["tasseling", "silking", "grain-filling"]
    },
    "sugarcane": {
        "temp_min": 20,
        "temp_max": 28,
        "moisture_min": 45,
        "ph_min": 6.5,
        "ph_max": 8.0,
        "temp_ideal": (20, 28),
        "ph_range": (6.5, 8.0),
        "critical_stages": ["sprouting", "grand-growth-phase"]
    }
}
