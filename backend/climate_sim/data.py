"""Static climate simulation datasets and alias mappings."""

# (base_temp_C, base_rain_mm_per_month)
REGIONAL_SEASONAL_BASELINES = {
    # region        kharif          rabi           zaid
    "northwest": {"kharif": (32, 120), "rabi": (15, 20), "zaid": (38, 15)},
    "northeast": {"kharif": (30, 350), "rabi": (18, 40), "zaid": (34, 80)},
    "central": {"kharif": (30, 200), "rabi": (20, 25), "zaid": (36, 20)},
    "west": {"kharif": (30, 180), "rabi": (22, 10), "zaid": (37, 10)},
    "south": {"kharif": (28, 160), "rabi": (24, 60), "zaid": (34, 30)},
    "southwest": {"kharif": (27, 600), "rabi": (26, 120), "zaid": (32, 80)},
    "east": {"kharif": (30, 280), "rabi": (19, 30), "zaid": (35, 40)},
}

# Crop-specific sensitivity coefficients
# temp_coeff  : fractional yield change per +1C above baseline
# rain_coeff  : fractional yield change per +10 mm/month above baseline
# opt_temp    : optimal temperature range (min, max) in C
# opt_rain    : optimal monthly rainfall range (min, max) in mm
CROP_PROFILES = {
    "wheat": {"temp_coeff": -0.06, "rain_coeff": 0.03, "opt_temp": (15, 25), "opt_rain": (40, 80)},
    "rice": {"temp_coeff": -0.05, "rain_coeff": 0.02, "opt_temp": (25, 35), "opt_rain": (150, 300)},
    "maize": {"temp_coeff": -0.07, "rain_coeff": 0.04, "opt_temp": (20, 30), "opt_rain": (80, 150)},
    "cotton": {"temp_coeff": -0.03, "rain_coeff": 0.01, "opt_temp": (25, 35), "opt_rain": (60, 120)},
    "sugarcane": {"temp_coeff": -0.02, "rain_coeff": 0.05, "opt_temp": (25, 35), "opt_rain": (150, 250)},
    "soybean": {"temp_coeff": -0.04, "rain_coeff": 0.03, "opt_temp": (20, 30), "opt_rain": (80, 150)},
    "potato": {"temp_coeff": -0.05, "rain_coeff": 0.04, "opt_temp": (15, 25), "opt_rain": (60, 100)},
    "groundnut": {"temp_coeff": -0.04, "rain_coeff": 0.02, "opt_temp": (25, 35), "opt_rain": (60, 120)},
    "mustard": {"temp_coeff": -0.05, "rain_coeff": 0.02, "opt_temp": (10, 25), "opt_rain": (30, 60)},
    "chickpea": {"temp_coeff": -0.06, "rain_coeff": 0.02, "opt_temp": (15, 25), "opt_rain": (30, 60)},
    "tomato": {"temp_coeff": -0.05, "rain_coeff": 0.03, "opt_temp": (20, 30), "opt_rain": (80, 120)},
    "onion": {"temp_coeff": -0.04, "rain_coeff": 0.02, "opt_temp": (15, 25), "opt_rain": (50, 80)},
    "default": {"temp_coeff": -0.04, "rain_coeff": 0.02, "opt_temp": (20, 30), "opt_rain": (80, 150)},
}

# Region aliases so common user inputs map to canonical keys
REGION_ALIASES = {
    "punjab": "northwest",
    "haryana": "northwest",
    "rajasthan": "northwest",
    "up": "northeast",
    "uttar pradesh": "northeast",
    "bihar": "northeast",
    "west bengal": "northeast",
    "assam": "northeast",
    "mp": "central",
    "madhya pradesh": "central",
    "chhattisgarh": "central",
    "vidarbha": "central",
    "maharashtra": "central",
    "gujarat": "west",
    "karnataka": "south",
    "andhra pradesh": "south",
    "telangana": "south",
    "kerala": "southwest",
    "odisha": "east",
    "jharkhand": "east",
}

VALID_SEASONS = {"kharif", "rabi", "zaid"}
