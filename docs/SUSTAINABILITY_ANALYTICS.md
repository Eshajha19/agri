# Crop Sustainability Analytics

LCA-style per-season **water footprint** and **carbon emission** estimates for the Advisor module.

## API

### `POST /api/sustainability/analyze`

Request body (JSON):

| Field | Type | Description |
|-------|------|-------------|
| `crop_type` | string | e.g. Rice, Wheat |
| `season` | string | Kharif, Rabi, Zaid |
| `acreage` | number | Farm area in acres |
| `irrigation_type` | string | `drip`, `sprinkler`, `flood`, `rainfed` |
| `irrigation_events` | int | Irrigation cycles per season |
| `fertilizer_n_kg` | float? | Optional; defaults from crop coefficients |
| `fertilizer_p_kg` | float? | Optional |
| `fertilizer_k_kg` | float? | Optional |
| `machinery_hours` | float? | Optional |
| `diesel_liters` | float? | Optional |
| `organic_practices` | bool | Applies emission reduction factor |
| `user_id` | string? | For server-side history |

Response `data` includes `water_footprint_m3`, `carbon_emissions_kg_co2e`, `sustainability_score`, `breakdown`, `comparison_chart`, `recommendations`.

### `GET /api/sustainability/history?user_id=&limit=`

Returns prior analyses for trend charts.

### `GET /api/sustainability/formulas`

Returns configurable emission factors and crop coefficients (`sustainability_analytics.py`).

## Frontend

Open **Advisor → Sustainability Analytics** card. Uses `frontend/SustainabilityAnalytics.jsx` and proxies `/api` via Vite in development.

Local history is also stored under `agri:sustainabilityHistory:{userId}` in `localStorage`.

## Notes

Figures are **advisory estimates**, not certified carbon credits or ISO-compliant LCAs. Coefficients can be tuned in `CROP_COEFFICIENTS` and `EMISSION_FACTORS` in `sustainability_analytics.py`.
