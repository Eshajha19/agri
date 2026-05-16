# ML Input Validation and Sanitization

## Overview

This document describes the input validation and sanitization system implemented for ML prediction endpoints. The system prevents internal server errors, nonsensical predictions, and security vulnerabilities by validating all numeric inputs against agricultural constraints.

## Problem Statement

**Original Issue**: ML prediction endpoints accepted parameters like Nitrogen (N), Phosphorus (P), and pH directly from requests without validation. This caused:

1. **Internal Server Errors**: Passing strings or extreme values into Scikit-Learn models caused crashes
2. **Nonsensical Predictions**: Out-of-range values (e.g., pH = 150) produced meaningless results
3. **Type Errors**: Non-numeric strings caused runtime exceptions

## Solution

A comprehensive validation layer that:
- Validates numeric types and ranges before model inference
- Provides clear, actionable error messages
- Follows agricultural best practices for parameter ranges
- Maintains backward compatibility with existing API contracts

## Architecture

### Components

1. **`ml/validators.py`**: Core validation logic
   - `validate_numeric_input()`: Single-field validation
   - `validate_ml_inputs()`: Full dictionary validation
   - `InputValidationError`: Custom exception with field metadata
   - `NUMERIC_CONSTRAINTS`: Agricultural parameter ranges

2. **`ml/preprocessing.py`**: Integration point
   - Calls `validate_ml_inputs()` before DataFrame creation
   - Raises `InputValidationError` on invalid inputs

3. **`main.py`**: API error handling
   - Catches `InputValidationError` and returns HTTP 422
   - Provides structured error responses with field details

### Validation Flow

```
API Request
    ↓
Pydantic Model (basic type validation)
    ↓
validate_ml_inputs() (range validation)
    ↓
FeaturePreprocessor.preprocess() (encoding)
    ↓
Model.predict()
    ↓
API Response
```

## Validated Parameters

### Soil Nutrients (kg/ha)

| Parameter | Min | Max | Description |
|-----------|-----|-----|-------------|
| N (Nitrogen) | 0.0 | 300.0 | Typical agricultural range |
| P (Phosphorus) | 0.0 | 150.0 | Based on FAO guidelines |
| K (Potassium) | 0.0 | 300.0 | Standard soil fertility range |

### Soil Chemistry

| Parameter | Min | Max | Description |
|-----------|-----|-----|-------------|
| ph / pH | 3.0 | 10.0 | Extreme acidic to alkaline |

### Climate Parameters

| Parameter | Min | Max | Description |
|-----------|-----|-----|-------------|
| temperature / temp | -10.0 | 55.0 | °C, covers extreme climates |
| rainfall / rain | 0.0 | 5000.0 | mm, annual precipitation |
| humidity | 0.0 | 100.0 | Percentage |

### Crop Parameters

| Parameter | Min | Max | Description |
|-----------|-----|-----|-------------|
| CropCoveredArea | 0.01 | 100000.0 | Hectares |
| WaterCov | 0 | 100 | Water coverage percentage |
| CHeight | 0 | 500 | Crop height in cm |
| IrriCount | 0 | 365 | Irrigation count per season |

## API Error Responses

### Invalid Numeric Input (HTTP 422)

```json
{
  "detail": {
    "error": "invalid_input",
    "field": "ph",
    "value": "15.0",
    "constraint": "pH must be between 3.0 and 10.0",
    "message": "Invalid value for 'ph': 15.0. pH must be between 3.0 and 10.0"
  }
}
```

### Invalid Type (HTTP 422)

```json
{
  "detail": {
    "error": "invalid_input",
    "field": "N",
    "value": "invalid",
    "constraint": "Must be a valid number",
    "message": "Invalid value for 'N': invalid. Must be a valid number"
  }
}
```

### Special Float Values (HTTP 422)

```json
{
  "detail": {
    "error": "invalid_input",
    "field": "temperature",
    "value": "inf",
    "constraint": "Cannot be infinite",
    "message": "Invalid value for 'temperature': inf. Cannot be infinite"
  }
}
```

## Usage Examples

### Python Client

```python
import requests

# Valid request
response = requests.post("http://localhost:8000/predict", json={
    "Crop": "Wheat",
    "CropCoveredArea": 10.5,
    "CHeight": 120,
    "CNext": "Rice",
    "CLast": "Fallow",
    "CTransp": "Manual",
    "IrriType": "Drip",
    "IrriSource": "Borewell",
    "IrriCount": 15,
    "WaterCov": 80,
    "Season": "Rabi"
})

if response.status_code == 200:
    print(f"Predicted yield: {response.json()['predicted_ExpYield']}")
elif response.status_code == 422:
    error = response.json()['detail']
    print(f"Validation error in field '{error['field']}': {error['constraint']}")
```

### JavaScript Client

```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    Crop: 'Rice',
    CropCoveredArea: 5.0,
    // ... other fields
  })
});

if (response.status === 422) {
  const error = await response.json();
  console.error(`Invalid ${error.detail.field}: ${error.detail.constraint}`);
}
```

## Testing

### Unit Tests

Run the comprehensive test suite:

```bash
# Full pytest suite (requires all dependencies)
python -m pytest tests/test_input_validation.py -v

# Standalone test script (no dependencies)
python test_validation_simple.py
```

### Manual Testing

Test invalid pH:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Crop": "Wheat",
    "CropCoveredArea": 10,
    "CHeight": 120,
    "CNext": "Rice",
    "CLast": "Fallow",
    "CTransp": "Manual",
    "IrriType": "Drip",
    "IrriSource": "Borewell",
    "IrriCount": 15,
    "WaterCov": 80,
    "Season": "Rabi",
    "ph": 15.0
  }'
```

Expected response (HTTP 422):
```json
{
  "detail": {
    "error": "invalid_input",
    "field": "ph",
    "value": "15.0",
    "constraint": "pH must be between 3.0 and 10.0",
    "message": "Invalid value for 'ph': 15.0. pH must be between 3.0 and 10.0"
  }
}
```

## Adding New Constraints

To add validation for a new parameter:

1. **Add constraint to `ml/validators.py`**:

```python
NUMERIC_CONSTRAINTS = {
    # ... existing constraints ...
    "new_param": (min_value, max_value, "Description of constraint"),
}
```

2. **Test the constraint**:

```python
# In tests/test_input_validation.py
def test_new_param_validation():
    assert validate_numeric_input("new_param", valid_value) == valid_value
    
    with pytest.raises(InputValidationError):
        validate_numeric_input("new_param", invalid_value)
```

3. **Update documentation**: Add the parameter to this document's tables.

## Security Considerations

### Type Safety

- All numeric inputs are explicitly cast to `float`
- Non-numeric strings raise `InputValidationError`
- Special float values (NaN, Inf) are rejected

### Range Validation

- Constraints based on real-world agricultural ranges
- Prevents model exploitation via extreme values
- Fail-fast: first invalid field raises immediately

### Error Messages

- Descriptive but not verbose
- Include field name, value, and constraint
- Do not leak internal model details

## Performance

- **Overhead**: ~0.1ms per request (negligible)
- **Validation**: O(n) where n = number of numeric fields
- **Caching**: Constraints are module-level constants

## Backward Compatibility

- Existing valid requests continue to work unchanged
- String numbers (e.g., `"6.5"`) are coerced to float
- Categorical fields pass through without modification
- Only invalid inputs that would have caused errors are now rejected earlier with better messages

## References

- **FAO Soil Guidelines**: http://www.fao.org/soils-portal/
- **USDA Soil Database**: https://www.nrcs.usda.gov/
- **Agronomic Research**: Standard agricultural parameter ranges

## Changelog

### Version 1.0.0 (2026-05-12)

- Initial implementation of input validation system
- Added `ml/validators.py` with comprehensive range checks
- Integrated validation into preprocessing pipeline
- Added structured error responses (HTTP 422)
- Created test suite with 40+ test cases
- Documented all validated parameters and constraints
