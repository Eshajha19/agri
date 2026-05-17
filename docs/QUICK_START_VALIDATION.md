# Quick Start: Input Validation

## For API Consumers

### Valid Request Example
```python
import requests

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
    print(f"Validation error: {error['message']}")
```

### Handling Validation Errors
```python
def predict_yield(data):
    response = requests.post("http://localhost:8000/predict", json=data)
    
    if response.status_code == 200:
        return response.json()['predicted_ExpYield']
    
    elif response.status_code == 422:
        error = response.json()['detail']
        
        # Field-specific error handling
        if error['error'] == 'invalid_input':
            raise ValueError(f"Invalid {error['field']}: {error['constraint']}")
        
        elif error['error'] == 'unknown_category':
            raise ValueError(f"Unknown {error['field']}: {error['value']}")
        
        elif error['error'] == 'missing_features':
            raise ValueError(f"Missing fields: {error['missing']}")
    
    else:
        response.raise_for_status()
```

## Parameter Constraints

### Quick Reference Table

| Parameter | Type | Min | Max | Unit |
|-----------|------|-----|-----|------|
| N | float | 0 | 300 | kg/ha |
| P | float | 0 | 150 | kg/ha |
| K | float | 0 | 300 | kg/ha |
| ph / pH | float | 3.0 | 10.0 | - |
| temperature | float | -10 | 55 | °C |
| rainfall | float | 0 | 5000 | mm |
| humidity | float | 0 | 100 | % |
| CropCoveredArea | float | 0.01 | 100000 | hectares |
| WaterCov | int | 0 | 100 | % |
| CHeight | int | 0 | 500 | cm |
| IrriCount | int | 0 | 365 | count |

## Common Validation Errors

### 1. Out of Range
```json
{
  "detail": {
    "error": "invalid_input",
    "field": "ph",
    "value": "15.0",
    "constraint": "pH must be between 3.0 and 10.0"
  }
}
```

**Fix**: Ensure value is within the valid range.

### 2. Invalid Type
```json
{
  "detail": {
    "error": "invalid_input",
    "field": "N",
    "value": "invalid",
    "constraint": "Must be a valid number"
  }
}
```

**Fix**: Provide a numeric value (int or float).

### 3. Special Values
```json
{
  "detail": {
    "error": "invalid_input",
    "field": "temperature",
    "value": "nan",
    "constraint": "Cannot be NaN (Not a Number)"
  }
}
```

**Fix**: Ensure no NaN or Infinity values are sent.

## Frontend Integration

### React Example
```javascript
const predictYield = async (formData) => {
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData)
    });
    
    if (response.status === 422) {
      const error = await response.json();
      // Show field-specific error
      setFieldError(error.detail.field, error.detail.constraint);
      return null;
    }
    
    if (!response.ok) {
      throw new Error('Prediction failed');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
};
```

### Form Validation
```javascript
const validateField = (name, value) => {
  const constraints = {
    ph: { min: 3.0, max: 10.0, label: 'pH' },
    N: { min: 0, max: 300, label: 'Nitrogen (kg/ha)' },
    P: { min: 0, max: 150, label: 'Phosphorus (kg/ha)' },
    temperature: { min: -10, max: 55, label: 'Temperature (°C)' },
    humidity: { min: 0, max: 100, label: 'Humidity (%)' }
  };
  
  const constraint = constraints[name];
  if (!constraint) return null;
  
  const numValue = parseFloat(value);
  if (isNaN(numValue)) {
    return `${constraint.label} must be a valid number`;
  }
  
  if (numValue < constraint.min || numValue > constraint.max) {
    return `${constraint.label} must be between ${constraint.min} and ${constraint.max}`;
  }
  
  return null;
};
```

## Testing Your Integration

### Test Invalid pH
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Crop":"Wheat","CropCoveredArea":10,"CHeight":120,"CNext":"Rice","CLast":"Fallow","CTransp":"Manual","IrriType":"Drip","IrriSource":"Borewell","IrriCount":15,"WaterCov":80,"Season":"Rabi","ph":15.0}'
```

Expected: HTTP 422 with validation error

### Test Valid Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Crop":"Wheat","CropCoveredArea":10.5,"CHeight":120,"CNext":"Rice","CLast":"Fallow","CTransp":"Manual","IrriType":"Drip","IrriSource":"Borewell","IrriCount":15,"WaterCov":80,"Season":"Rabi"}'
```

Expected: HTTP 200 with prediction

## Best Practices

1. **Client-Side Validation**: Validate inputs before sending to reduce server load
2. **Error Display**: Show field-specific errors next to the input field
3. **Type Coercion**: The API accepts string numbers (e.g., `"6.5"`), but prefer sending actual numbers
4. **Boundary Testing**: Test with minimum and maximum values during development
5. **Error Logging**: Log validation errors for debugging and monitoring

## Need Help?

- **Full Documentation**: See `docs/INPUT_VALIDATION.md`
- **Test Examples**: See `tests/test_input_validation.py`
- **Security Details**: See `SECURITY_FIX_INPUT_SANITIZATION.md`
