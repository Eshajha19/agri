# ml/validators.py
"""
Input validation and sanitization for ML prediction endpoints.

This module provides range validation for agricultural parameters to prevent:
1. Internal server errors from extreme/invalid values
2. Nonsensical predictions from out-of-range inputs
3. Type errors from non-numeric strings

All validators follow the fail-fast principle: invalid inputs raise
ValueError with actionable error messages rather than being silently
coerced or clamped.
"""

from typing import Dict, Any, List, Tuple
import math


class InputValidationError(ValueError):
    """
    Raised when ML input parameters fail validation.

    Attributes
    ----------
    field : str
        The parameter name that failed validation.
    value : Any
        The invalid value that was provided.
    constraint : str
        Human-readable description of the constraint that was violated.
    """

    def __init__(self, field: str, value: Any, constraint: str):
        self.field = field
        self.value = value
        self.constraint = constraint
        super().__init__(
            f"Invalid value for '{field}': {value}. {constraint}"
        )


# Agricultural parameter constraints based on real-world ranges
# Sources: FAO guidelines, USDA soil databases, agronomic research
NUMERIC_CONSTRAINTS = {
    # Soil nutrients (kg/ha) - typical agricultural ranges
    "N": (0.0, 300.0, "Nitrogen must be between 0 and 300 kg/ha"),
    "P": (0.0, 150.0, "Phosphorus must be between 0 and 150 kg/ha"),
    "K": (0.0, 300.0, "Potassium must be between 0 and 300 kg/ha"),
    
    # Soil chemistry
    "ph": (3.0, 10.0, "pH must be between 3.0 and 10.0"),
    "pH": (3.0, 10.0, "pH must be between 3.0 and 10.0"),
    
    # Climate parameters
    "temperature": (-10.0, 55.0, "Temperature must be between -10°C and 55°C"),
    "temp": (-10.0, 55.0, "Temperature must be between -10°C and 55°C"),
    "rainfall": (0.0, 5000.0, "Rainfall must be between 0 and 5000 mm"),
    "rain": (0.0, 5000.0, "Rainfall must be between 0 and 5000 mm"),
    "humidity": (0.0, 100.0, "Humidity must be between 0% and 100%"),
    
    # Area and coverage (already validated by Pydantic but included for completeness)
    "CropCoveredArea": (0.01, 100000.0, "Crop area must be between 0.01 and 100,000 hectares"),
    "WaterCov": (0, 100, "Water coverage must be between 0% and 100%"),
    
    # Crop parameters
    "CHeight": (0, 500, "Crop height must be between 0 and 500 cm"),
    "IrriCount": (0, 365, "Irrigation count must be between 0 and 365 per season"),
}


def validate_numeric_input(field: str, value: Any) -> float:
    """
    Validate and sanitize a single numeric input parameter.

    Parameters
    ----------
    field : str
        The parameter name (e.g., "N", "ph", "temperature").
    value : Any
        The value to validate (will be coerced to float).

    Returns
    -------
    float
        The validated numeric value.

    Raises
    ------
    InputValidationError
        If the value cannot be converted to float, is NaN/Inf, or is
        outside the acceptable range for this parameter.

    Examples
    --------
    >>> validate_numeric_input("ph", 7.5)
    7.5
    >>> validate_numeric_input("ph", "6.8")
    6.8
    >>> validate_numeric_input("ph", 15.0)
    InputValidationError: Invalid value for 'ph': 15.0. pH must be between 3.0 and 10.0
    >>> validate_numeric_input("ph", "invalid")
    InputValidationError: Invalid value for 'ph': invalid. Must be a valid number
    """
    # Step 1: Type coercion
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise InputValidationError(
            field=field,
            value=value,
            constraint="Must be a valid number"
        )

    # Step 2: Check for special float values
    if math.isnan(numeric_value):
        raise InputValidationError(
            field=field,
            value=value,
            constraint="Cannot be NaN (Not a Number)"
        )
    
    if math.isinf(numeric_value):
        raise InputValidationError(
            field=field,
            value=value,
            constraint="Cannot be infinite"
        )

    # Step 3: Range validation (if constraints are defined for this field)
    if field in NUMERIC_CONSTRAINTS:
        min_val, max_val, message = NUMERIC_CONSTRAINTS[field]
        if not (min_val <= numeric_value <= max_val):
            raise InputValidationError(
                field=field,
                value=numeric_value,
                constraint=message
            )

    return numeric_value


def validate_ml_inputs(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate all numeric parameters in an ML prediction request.

    This function scans the input dictionary for known numeric parameters
    and validates each one according to agricultural constraints. Non-numeric
    fields (categorical variables like crop names) are passed through unchanged.

    Parameters
    ----------
    input_data : dict
        Raw input dictionary from the API request.

    Returns
    -------
    dict
        Validated input dictionary with all numeric values sanitized.
        Categorical fields are unchanged.

    Raises
    ------
    InputValidationError
        If any numeric parameter fails validation. The exception contains
        the field name, invalid value, and constraint description.

    Examples
    --------
    >>> validate_ml_inputs({"N": 50, "P": 30, "ph": 6.5, "Crop": "Wheat"})
    {'N': 50.0, 'P': 30.0, 'ph': 6.5, 'Crop': 'Wheat'}
    
    >>> validate_ml_inputs({"N": "invalid", "Crop": "Rice"})
    InputValidationError: Invalid value for 'N': invalid. Must be a valid number
    
    >>> validate_ml_inputs({"ph": 15.0, "Crop": "Maize"})
    InputValidationError: Invalid value for 'ph': 15.0. pH must be between 3.0 and 10.0
    """
    validated = {}
    errors: List[InputValidationError] = []

    for field, value in input_data.items():
        # Check if this field has numeric constraints
        if field in NUMERIC_CONSTRAINTS:
            try:
                validated[field] = validate_numeric_input(field, value)
            except InputValidationError as e:
                errors.append(e)
        else:
            # Pass through non-numeric fields unchanged
            validated[field] = value

    # Report the first validation error (fail-fast)
    if errors:
        raise errors[0]

    return validated


def get_validation_summary() -> Dict[str, Tuple[float, float, str]]:
    """
    Return a summary of all numeric validation constraints.

    Useful for API documentation and client-side validation.

    Returns
    -------
    dict
        Mapping of field names to (min, max, description) tuples.

    Examples
    --------
    >>> summary = get_validation_summary()
    >>> summary["ph"]
    (3.0, 10.0, 'pH must be between 3.0 and 10.0')
    """
    return NUMERIC_CONSTRAINTS.copy()
