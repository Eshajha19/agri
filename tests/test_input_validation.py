"""
Tests for ML input validation and sanitization.

This test suite verifies that:
1. Valid inputs pass through unchanged
2. Invalid numeric types are rejected
3. Out-of-range values are rejected with clear error messages
4. Special float values (NaN, Inf) are rejected
5. Categorical fields are not affected by numeric validation
"""

import pytest
import math
from ml.validators import (
    validate_numeric_input,
    validate_ml_inputs,
    InputValidationError,
    get_validation_summary,
)


class TestValidateNumericInput:
    """Test suite for single-field numeric validation."""

    def test_valid_ph_value(self):
        """Valid pH values should pass through unchanged."""
        assert validate_numeric_input("ph", 7.0) == 7.0
        assert validate_numeric_input("ph", 3.0) == 3.0
        assert validate_numeric_input("ph", 10.0) == 10.0

    def test_valid_ph_string(self):
        """Numeric strings should be coerced to float."""
        assert validate_numeric_input("ph", "6.5") == 6.5
        assert validate_numeric_input("pH", "7.2") == 7.2

    def test_invalid_ph_too_low(self):
        """pH below 3.0 should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_numeric_input("ph", 2.5)
        
        error = exc_info.value
        assert error.field == "ph"
        assert error.value == 2.5
        assert "3.0 and 10.0" in error.constraint

    def test_invalid_ph_too_high(self):
        """pH above 10.0 should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_numeric_input("ph", 15.0)
        
        error = exc_info.value
        assert error.field == "ph"
        assert error.value == 15.0
        assert "3.0 and 10.0" in error.constraint

    def test_invalid_ph_string(self):
        """Non-numeric strings should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_numeric_input("ph", "invalid")
        
        error = exc_info.value
        assert error.field == "ph"
        assert "valid number" in error.constraint

    def test_invalid_ph_nan(self):
        """NaN values should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_numeric_input("ph", float('nan'))
        
        error = exc_info.value
        assert error.field == "ph"
        assert "NaN" in error.constraint

    def test_invalid_ph_infinity(self):
        """Infinite values should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_numeric_input("ph", float('inf'))
        
        error = exc_info.value
        assert error.field == "ph"
        assert "infinite" in error.constraint

    def test_valid_nitrogen_values(self):
        """Valid nitrogen values should pass."""
        assert validate_numeric_input("N", 0.0) == 0.0
        assert validate_numeric_input("N", 150.0) == 150.0
        assert validate_numeric_input("N", 300.0) == 300.0

    def test_invalid_nitrogen_negative(self):
        """Negative nitrogen should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_numeric_input("N", -10.0)
        
        error = exc_info.value
        assert error.field == "N"
        assert error.value == -10.0

    def test_invalid_nitrogen_too_high(self):
        """Nitrogen above 300 kg/ha should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_numeric_input("N", 500.0)
        
        error = exc_info.value
        assert error.field == "N"
        assert error.value == 500.0

    def test_valid_phosphorus_values(self):
        """Valid phosphorus values should pass."""
        assert validate_numeric_input("P", 0.0) == 0.0
        assert validate_numeric_input("P", 75.0) == 75.0
        assert validate_numeric_input("P", 150.0) == 150.0

    def test_valid_potassium_values(self):
        """Valid potassium values should pass."""
        assert validate_numeric_input("K", 0.0) == 0.0
        assert validate_numeric_input("K", 150.0) == 150.0
        assert validate_numeric_input("K", 300.0) == 300.0

    def test_valid_temperature_values(self):
        """Valid temperature values should pass."""
        assert validate_numeric_input("temperature", -10.0) == -10.0
        assert validate_numeric_input("temperature", 25.0) == 25.0
        assert validate_numeric_input("temperature", 55.0) == 55.0

    def test_invalid_temperature_too_low(self):
        """Temperature below -10°C should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_numeric_input("temperature", -20.0)
        
        error = exc_info.value
        assert error.field == "temperature"

    def test_invalid_temperature_too_high(self):
        """Temperature above 55°C should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_numeric_input("temperature", 60.0)
        
        error = exc_info.value
        assert error.field == "temperature"

    def test_valid_rainfall_values(self):
        """Valid rainfall values should pass."""
        assert validate_numeric_input("rainfall", 0.0) == 0.0
        assert validate_numeric_input("rainfall", 1000.0) == 1000.0
        assert validate_numeric_input("rainfall", 5000.0) == 5000.0

    def test_valid_humidity_values(self):
        """Valid humidity values should pass."""
        assert validate_numeric_input("humidity", 0.0) == 0.0
        assert validate_numeric_input("humidity", 50.0) == 50.0
        assert validate_numeric_input("humidity", 100.0) == 100.0

    def test_invalid_humidity_too_high(self):
        """Humidity above 100% should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_numeric_input("humidity", 150.0)
        
        error = exc_info.value
        assert error.field == "humidity"

    def test_unconstrained_field(self):
        """Fields without defined constraints should pass any numeric value."""
        # This tests that unknown fields don't cause errors
        assert validate_numeric_input("unknown_field", 999999.0) == 999999.0


class TestValidateMLInputs:
    """Test suite for full input dictionary validation."""

    def test_valid_complete_input(self):
        """A complete valid input should pass through unchanged."""
        input_data = {
            "N": 50.0,
            "P": 30.0,
            "K": 40.0,
            "ph": 6.5,
            "temperature": 25.0,
            "rainfall": 800.0,
            "humidity": 65.0,
            "Crop": "Wheat",
            "Season": "Rabi",
        }
        
        result = validate_ml_inputs(input_data)
        
        # Numeric fields should be validated and converted to float
        assert result["N"] == 50.0
        assert result["P"] == 30.0
        assert result["K"] == 40.0
        assert result["ph"] == 6.5
        assert result["temperature"] == 25.0
        assert result["rainfall"] == 800.0
        assert result["humidity"] == 65.0
        
        # Categorical fields should pass through unchanged
        assert result["Crop"] == "Wheat"
        assert result["Season"] == "Rabi"

    def test_valid_input_with_string_numbers(self):
        """Numeric strings should be coerced to float."""
        input_data = {
            "N": "50",
            "P": "30.5",
            "ph": "6.5",
            "Crop": "Rice",
        }
        
        result = validate_ml_inputs(input_data)
        
        assert result["N"] == 50.0
        assert result["P"] == 30.5
        assert result["ph"] == 6.5
        assert result["Crop"] == "Rice"

    def test_invalid_ph_in_dict(self):
        """Invalid pH in a dictionary should raise InputValidationError."""
        input_data = {
            "N": 50.0,
            "P": 30.0,
            "ph": 15.0,  # Invalid: too high
            "Crop": "Wheat",
        }
        
        with pytest.raises(InputValidationError) as exc_info:
            validate_ml_inputs(input_data)
        
        error = exc_info.value
        assert error.field == "ph"
        assert error.value == 15.0

    def test_invalid_nitrogen_in_dict(self):
        """Invalid nitrogen in a dictionary should raise InputValidationError."""
        input_data = {
            "N": -10.0,  # Invalid: negative
            "P": 30.0,
            "ph": 6.5,
            "Crop": "Maize",
        }
        
        with pytest.raises(InputValidationError) as exc_info:
            validate_ml_inputs(input_data)
        
        error = exc_info.value
        assert error.field == "N"
        assert error.value == -10.0

    def test_invalid_string_in_numeric_field(self):
        """Non-numeric string in a numeric field should raise InputValidationError."""
        input_data = {
            "N": "invalid",  # Invalid: not a number
            "P": 30.0,
            "ph": 6.5,
            "Crop": "Cotton",
        }
        
        with pytest.raises(InputValidationError) as exc_info:
            validate_ml_inputs(input_data)
        
        error = exc_info.value
        assert error.field == "N"
        assert "valid number" in error.constraint

    def test_nan_in_numeric_field(self):
        """NaN in a numeric field should raise InputValidationError."""
        input_data = {
            "N": 50.0,
            "P": float('nan'),  # Invalid: NaN
            "ph": 6.5,
            "Crop": "Soybean",
        }
        
        with pytest.raises(InputValidationError) as exc_info:
            validate_ml_inputs(input_data)
        
        error = exc_info.value
        assert error.field == "P"
        assert "NaN" in error.constraint

    def test_infinity_in_numeric_field(self):
        """Infinity in a numeric field should raise InputValidationError."""
        input_data = {
            "N": 50.0,
            "P": 30.0,
            "temperature": float('inf'),  # Invalid: infinite
            "Crop": "Potato",
        }
        
        with pytest.raises(InputValidationError) as exc_info:
            validate_ml_inputs(input_data)
        
        error = exc_info.value
        assert error.field == "temperature"
        assert "infinite" in error.constraint

    def test_categorical_fields_unchanged(self):
        """Categorical fields should not be affected by validation."""
        input_data = {
            "Crop": "Wheat|Rice",  # Pipe character should pass through
            "Season": "Kharif",
            "IrriType": "Drip",
            "N": 50.0,
        }
        
        result = validate_ml_inputs(input_data)
        
        assert result["Crop"] == "Wheat|Rice"
        assert result["Season"] == "Kharif"
        assert result["IrriType"] == "Drip"
        assert result["N"] == 50.0

    def test_empty_input(self):
        """Empty input should return empty output."""
        result = validate_ml_inputs({})
        assert result == {}

    def test_only_categorical_fields(self):
        """Input with only categorical fields should pass through."""
        input_data = {
            "Crop": "Wheat",
            "Season": "Rabi",
            "IrriType": "Flood",
        }
        
        result = validate_ml_inputs(input_data)
        assert result == input_data

    def test_boundary_values(self):
        """Boundary values should be accepted."""
        input_data = {
            "ph": 3.0,  # Minimum
            "N": 0.0,   # Minimum
            "P": 150.0, # Maximum
            "humidity": 100.0,  # Maximum
        }
        
        result = validate_ml_inputs(input_data)
        
        assert result["ph"] == 3.0
        assert result["N"] == 0.0
        assert result["P"] == 150.0
        assert result["humidity"] == 100.0

    def test_just_outside_boundaries(self):
        """Values just outside boundaries should be rejected."""
        # Test pH just below minimum
        with pytest.raises(InputValidationError):
            validate_ml_inputs({"ph": 2.99})
        
        # Test pH just above maximum
        with pytest.raises(InputValidationError):
            validate_ml_inputs({"ph": 10.01})
        
        # Test nitrogen just below minimum
        with pytest.raises(InputValidationError):
            validate_ml_inputs({"N": -0.01})
        
        # Test phosphorus just above maximum
        with pytest.raises(InputValidationError):
            validate_ml_inputs({"P": 150.01})


class TestGetValidationSummary:
    """Test suite for validation summary retrieval."""

    def test_summary_contains_all_constraints(self):
        """Summary should contain all defined constraints."""
        summary = get_validation_summary()
        
        assert "N" in summary
        assert "P" in summary
        assert "K" in summary
        assert "ph" in summary
        assert "pH" in summary
        assert "temperature" in summary
        assert "rainfall" in summary
        assert "humidity" in summary

    def test_summary_structure(self):
        """Each constraint should be a tuple of (min, max, description)."""
        summary = get_validation_summary()
        
        for field, constraint in summary.items():
            assert isinstance(constraint, tuple)
            assert len(constraint) == 3
            min_val, max_val, description = constraint
            assert isinstance(min_val, (int, float))
            assert isinstance(max_val, (int, float))
            assert isinstance(description, str)
            assert min_val < max_val

    def test_summary_is_copy(self):
        """Summary should be a copy, not a reference to the original."""
        summary1 = get_validation_summary()
        summary2 = get_validation_summary()
        
        # Modifying one should not affect the other
        summary1["test_field"] = (0, 100, "Test")
        assert "test_field" not in summary2


class TestInputValidationError:
    """Test suite for InputValidationError exception."""

    def test_error_attributes(self):
        """Error should contain field, value, and constraint attributes."""
        error = InputValidationError(
            field="ph",
            value=15.0,
            constraint="pH must be between 3.0 and 10.0"
        )
        
        assert error.field == "ph"
        assert error.value == 15.0
        assert error.constraint == "pH must be between 3.0 and 10.0"

    def test_error_message(self):
        """Error message should be descriptive."""
        error = InputValidationError(
            field="N",
            value=-10,
            constraint="Nitrogen must be between 0 and 300 kg/ha"
        )
        
        message = str(error)
        assert "N" in message
        assert "-10" in message
        assert "Nitrogen must be between 0 and 300 kg/ha" in message

    def test_error_is_value_error(self):
        """InputValidationError should be a subclass of ValueError."""
        error = InputValidationError("field", "value", "constraint")
        assert isinstance(error, ValueError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
