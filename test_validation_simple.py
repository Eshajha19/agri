#!/usr/bin/env python3
"""
Simple standalone test script for input validation.
Run with: python test_validation_simple.py
"""

from ml.validators import validate_numeric_input, validate_ml_inputs, InputValidationError

def test_valid_inputs():
    """Test that valid inputs pass through correctly."""
    print("Testing valid inputs...")
    
    # Test valid pH
    result = validate_numeric_input("ph", 7.0)
    assert result == 7.0, f"Expected 7.0, got {result}"
    print("  ✓ Valid pH 7.0 passed")
    
    # Test valid nitrogen
    result = validate_numeric_input("N", 50.0)
    assert result == 50.0, f"Expected 50.0, got {result}"
    print("  ✓ Valid N 50.0 passed")
    
    # Test string coercion
    result = validate_numeric_input("ph", "6.5")
    assert result == 6.5, f"Expected 6.5, got {result}"
    print("  ✓ String '6.5' coerced to float")
    
    print("✓ All valid input tests passed\n")


def test_invalid_ranges():
    """Test that out-of-range values are rejected."""
    print("Testing invalid ranges...")
    
    # Test pH too high
    try:
        validate_numeric_input("ph", 15.0)
        print("  ✗ FAILED: pH 15.0 should have been rejected")
        return False
    except InputValidationError as e:
        assert e.field == "ph"
        assert e.value == 15.0
        assert "3.0 and 10.0" in e.constraint
        print(f"  ✓ pH 15.0 correctly rejected: {e.constraint}")
    
    # Test pH too low
    try:
        validate_numeric_input("ph", 2.0)
        print("  ✗ FAILED: pH 2.0 should have been rejected")
        return False
    except InputValidationError as e:
        assert e.field == "ph"
        print(f"  ✓ pH 2.0 correctly rejected: {e.constraint}")
    
    # Test negative nitrogen
    try:
        validate_numeric_input("N", -10.0)
        print("  ✗ FAILED: N -10.0 should have been rejected")
        return False
    except InputValidationError as e:
        assert e.field == "N"
        assert e.value == -10.0
        print(f"  ✓ N -10.0 correctly rejected: {e.constraint}")
    
    # Test nitrogen too high
    try:
        validate_numeric_input("N", 500.0)
        print("  ✗ FAILED: N 500.0 should have been rejected")
        return False
    except InputValidationError as e:
        assert e.field == "N"
        print(f"  ✓ N 500.0 correctly rejected: {e.constraint}")
    
    print("✓ All invalid range tests passed\n")
    return True


def test_invalid_types():
    """Test that invalid types are rejected."""
    print("Testing invalid types...")
    
    # Test non-numeric string
    try:
        validate_numeric_input("ph", "invalid")
        print("  ✗ FAILED: 'invalid' should have been rejected")
        return False
    except InputValidationError as e:
        assert e.field == "ph"
        assert "valid number" in e.constraint
        print(f"  ✓ String 'invalid' correctly rejected: {e.constraint}")
    
    # Test NaN
    try:
        validate_numeric_input("ph", float('nan'))
        print("  ✗ FAILED: NaN should have been rejected")
        return False
    except InputValidationError as e:
        assert e.field == "ph"
        assert "NaN" in e.constraint
        print(f"  ✓ NaN correctly rejected: {e.constraint}")
    
    # Test infinity
    try:
        validate_numeric_input("ph", float('inf'))
        print("  ✗ FAILED: Infinity should have been rejected")
        return False
    except InputValidationError as e:
        assert e.field == "ph"
        assert "infinite" in e.constraint
        print(f"  ✓ Infinity correctly rejected: {e.constraint}")
    
    print("✓ All invalid type tests passed\n")
    return True


def test_full_input_validation():
    """Test validation of complete input dictionaries."""
    print("Testing full input validation...")
    
    # Test valid complete input
    input_data = {
        "N": 50.0,
        "P": 30.0,
        "K": 40.0,
        "ph": 6.5,
        "temperature": 25.0,
        "Crop": "Wheat",
        "Season": "Rabi",
    }
    
    result = validate_ml_inputs(input_data)
    assert result["N"] == 50.0
    assert result["P"] == 30.0
    assert result["ph"] == 6.5
    assert result["Crop"] == "Wheat"
    print("  ✓ Valid complete input passed")
    
    # Test input with string numbers
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
    print("  ✓ String numbers correctly coerced")
    
    # Test invalid pH in dictionary
    try:
        validate_ml_inputs({
            "N": 50.0,
            "ph": 15.0,
            "Crop": "Wheat",
        })
        print("  ✗ FAILED: Invalid pH in dict should have been rejected")
        return False
    except InputValidationError as e:
        assert e.field == "ph"
        print(f"  ✓ Invalid pH in dict correctly rejected")
    
    # Test categorical fields unchanged
    input_data = {
        "Crop": "Wheat|Rice",
        "Season": "Kharif",
        "N": 50.0,
    }
    
    result = validate_ml_inputs(input_data)
    assert result["Crop"] == "Wheat|Rice"
    assert result["Season"] == "Kharif"
    print("  ✓ Categorical fields passed through unchanged")
    
    print("✓ All full input validation tests passed\n")
    return True


def test_boundary_values():
    """Test boundary values are accepted."""
    print("Testing boundary values...")
    
    # Test minimum values
    assert validate_numeric_input("ph", 3.0) == 3.0
    assert validate_numeric_input("N", 0.0) == 0.0
    print("  ✓ Minimum boundary values accepted")
    
    # Test maximum values
    assert validate_numeric_input("ph", 10.0) == 10.0
    assert validate_numeric_input("P", 150.0) == 150.0
    assert validate_numeric_input("humidity", 100.0) == 100.0
    print("  ✓ Maximum boundary values accepted")
    
    # Test just outside boundaries
    try:
        validate_numeric_input("ph", 2.99)
        print("  ✗ FAILED: pH 2.99 should have been rejected")
        return False
    except InputValidationError:
        print("  ✓ pH 2.99 (just below min) correctly rejected")
    
    try:
        validate_numeric_input("ph", 10.01)
        print("  ✗ FAILED: pH 10.01 should have been rejected")
        return False
    except InputValidationError:
        print("  ✓ pH 10.01 (just above max) correctly rejected")
    
    print("✓ All boundary value tests passed\n")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("ML Input Validation Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_valid_inputs()
        test_invalid_ranges()
        test_invalid_types()
        test_full_input_validation()
        test_boundary_values()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
