import pytest
from fastapi import HTTPException

from backend.utils.numeric_validation import validate_numeric_bounds


def test_valid_ph_lower_bound():
    result = validate_numeric_bounds({"ph": 0}, ["ph"])
    assert result["ph"] == 0


def test_valid_ph_upper_bound():
    result = validate_numeric_bounds({"ph": 14}, ["ph"])
    assert result["ph"] == 14


def test_invalid_ph_low():
    with pytest.raises(HTTPException):
        validate_numeric_bounds({"ph": -0.01}, ["ph"])


def test_invalid_ph_high():
    with pytest.raises(HTTPException):
        validate_numeric_bounds({"ph": 14.01}, ["ph"])


def test_nan_rejected():
    with pytest.raises(HTTPException):
        validate_numeric_bounds({"n": float("nan")}, ["n"])


def test_inf_rejected():
    with pytest.raises(HTTPException):
        validate_numeric_bounds({"n": float("inf")}, ["n"])


def test_negative_inf_rejected():
    with pytest.raises(HTTPException):
        validate_numeric_bounds({"n": float("-inf")}, ["n"])


def test_large_value_rejected():
    with pytest.raises(HTTPException):
        validate_numeric_bounds({"n": 1e12}, ["n"])


def test_small_value_rejected():
    with pytest.raises(HTTPException):
        validate_numeric_bounds({"n": 1e-20}, ["n"])


def test_scientific_notation_allowed():
    result = validate_numeric_bounds({"n": "1e6"}, ["n"])
    assert result["n"] == 1000000.0


def test_excessive_precision_rejected():
    with pytest.raises(HTTPException):
        validate_numeric_bounds(
            {"n": "0.123456789012345678"},
            ["n"],
        )