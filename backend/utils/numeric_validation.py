"""
Shared numeric input validation utilities.

Extracted from ``_coerce_prediction_inputs`` in main.py so that any router
that processes soil or agronomic data can import and apply the same checks
without duplicating logic.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable

from fastapi import HTTPException

#: Accepted pH range for all soil and crop advisory endpoints.
_PH_MIN = 0.0
_PH_MAX = 14.0

# Generic numeric validation limits
_MAX_ABSOLUTE_VALUE = 1_000_000_000
_MIN_NONZERO_MAGNITUDE = 1e-12
_MAX_DECIMAL_PLACES = 12

def validate_numeric_bounds(
    data: Dict[str, Any],
    field_list: Iterable[str],
    *,
    ph_fields: Iterable[str] = ("ph", "pH"),
    max_absolute_value: float = _MAX_ABSOLUTE_VALUE,
    min_nonzero_magnitude: float = _MIN_NONZERO_MAGNITUDE,
) -> Dict[str, Any]:
    """Validate and coerce numeric fields in *data*, returning a sanitised copy.

    For every field in *field_list* that is present and non-``None`` in *data*:

    1. Coerce the value to ``float``; raise ``HTTP 400`` on conversion failure.
    2. Reject non-finite values (``inf``, ``-inf``, ``nan``); raise ``HTTP 400``.
    3. For any field whose name also appears in *ph_fields*, reject values
       outside ``[0, 14]``; raise ``HTTP 400``.

    Parameters
    ----------
    data:
        Mapping of field names to their raw values (typically from a Pydantic
        model's ``model_dump()`` or a form-parsing step).
    field_list:
        Names of the fields to validate.  Fields absent from *data* or whose
        value is ``None`` are silently skipped.
    ph_fields:
        Field names that represent pH measurements and must stay within
        ``[0, 14]``.  Defaults to ``("ph", "pH")``.

    Returns
    -------
    dict
        A shallow copy of *data* with validated fields replaced by their
        ``float`` representations.

    Raises
    ------
    fastapi.HTTPException
        ``status_code=400`` for any invalid or out-of-range value.
    """
    sanitized = dict(data)
    ph_fields_set = frozenset(ph_fields)

    for field in field_list:
        if field not in sanitized or sanitized[field] is None:
            continue

        try:
            numeric_value = float(sanitized[field])
        except (TypeError, ValueError, OverflowError):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value for '{field}': must be a number.",
            )

        if not math.isfinite(numeric_value):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value for '{field}': must be a finite number (not inf or NaN).",
            )
        
        if abs(numeric_value) > max_absolute_value:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid value for '{field}': "
                    f"absolute value exceeds {max_absolute_value}."
                ),
            )

        if (
            numeric_value != 0
            and abs(numeric_value) < min_nonzero_magnitude
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid value for '{field}': "
                    "value is below supported precision."
                ),
            )

        text_value = str(sanitized[field]).lower()

        if "." in text_value:
            fractional = text_value.split(".", 1)[1]

            if "e" in fractional:
                fractional = fractional.split("e", 1)[0]

            if len(fractional.rstrip("0")) > _MAX_DECIMAL_PLACES:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid value for '{field}': "
                        "excessive decimal precision."
                    ),
                )
            
        sanitized[field] = numeric_value

        if field in ph_fields_set and not (_PH_MIN <= numeric_value <= _PH_MAX):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid pH value for '{field}': "
                    f"must be between {_PH_MIN} and {_PH_MAX}."
                ),
            )

    return sanitized
