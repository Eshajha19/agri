import logging
from typing import Dict, List

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM ERRORS
# =============================================================================

class UnknownCategoryError(Exception):
    """
    Raised when unseen categorical value appears.
    """

    def __init__(self, column: str, value):
        self.column = column
        self.value = value

        super().__init__(
            f"Unknown category '{value}' for column '{column}'"
        )


class MissingFeatureError(Exception):
    """
    Raised when required feature columns are missing.
    """

    def __init__(self, missing_columns: List[str]):
        self.missing_columns = missing_columns

        super().__init__(
            f"Missing required features: {missing_columns}"
        )


# =============================================================================
# HELPERS
# =============================================================================

def ensure_required_features(
    dataframe: pd.DataFrame,
    required_columns: List[str],
):
    """
    Ensure all required columns exist.
    """

    missing = [
        col
        for col in required_columns
        if col not in dataframe.columns
    ]

    if missing:
        raise MissingFeatureError(missing)


def sanitize_numeric_columns(
    dataframe: pd.DataFrame,
    numeric_columns: List[str],
):
    """
    Safely convert numeric columns.
    """

    def __init__(self, feature_cols: List[str] = None, category_vocab: dict = None):
        self.feature_cols = feature_cols
        self.category_vocab = category_vocab or {}
        self.dummy_cols = [
            "Crop", "CNext", "CLast", "CTransp",
            "IrriType", "IrriSource", "Season",
        ]

    def preprocess(self, input_data: dict) -> pd.DataFrame:
        """
        Convert a raw input dictionary to a validated, encoded DataFrame.

        Parameters
        ----------
        input_data : dict
            Raw feature dictionary from the API request.

        Returns
        -------
        pd.DataFrame
            A single-row DataFrame with columns matching ``self.feature_cols``.

        Raises
        ------
        InputValidationError
            If any numeric parameter is out of acceptable range or invalid type.
        UnknownCategoryError
            If a categorical value produces no encoded columns (unknown category).
        MissingFeatureError
            If required numeric feature columns are absent after encoding.
        """
        # Step 1: Validate and sanitize numeric inputs BEFORE any processing
        # This prevents invalid values from reaching the model
        validated_data = validate_ml_inputs(input_data)
        
        # Ensure deterministic column order from the start
        df = pd.DataFrame([validated_data])
        df = df.reindex(sorted(df.columns), axis=1)

        # --- Validate categorical values against training vocabulary ---
        if self.category_vocab:
            for col in self.dummy_cols:
                if col in df.columns:
                    val = str(df[col].iloc[0])
                    valid_values = self.category_vocab.get(col)
                    if valid_values and val not in valid_values:
                        expected_columns = [
                            c for c in (self.feature_cols or [])
                            if c.startswith(f"{col}_")
                        ]
                        raise UnknownCategoryError(
                            column=col,
                            value=val,
                            expected_columns=expected_columns,
                        )

        # --- One-hot encode with drop_first=False ---
        # Using drop_first=True on a single-row DataFrame silently drops ALL
        # categorical columns because every column has only one unique value.
        # We keep all dummies here and align to feature_cols below instead.
        categorical_cols_present = [
            col for col in self.dummy_cols if col in df.columns
        ]
        df = pd.get_dummies(df, columns=categorical_cols_present, drop_first=False)

        # --- Validate and align to expected feature schema ---
        if self.feature_cols:
            missing = [col for col in self.feature_cols if col not in df.columns]

            if missing:
                # Classify each missing column: unknown category vs truly absent.
                unknown_category_errors = []
                truly_missing = []

                for col in missing:
                    # e.g. "Crop_Rice" → base column is "Crop"
                    base_col = next(
                        (c for c in self.dummy_cols if col.startswith(f"{c}_")),
                        None,
                    )
                    if base_col and base_col in input_data:
                        # The base categorical column was provided but its value
                        # produced no encoded column → unknown category.
                        expected_for_group = [
                            c for c in self.feature_cols
                            if c.startswith(f"{base_col}_")
                        ]
                        # Check whether ANY column for this group was produced.
                        # If at least one was produced, the value is known but
                        # this particular dummy is the baseline (dropped during
                        # training) — fill with 0, do NOT raise.
                        produced_for_group = [
                            c for c in df.columns
                            if c.startswith(f"{base_col}_")
                        ]
                        if not produced_for_group:
                            # No column at all for this group → truly unknown value.
                            unknown_category_errors.append(
                                UnknownCategoryError(
                                    column=base_col,
                                    value=input_data[base_col],
                                    expected_columns=expected_for_group,
                                )
                            )
                        else:
                            # The group has at least one produced column; this
                            # missing column is just the baseline — add as 0.
                            df[col] = 0
                    else:
                        truly_missing.append(col)

                # Report unknown categories first — they are the most actionable.
                if unknown_category_errors:
                    raise unknown_category_errors[0]

                # Fill any remaining baseline/dropped columns with 0.
                still_missing = [
                    col for col in self.feature_cols if col not in df.columns
                ]
                numeric_missing = [
                    col for col in still_missing
                    if not any(col.startswith(f"{c}_") for c in self.dummy_cols)
                ]
                if numeric_missing:
                    raise MissingFeatureError(numeric_missing)

                # Add zero columns for any remaining categorical baselines.
                for col in still_missing:
                    df[col] = 0

            # Reorder columns to exactly match model expectations and drop extras.
            df = df[self.feature_cols]
        else:
            # If no feature_cols are provided, still ensure deterministic order
            df = df.reindex(sorted(df.columns), axis=1)

        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Placeholder for normalization (e.g. MinMaxScaler or StandardScaler).
        Can be extended for specific model requirements.
        """
        return df
