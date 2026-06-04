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

    for column in numeric_columns:

        if column not in dataframe.columns:
            continue

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        dataframe[column] = dataframe[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    return dataframe


def validate_categorical_values(
    dataframe: pd.DataFrame,
    categorical_vocab: Dict[str, List[str]],
):
    """
    Validate categorical values against vocab.
    """

    for column, allowed_values in categorical_vocab.items():

        if column not in dataframe.columns:
            continue

        for value in dataframe[column].dropna().unique():

            if value not in allowed_values:
                raise UnknownCategoryError(
                    column,
                    value,
                )


# =============================================================================
# MAIN PREPROCESSOR
# =============================================================================

def preprocess_prediction_input(
    input_data: Dict,
    required_columns: List[str],
    numeric_columns: List[str],
    categorical_vocab: Dict[str, List[str]],
):
    """
    Main preprocessing pipeline.
    """

    if not isinstance(input_data, dict):
        raise ValueError(
            "input_data must be dictionary"
        )

    dataframe = pd.DataFrame([input_data])

    ensure_required_features(
        dataframe,
        required_columns,
    )

    dataframe = sanitize_numeric_columns(
        dataframe,
        numeric_columns,
    )

    validate_categorical_values(
        dataframe,
        categorical_vocab,
    )

    dataframe = dataframe.fillna(0)

    logger.info(
        "Preprocessing completed successfully"
    )

    return dataframe