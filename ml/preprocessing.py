import pandas as pd
import numpy as np
from typing import List


class UnknownCategoryError(ValueError):
    """
    Raised when a categorical input value was not seen during training.

    Attributes
    ----------
    column : str
        The original (pre-encoding) categorical column name.
    value : object
        The value that was not recognised.
    expected_columns : list[str]
        The one-hot columns the model expected for this feature group.
    """

    def __init__(self, column: str, value: object, expected_columns: List[str]):
        self.column = column
        self.value = value
        self.expected_columns = expected_columns
        super().__init__(
            f"Unknown value '{value}' for categorical feature '{column}'. "
            f"The model was not trained on this value. "
            f"Expected one of the encoded columns: {expected_columns}"
        )


class MissingFeatureError(ValueError):
    """
    Raised when a required numeric feature column is absent from the input.

    Attributes
    ----------
    missing_columns : list[str]
        Feature columns that were expected but not present after encoding.
    """

    def __init__(self, missing_columns: List[str]):
        self.missing_columns = missing_columns
        super().__init__(
            f"Input is missing {len(missing_columns)} required feature(s): "
            f"{missing_columns}. "
            "Provide all required fields or check that categorical values "
            "match the training vocabulary."
        )


class FeaturePreprocessor:
    """
    Standardises and validates input features for yield prediction models.

    Key design note
    ---------------
    The model was trained with ``pd.get_dummies(..., drop_first=True)`` on a
    multi-row dataset.  At inference time we receive a **single row**, and
    ``drop_first=True`` on a single row silently drops *every* categorical
    column (there is only one unique value per column, so the "first" is the
    only one and it gets dropped).  This caused a 500 error at prediction time.

    The fix is to encode with ``drop_first=False`` (keeping all dummies) and
    then align the resulting DataFrame to ``feature_cols`` by:
      - adding any column that is in ``feature_cols`` but absent from the
        encoded row as a zero column (this covers the baseline/dropped
        categories from training), and
      - dropping any column that is not in ``feature_cols``.

    Unknown categories (values the model was never trained on) are still
    detected and raised as ``UnknownCategoryError``.

    Raises
    ------
    UnknownCategoryError
        When a categorical column value was not present in the training data.
    MissingFeatureError
        When one or more required *numeric* feature columns are absent after
        encoding (i.e. the caller omitted a required field entirely).
    """

    def __init__(self, feature_cols: List[str] = None):
        self.feature_cols = feature_cols
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
        UnknownCategoryError
            If a categorical value produces no encoded columns (unknown category).
        MissingFeatureError
            If required numeric feature columns are absent after encoding.
        """
        df = pd.DataFrame([input_data])

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

        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Placeholder for normalization (e.g. MinMaxScaler or StandardScaler).
        Can be extended for specific model requirements.
        """
        return df
