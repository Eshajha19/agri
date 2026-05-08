import pandas as pd
import numpy as np
from typing import List


class UnknownCategoryError(ValueError):
    """
    Raised when a categorical input value was not seen during training.

    The one-hot encoder (pd.get_dummies) silently drops unknown categories,
    producing no encoded columns for that value.  If we then fill those
    missing columns with 0 the model receives a row where *every* category
    for that feature is 0 — a structurally invalid input that looks valid
    but produces meaningless predictions.

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

    Raises
    ------
    UnknownCategoryError
        When a categorical column value was not present in the training data,
        meaning pd.get_dummies produced no encoded column for it.
    MissingFeatureError
        When one or more required numeric feature columns are absent after
        encoding and cannot be inferred from the input.
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
            If required feature columns are absent after encoding.
        """
        df = pd.DataFrame([input_data])

        # --- One-hot encode categorical columns ---
        categorical_cols_present = [
            col for col in self.dummy_cols if col in df.columns
        ]
        df = pd.get_dummies(df, columns=categorical_cols_present, drop_first=True)

        # --- Validate and align to expected feature schema ---
        if self.feature_cols:
            missing = [col for col in self.feature_cols if col not in df.columns]

            if missing:
                # Determine whether each missing column belongs to a categorical
                # group that was present in the input (unknown category) or is a
                # genuinely absent numeric feature.
                unknown_category_errors = []
                truly_missing = []

                for col in missing:
                    # e.g. "Crop_rice" → base column is "Crop"
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
                        unknown_category_errors.append(
                            UnknownCategoryError(
                                column=base_col,
                                value=input_data[base_col],
                                expected_columns=expected_for_group,
                            )
                        )
                    else:
                        truly_missing.append(col)

                # Report unknown categories first — they are the most actionable.
                if unknown_category_errors:
                    # Raise the first one; callers can catch and inspect .column / .value.
                    raise unknown_category_errors[0]

                if truly_missing:
                    raise MissingFeatureError(truly_missing)

            # Reorder columns to exactly match model expectations.
            df = df[self.feature_cols]

        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Placeholder for normalization (e.g. MinMaxScaler or StandardScaler).
        Can be extended for specific model requirements.
        """
        return df
