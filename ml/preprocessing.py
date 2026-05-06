import pandas as pd
import numpy as np
from typing import List

class FeaturePreprocessor:
    """
    Standardizes and normalizes input features for yield prediction models.
    """
    
    def __init__(self, feature_cols: List[str] = None):
        self.feature_cols = feature_cols
        self.dummy_cols = ['Crop', 'CNext', 'CLast', 'CTransp', 'IrriType', 'IrriSource', 'Season']

    def preprocess(self, input_data: dict) -> pd.DataFrame:
        """
        Convert raw input dictionary to a processed DataFrame.
        """
        df = pd.DataFrame([input_data])
        
        # Apply one-hot encoding for categorical variables
        df = pd.get_dummies(df, columns=[col for col in self.dummy_cols if col in df.columns], drop_first=True)
        
        # Ensure all expected feature columns are present (fill with 0 if missing)
        if self.feature_cols:
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0
            # Reorder columns to match model expectations
            df = df[self.feature_cols]
        
        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Placeholder for normalization (e.g., MinMaxScaler or StandardScaler).
        Can be extended for specific model requirements.
        """
        # For now, just return as is or implement basic scaling if needed
        return df
