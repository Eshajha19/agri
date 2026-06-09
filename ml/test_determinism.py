import pandas as pd
import unittest
from ml.preprocessing import FeaturePreprocessor

class TestMLDeterminism(unittest.TestCase):
    def setUp(self):
        # Mock feature columns to simulate a real model expectation
        self.feature_cols = [
            "N", "P", "K", "ph", "temperature", "rainfall", "humidity",
            "Crop_Rice", "Crop_Wheat", "Season_Kharif", "Season_Rabi"
        ]
        self.preprocessor = FeaturePreprocessor(feature_cols=self.feature_cols)

    def test_json_key_order_determinism(self):
        """Verify that same input values produce same output regardless of key order."""
        
        # Input with order A
        input_a = {
            "N": 50,
            "P": 30,
            "K": 20,
            "ph": 6.5,
            "temperature": 25.0,
            "rainfall": 1000,
            "humidity": 80,
            "Crop": "Rice",
            "Season": "Kharif"
        }
        
        # Input with order B (reversed)
        input_b = {
            "Season": "Kharif",
            "Crop": "Rice",
            "humidity": 80,
            "rainfall": 1000,
            "temperature": 25.0,
            "ph": 6.5,
            "K": 20,
            "P": 30,
            "N": 50
        }
        
        df_a = self.preprocessor.preprocess(input_a)
        df_b = self.preprocessor.preprocess(input_b)
        
        # Check if DataFrames are identical in values and column order
        pd.testing.assert_frame_equal(df_a, df_b)
        
        # Check specifically that the column order matches self.feature_cols
        self.assertEqual(list(df_a.columns), self.feature_cols)
        self.assertEqual(list(df_b.columns), self.feature_cols)

    def test_determinism_without_feature_cols(self):
        """Verify determinism even when feature_cols is not provided (should sort alphabetically)."""
        preprocessor_no_cols = FeaturePreprocessor(feature_cols=None)
        
        input_a = {"B": 2, "A": 1, "C": 3}
        input_b = {"C": 3, "B": 2, "A": 1}
        
        df_a = preprocessor_no_cols.preprocess(input_a)
        df_b = preprocessor_no_cols.preprocess(input_b)
        
        pd.testing.assert_frame_equal(df_a, df_b)
        # Columns should be sorted alphabetically: A, B, C
        self.assertEqual(list(df_a.columns), ["A", "B", "C"])

if __name__ == "__main__":
    unittest.main()
