"""
Comprehensive Quality Validation Suite for Credit Card Fraud Detection System.
Executes systematic unit and integration tests across artifacts, schemas, predictions, and batch operations.
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.validation import (
    validate_features,
    validate_dataframe,
    ValidationError,
    EXPECTED_FEATURES,
)
from src.prediction import (
    load_model_artifacts,
    predict_single_transaction,
    predict_batch_transactions,
    determine_risk_level,
)


class TestFraudDetectionSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load artifacts once for the test suite."""
        cls.model, cls.threshold = load_model_artifacts()
        cls.valid_sample = {f"V{i}": 0.0 for i in range(1, 29)}
        cls.valid_sample["Time"] = 100.0
        cls.valid_sample["Amount"] = 45.50

    def test_01_artifact_loading(self):
        """Verify model artifact and threshold load successfully."""
        self.assertIsNotNone(self.model, "Model should not be None")
        self.assertIsInstance(self.threshold, float, "Threshold should be a float")
        self.assertAlmostEqual(self.threshold, 0.30, places=2, msg="Threshold should be 0.30")

    def test_02_expected_features_count(self):
        """Check all 30 input features are accounted for."""
        self.assertEqual(len(EXPECTED_FEATURES), 30, "There must be exactly 30 expected features")
        self.assertIn("Time", EXPECTED_FEATURES)
        self.assertIn("Amount", EXPECTED_FEATURES)
        for i in range(1, 29):
            self.assertIn(f"V{i}", EXPECTED_FEATURES)

    def test_03_valid_single_transaction_prediction(self):
        """Test prediction with a valid transaction input dictionary."""
        result = predict_single_transaction(self.valid_sample, model=self.model, threshold=self.threshold)
        self.assertIn("fraud_probability", result)
        self.assertIn("predicted_class", result)
        self.assertIn("risk_level", result)
        self.assertIn("threshold_used", result)
        self.assertIsInstance(result["fraud_probability"], float)
        self.assertIn(result["predicted_class"], [0, 1])
        self.assertTrue(0.0 <= result["fraud_probability"] <= 1.0)

    def test_04_missing_columns_validation(self):
        """Test that missing required columns raise ValidationError."""
        broken_sample = self.valid_sample.copy()
        del broken_sample["Amount"]
        with self.assertRaises(ValidationError):
            validate_features(broken_sample)

    def test_05_incorrect_column_types(self):
        """Test that non-numeric types raise ValidationError."""
        broken_sample = self.valid_sample.copy()
        broken_sample["V1"] = "invalid_string_input"
        with self.assertRaises(ValidationError):
            validate_features(broken_sample)

    def test_06_negative_amount_and_time(self):
        """Test that negative amounts or times raise ValidationError."""
        neg_amount = self.valid_sample.copy()
        neg_amount["Amount"] = -15.0
        with self.assertRaises(ValidationError):
            validate_features(neg_amount)

        neg_time = self.valid_sample.copy()
        neg_time["Time"] = -5.0
        with self.assertRaises(ValidationError):
            validate_features(neg_time)

    def test_07_null_and_nan_values(self):
        """Test that NaN values raise ValidationError."""
        nan_sample = self.valid_sample.copy()
        nan_sample["V5"] = np.nan
        with self.assertRaises(ValidationError):
            validate_features(nan_sample)

    def test_08_batch_dataframe_validation_and_prediction(self):
        """Test batch dataframe upload validation and scoring."""
        # Create a mock batch of 10 rows
        data = {f"V{i}": np.random.randn(10) for i in range(1, 29)}
        data["Time"] = np.linspace(10, 100, 10)
        data["Amount"] = np.random.uniform(5, 500, 10)
        df = pd.DataFrame(data)

        # Batch prediction
        result_df, summary, warnings = predict_batch_transactions(df, model=self.model, threshold=self.threshold)
        self.assertEqual(len(result_df), 10)
        self.assertEqual(summary["total_transactions"], 10)
        self.assertIn("Fraud_Probability", result_df.columns)
        self.assertIn("Predicted_Class", result_df.columns)
        self.assertIn("Risk_Level", result_df.columns)

    def test_09_batch_extra_columns_ignored_gracefully(self):
        """Verify that extra columns (such as Class) are tolerated and ignored."""
        df = pd.DataFrame([self.valid_sample])
        df["Class"] = 0
        df["Extra_Meta"] = "User123"

        result_df, summary, warnings = predict_batch_transactions(df, model=self.model, threshold=self.threshold)
        self.assertEqual(len(result_df), 1)
        self.assertIn("Class", result_df.columns)
        self.assertIn("Extra_Meta", result_df.columns)

    def test_10_risk_tiering(self):
        """Test categorization of risk levels."""
        lvl_low, _ = determine_risk_level(0.05, 0.30)
        self.assertEqual(lvl_low, "Low Risk")

        lvl_med, _ = determine_risk_level(0.20, 0.30)
        self.assertEqual(lvl_med, "Medium Risk")

        lvl_high, _ = determine_risk_level(0.45, 0.30)
        self.assertEqual(lvl_high, "High Risk")

        lvl_crit, _ = determine_risk_level(0.85, 0.30)
        self.assertEqual(lvl_crit, "Critical Risk")

    def test_11_outputs_files_presence(self):
        """Verify that all required output files exist and are populated."""
        for filename in ["model_comparison.csv", "feature_importance.csv", "threshold_analysis.csv", "test_predictions.csv"]:
            path = os.path.join("outputs", filename)
            self.assertTrue(os.path.exists(path), f"File {path} must exist")
            self.assertGreater(os.path.getsize(path), 0, f"File {path} must not be empty")


if __name__ == "__main__":
    unittest.main()
