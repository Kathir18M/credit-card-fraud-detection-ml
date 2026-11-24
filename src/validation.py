"""
Data Validation Module for Credit Card Fraud Detection.

Provides robust schema, type, null, and range validation for single and batch
transactions before inference.
"""

from typing import Dict, Any, List, Tuple
import logging
import pandas as pd
import numpy as np

# Configure module-level logger without logging sensitive transaction data
logger = logging.getLogger(__name__)

# Expected 30 input features in canonical order
EXPECTED_FEATURES: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
NUMERIC_SCALED_FEATURES: List[str] = ["Time", "Amount"]


class ValidationError(Exception):
    """Custom exception raised when transaction validation fails."""
    pass


def validate_features(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Validate a single transaction input dictionary and return a 1-row DataFrame.

    Parameters
    ----------
    data : dict
        Key-value mapping of feature names to numeric values.

    Returns
    -------
    pd.DataFrame
        Validated 1-row DataFrame with features in canonical order.

    Raises
    ------
    ValidationError
        If missing features, invalid types, NaN values, or invalid ranges are detected.
    """
    if not isinstance(data, dict):
        raise ValidationError(f"Expected input as a dictionary, received {type(data).__name__}.")

    # Check for missing features
    missing = [feat for feat in EXPECTED_FEATURES if feat not in data]
    if missing:
        raise ValidationError(f"Missing required feature(s): {', '.join(missing)}")

    row_data: Dict[str, float] = {}

    for feat in EXPECTED_FEATURES:
        val = data[feat]
        if val is None or pd.isna(val):
            raise ValidationError(f"Feature '{feat}' cannot be null or NaN.")

        try:
            numeric_val = float(val)
        except (ValueError, TypeError):
            raise ValidationError(f"Feature '{feat}' must be numeric, received: {val!r}")

        if np.isinf(numeric_val):
            raise ValidationError(f"Feature '{feat}' contains infinite values.")

        row_data[feat] = numeric_val

    # Domain specific constraints
    if row_data["Amount"] < 0:
        raise ValidationError(f"Transaction 'Amount' cannot be negative ({row_data['Amount']}).")

    if row_data["Time"] < 0:
        raise ValidationError(f"Transaction 'Time' cannot be negative ({row_data['Time']}).")

    # Construct ordered DataFrame
    df = pd.DataFrame([row_data], columns=EXPECTED_FEATURES)
    return df


def validate_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate a batch transaction DataFrame.

    Checks:
    - Non-empty dataframe
    - Presence of all 30 expected feature columns
    - Castability to float64
    - Absence of null/inf values
    - Amount >= 0 and Time >= 0
    - Ignores extra columns (e.g., 'Class') if present.

    Parameters
    ----------
    df : pd.DataFrame
        Uploaded batch DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Cleaned, ordered DataFrame containing only the 30 features,
        plus a list of any warnings encountered.

    Raises
    ------
    ValidationError
        If structural or data validation fails.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValidationError("Input must be a valid pandas DataFrame.")

    if df.empty:
        raise ValidationError("The provided dataset is empty (0 transactions).")

    warnings: List[str] = []

    # Check missing required columns
    missing = [feat for feat in EXPECTED_FEATURES if feat not in df.columns]
    if missing:
        raise ValidationError(
            f"Uploaded file is missing {len(missing)} required feature(s): {', '.join(missing[:5])}"
            + ("..." if len(missing) > 5 else "")
        )

    # Note any extra columns being ignored
    extra = [col for col in df.columns if col not in EXPECTED_FEATURES]
    if extra:
        warnings.append(f"Ignored {len(extra)} non-feature column(s): {', '.join(extra[:3])}")

    # Extract required columns in exact order
    ordered_df = df[EXPECTED_FEATURES].copy()

    # Check for nulls
    null_counts = ordered_df.isnull().sum()
    if null_counts.any():
        bad_cols = null_counts[null_counts > 0].to_dict()
        bad_desc = ", ".join([f"{k} ({v} missing)" for k, v in list(bad_cols.items())[:3]])
        raise ValidationError(f"Dataset contains null values in columns: {bad_desc}")

    # Ensure numeric conversion
    for col in EXPECTED_FEATURES:
        if not pd.api.types.is_numeric_dtype(ordered_df[col]):
            try:
                ordered_df[col] = pd.to_numeric(ordered_df[col])
            except Exception:
                raise ValidationError(f"Column '{col}' contains non-numeric data that cannot be parsed.")

    # Check for infinite values
    inf_mask = np.isinf(ordered_df.values)
    if inf_mask.any():
        raise ValidationError("Dataset contains infinite values (inf or -inf).")

    # Domain constraints
    negative_amounts = (ordered_df["Amount"] < 0).sum()
    if negative_amounts > 0:
        raise ValidationError(f"Found {negative_amounts} transaction(s) with negative Amount.")

    negative_times = (ordered_df["Time"] < 0).sum()
    if negative_times > 0:
        raise ValidationError(f"Found {negative_times} transaction(s) with negative Time.")

    return ordered_df, warnings
