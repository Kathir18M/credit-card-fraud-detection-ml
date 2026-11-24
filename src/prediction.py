"""
Prediction Module for Credit Card Fraud Detection.

Handles model loading with cross-version unpickling compatibility, single-row
risk scoring, batch inference, threshold-based classification, and risk tiering.
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple, Optional, List
from collections import UserList

import joblib
import numpy as np
import pandas as pd

from src.validation import (
    validate_features,
    validate_dataframe,
    EXPECTED_FEATURES,
    NUMERIC_SCALED_FEATURES,
    ValidationError,
)

logger = logging.getLogger(__name__)

# Register compatibility shim for scikit-learn unpickling across versions
try:
    import sklearn.compose._column_transformer as ct
    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(UserList):
            """Compatibility stub for unpickling ColumnTransformer across sklearn versions."""
            pass
        ct._RemainderColsList = _RemainderColsList
except Exception as e:
    logger.debug(f"Column transformer compatibility check: {e}")

# Global cached artifacts
_CACHED_MODEL = None
_CACHED_THRESHOLD = None


def get_default_artifact_paths() -> Tuple[str, str]:
    """Return default absolute paths for model and threshold artifacts."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "artifacts", "fraud_detection_model.pkl")
    thresh_path = os.path.join(base_dir, "artifacts", "fraud_threshold.pkl")
    return model_path, thresh_path


def load_model_artifacts(
    model_path: Optional[str] = None,
    threshold_path: Optional[str] = None,
    force_reload: bool = False,
) -> Tuple[Any, float]:
    """
    Load the trained pipeline model and optimal classification threshold.

    Parameters
    ----------
    model_path : str, optional
        Path to the fraud detection model pickle file.
    threshold_path : str, optional
        Path to the fraud threshold pickle file.
    force_reload : bool, default False
        Whether to bypass memory caching and reload artifacts from disk.

    Returns
    -------
    Tuple[Any, float]
        Loaded scikit-learn Pipeline and tuned decision threshold (float).

    Raises
    ------
    FileNotFoundError
        If either artifact file is missing from the designated path.
    RuntimeError
        If unpickling fails.
    """
    global _CACHED_MODEL, _CACHED_THRESHOLD

    if not force_reload and _CACHED_MODEL is not None and _CACHED_THRESHOLD is not None:
        return _CACHED_MODEL, _CACHED_THRESHOLD

    default_model, default_thresh = get_default_artifact_paths()
    m_path = model_path or default_model
    t_path = threshold_path or default_thresh

    if not os.path.exists(m_path):
        raise FileNotFoundError(
            f"Model artifact not found at: {m_path}. "
            "Please ensure 'artifacts/fraud_detection_model.pkl' exists."
        )

    if not os.path.exists(t_path):
        raise FileNotFoundError(
            f"Threshold artifact not found at: {t_path}. "
            "Please ensure 'artifacts/fraud_threshold.pkl' exists."
        )

    try:
        model = joblib.load(m_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {m_path}: {exc}") from exc

    try:
        threshold_val = joblib.load(t_path)
        threshold = float(threshold_val)
    except Exception as exc:
        raise RuntimeError(f"Failed to load threshold from {t_path}: {exc}") from exc

    _CACHED_MODEL = model
    _CACHED_THRESHOLD = threshold
    return model, threshold


def determine_risk_level(probability: float, threshold: float) -> Tuple[str, str]:
    """
    Determine categorical risk level and badge color code based on fraud probability.

    Parameters
    ----------
    probability : float
        Predicted probability of fraud [0.0, 1.0].
    threshold : float
        Optimal classification decision threshold.

    Returns
    -------
    Tuple[str, str]
        (Risk Level Label, Hex Color Code)
    """
    if probability >= 0.70:
        return "Critical Risk", "#DC2626"   # Deep Red
    elif probability >= threshold:
        return "High Risk", "#EA580C"       # Vivid Orange
    elif probability >= 0.15:
        return "Medium Risk", "#CA8A04"     # Amber / Gold
    else:
        return "Low Risk", "#16A34A"        # Green


def predict_single_transaction(
    features: Dict[str, Any],
    threshold: Optional[float] = None,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Validate and score a single transaction input.

    Parameters
    ----------
    features : dict
        Dictionary of raw features (Time, V1..V28, Amount).
    threshold : float, optional
        Custom decision threshold. If None, uses the artifact threshold.
    model : Pipeline, optional
        Pre-loaded model. If None, uses cached loaded model.

    Returns
    -------
    dict
        Dictionary containing prediction results, probabilities, risk level,
        and threshold context.
    """
    validated_df = validate_features(features)

    if model is None or threshold is None:
        loaded_model, loaded_threshold = load_model_artifacts()
        if model is None:
            model = loaded_model
        if threshold is None:
            threshold = loaded_threshold

    # Calculate probabilities
    probs = model.predict_proba(validated_df)[0]
    prob_legit = float(probs[0])
    prob_fraud = float(probs[1])

    is_fraud = bool(prob_fraud >= threshold)
    risk_level, risk_color = determine_risk_level(prob_fraud, threshold)

    explanation = (
        "Statistical machine learning prediction based on historical feature anomalies. "
        "This is an automated probability assessment, not an authoritative fraud finding. "
        "Transactions flagged as High or Critical Risk should be queued for operational human review."
    )

    return {
        "fraud_probability": round(prob_fraud, 6),
        "legitimate_probability": round(prob_legit, 6),
        "predicted_class": 1 if is_fraud else 0,
        "is_fraud": is_fraud,
        "classification": "Fraudulent" if is_fraud else "Legitimate",
        "threshold_used": round(threshold, 4),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "explanation": explanation,
    }


def predict_batch_transactions(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
    model: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Validate, preprocess, and perform batch predictions on an uploaded dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Uploaded batch DataFrame.
    threshold : float, optional
        Classification threshold. Defaults to loaded artifact threshold.
    model : Pipeline, optional
        Trained model pipeline. Defaults to cached loaded artifact.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any], List[str]]
        - Result DataFrame with original features, Fraud_Probability, Predicted_Class, Risk_Level
        - Summary metrics dictionary
        - Validation warnings list
    """
    clean_df, warnings = validate_dataframe(df)

    if model is None or threshold is None:
        loaded_model, loaded_threshold = load_model_artifacts()
        if model is None:
            model = loaded_model
        if threshold is None:
            threshold = loaded_threshold

    probabilities = model.predict_proba(clean_df)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    risk_levels: List[str] = []
    for p in probabilities:
        lvl, _ = determine_risk_level(float(p), threshold)
        risk_levels.append(lvl)

    result_df = clean_df.copy()
    result_df["Fraud_Probability"] = np.round(probabilities, 6)
    result_df["Predicted_Class"] = predictions
    result_df["Risk_Level"] = risk_levels

    # If the user included Actual Class or other columns, retain them for convenience
    for col in df.columns:
        if col not in result_df.columns:
            result_df[col] = df[col].values

    total_count = len(result_df)
    fraud_count = int(predictions.sum())
    legit_count = total_count - fraud_count
    fraud_rate = (fraud_count / total_count * 100) if total_count > 0 else 0.0

    summary = {
        "total_transactions": total_count,
        "predicted_fraud": fraud_count,
        "predicted_legitimate": legit_count,
        "fraud_rate_percentage": round(fraud_rate, 4),
        "threshold_used": round(threshold, 4),
        "mean_fraud_probability": round(float(probabilities.mean()), 6),
        "max_fraud_probability": round(float(probabilities.max()), 6),
        "min_fraud_probability": round(float(probabilities.min()), 6),
    }

    return result_df, summary, warnings
