"""Evaluation metrics for hierarchical and probabilistic time series forecasting.

Includes:
- WRMSSE (Weighted Root Mean Squared Scaled Error - official M5 competition metric)
- RMSSE (Root Mean Squared Scaled Error)
- WAPE (Weighted Absolute Percentage Error)
- Pinball Loss (Quantile Loss) for probabilistic forecasts
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union

def calculate_rmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: np.ndarray
) -> float:
    """Calculates RMSSE for a single series.

    Args:
        y_true: Actual values in forecast horizon (length H).
        y_pred: Predicted values in forecast horizon (length H).
        y_train_history: In-sample historical training values (length N).

    Returns:
        float: RMSSE value.
    """
    h = len(y_true)
    if h == 0:
        return 0.0

    # Scale denominator: mean squared successive difference in history
    diffs = np.diff(y_train_history)
    scale = np.mean(diffs ** 2)
    if scale == 0 or np.isnan(scale):
        scale = 1e-4

    numerator = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(numerator / scale))

def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Weighted Absolute Percentage Error (WAPE)."""
    sum_actual = np.sum(np.abs(y_true))
    if sum_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / sum_actual)

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Calculates the pinball (quantile) loss for a given quantile alpha in (0, 1)."""
    error = y_true - y_pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))

def calculate_multi_quantile_loss(
    y_true: np.ndarray,
    quantile_predictions: Dict[float, np.ndarray]
) -> Dict[str, float]:
    """Computes pinball loss across multiple quantiles (e.g. 0.1, 0.5, 0.9)."""
    results = {}
    losses = []
    for q, pred in quantile_predictions.items():
        loss = pinball_loss(y_true, pred, q)
        results[f"pinball_loss_q{int(q*100):02d}"] = loss
        losses.append(loss)
    results["mean_pinball_loss"] = float(np.mean(losses))
    return results

