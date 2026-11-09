"""Evaluation metrics for hierarchical and probabilistic time series forecasting.

Includes:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- WAPE (Weighted Absolute Percentage Error)
- RMSSE (Root Mean Squared Scaled Error)
- Pinball Loss (Quantile Loss)
- Official M5 Competition WRMSSE (Weighted Root Mean Squared Scaled Error across 12 levels)
"""
from typing import Dict, List, Union, Optional, Tuple, Any
import numpy as np
import pandas as pd

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (WAPE)."""
    sum_actual = np.sum(np.abs(y_true))
    if sum_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / sum_actual)

def calculate_rmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: np.ndarray
) -> float:
    """Calculates RMSSE for a single series.

    Scale denominator: mean squared successive difference in in-sample history.
    """
    h = len(y_true)
    if h == 0:
        return 0.0

    diffs = np.diff(y_train_history)
    scale = np.mean(diffs ** 2)
    if scale == 0 or np.isnan(scale):
        scale = 1e-4

    numerator = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(numerator / scale))

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Pinball (quantile) loss for quantile alpha in (0, 1)."""
    error = y_true - y_pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))

def calculate_multi_quantile_loss(
    y_true: np.ndarray,
    quantile_predictions: Dict[float, np.ndarray]
) -> Dict[str, float]:
    """Computes pinball loss across multiple quantiles."""
    results = {}
    losses = []
    for q, pred in quantile_predictions.items():
        loss = pinball_loss(y_true, pred, q)
        results[f"pinball_loss_q{int(q*100):02d}"] = loss
        losses.append(loss)
    results["mean_pinball_loss"] = float(np.mean(losses))
    return results

class M5WRMSSEMetric:
    """Official M5 Competition Weighted Root Mean Squared Scaled Error (WRMSSE).

    Evaluates predictions across the 12 hierarchical aggregation levels:
    Level 1: Total
    Level 2: State
    Level 3: Store
    Level 4: Category
    Level 5: Department
    Level 6: State x Category
    Level 7: State x Department
    Level 8: Store x Category
    Level 9: Store x Department
    Level 10: Item
    Level 11: Item x State
    Level 12: Item x Store (Bottom Level)
    """

    HIERARCHY_LEVELS = [
        [],                                            # Level 1: Total
        ["state_id"],                                  # Level 2: State
        ["store_id"],                                  # Level 3: Store
        ["cat_id"],                                    # Level 4: Category
        ["dept_id"],                                   # Level 5: Department
        ["state_id", "cat_id"],                        # Level 6: State x Category
        ["state_id", "dept_id"],                       # Level 7: State x Department
        ["store_id", "cat_id"],                        # Level 8: Store x Category
        ["store_id", "dept_id"],                       # Level 9: Store x Department
        ["item_id"],                                   # Level 10: Item
        ["item_id", "state_id"],                       # Level 11: Item x State
        ["item_id", "store_id"],                       # Level 12: Item x Store
    ]

    def __init__(self, train_history_df: pd.DataFrame, sell_prices_df: Optional[pd.DataFrame] = None):
        """Precomputes scale denominators and revenue weights from historical training window.

        Args:
            train_history_df: DataFrame containing historical sales with columns
                              ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'date', 'd_num', 'sales', 'sell_price']
        """
        self.train_df = train_history_df.copy()
        self.scale_denominators: Dict[int, Dict[Any, float]] = {}
        self.weights: Dict[int, Dict[Any, float]] = {}
        self._precompute_scales_and_weights()

    def _precompute_scales_and_weights(self):
        """Precomputes in-sample squared differences (scale) and last 28d dollar revenue weights."""
        # Calculate dollar sales for weight calculation (last 28 days of history)
        max_d = self.train_df["d_num"].max()
        last_28d = self.train_df[self.train_df["d_num"] > max_d - 28].copy()
        last_28d["dollar_sales"] = last_28d["sales"] * last_28d["sell_price"]
        total_dollars = last_28d["dollar_sales"].sum()
        if total_dollars == 0:
            total_dollars = 1.0

        for level_idx, group_cols in enumerate(self.HIERARCHY_LEVELS, start=1):
            self.scale_denominators[level_idx] = {}
            self.weights[level_idx] = {}

            if len(group_cols) == 0:
                # Level 1: Total
                agg_series = self.train_df.groupby("date")["sales"].sum().values
                diffs = np.diff(agg_series)
                self.scale_denominators[level_idx]["Total"] = float(np.mean(diffs ** 2)) if len(diffs) > 0 else 1.0
                self.weights[level_idx]["Total"] = 1.0 / 12.0  # Equal weight across 12 levels
            else:
                # Levels 2 to 12
                # Precompute scale denominators per series in this level
                grouped = self.train_df.groupby(group_cols)
                level_dollar_sum = last_28d.groupby(group_cols)["dollar_sales"].sum()

                for key, grp in grouped:
                    series = grp.sort_values("date").groupby("date")["sales"].sum().values
                    diffs = np.diff(series)
                    scale = float(np.mean(diffs ** 2)) if len(diffs) > 0 else 1.0
                    self.scale_denominators[level_idx][key] = scale if scale > 0 else 1e-4

                    # Revenue weight for this series
                    dol = level_dollar_sum.get(key, 0.0)
                    self.weights[level_idx][key] = (dol / total_dollars) / 12.0

    def score(self, val_df: pd.DataFrame, pred_col: str = "point_pred") -> Tuple[float, Dict[int, float]]:
        """Calculates official M5 WRMSSE on evaluation holdout dataframe.

        Returns:
            Tuple[float, Dict[int, float]]: (Overall WRMSSE scalar, Dict of per-level WRMSSE)
        """
        level_scores = {}
        total_wrmsse = 0.0

        for level_idx, group_cols in enumerate(self.HIERARCHY_LEVELS, start=1):
            level_rmsse_sum = 0.0
            if len(group_cols) == 0:
                # Level 1 Total
                tot_true = val_df.groupby("date")["sales"].sum().values
                tot_pred = val_df.groupby("date")[pred_col].sum().values
                scale = self.scale_denominators[level_idx]["Total"]
                weight = self.weights[level_idx]["Total"]
                rmsse = np.sqrt(np.mean((tot_true - tot_pred) ** 2) / scale)
                level_scores[level_idx] = float(rmsse)
                total_wrmsse += float(weight * rmsse)
            else:
                grouped_val = val_df.groupby(group_cols)
                for key, grp in grouped_val:
                    y_true = grp.sort_values("date").groupby("date")["sales"].sum().values
                    y_pred = grp.sort_values("date").groupby("date")[pred_col].sum().values
                    scale = self.scale_denominators[level_idx].get(key, 1.0)
                    weight = self.weights[level_idx].get(key, 0.0)
                    if len(y_true) > 0 and scale > 0:
                        rmsse = np.sqrt(np.mean((y_true - y_pred) ** 2) / scale)
                        total_wrmsse += float(weight * rmsse)
                        level_rmsse_sum += float(rmsse)
                level_scores[level_idx] = float(level_rmsse_sum / max(len(grouped_val), 1))

        return float(total_wrmsse), level_scores
