"""Leakage-free lag, rolling statistics, and intermittent demand feature generator.

Ensures minimum 28-day shift to allow direct 28-day operational forecasting horizons
without recursive autoregressive data leakage.
"""
from typing import List, Optional
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("lag_generator")

def generate_lag_features(
    df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    min_shift: int = 28
) -> pd.DataFrame:
    """Generates lag and rolling features grouped by item_id within a store partition.

    All features use a strict forecast-origin boundary: any observation at time t
    only aggregates information from <= t - min_shift.

    Args:
        df: DataFrame sorted by item_id and d_num / date.
        lags: Explicit lag offsets (all >= min_shift).
              Defaults to [28, 29, 30, 35, 42, 56, 84, 364].
        rolling_windows: Windows for rolling mean/std. Defaults to [7, 14, 28, 56].
        min_shift: Base minimum shift for rolling windows to prevent leakage (default 28).

    Returns:
        pd.DataFrame: Augmented DataFrame with lag & rolling columns.
    """
    if lags is None:
        lags = [28, 29, 30, 35, 42, 56, 84, 364]
    if rolling_windows is None:
        rolling_windows = [7, 14, 28, 56]

    logger.info(f"Generating lag features: lags={lags}, rolling_windows={rolling_windows}, min_shift={min_shift}")

    # Ensure strictly sorted by item and sequence
    df = df.sort_values(["item_id", "d_num"]).reset_index(drop=True)
    grouped = df.groupby("item_id")["sales"]

    # 1. Direct shifted lags
    for lag in lags:
        assert lag >= min_shift, f"Lag {lag} violates minimum safe shift of {min_shift} days!"
        df[f"sales_lag_{lag}"] = grouped.shift(lag).astype(np.float32)

    # 2. Shifted rolling statistics (shift by min_shift FIRST, then apply rolling)
    base_shifted = grouped.shift(min_shift)
    for w in rolling_windows:
        roll = base_shifted.groupby(df["item_id"]).rolling(window=w, min_periods=1)
        df[f"roll_mean_{w}"] = roll.mean().reset_index(drop=True).astype(np.float32)
        df[f"roll_std_{w}"] = roll.std().reset_index(drop=True).fillna(0.0).astype(np.float32)

    # 3. Shifted Exponential Moving Average (EWMA)
    for alpha in [0.1, 0.3]:
        df[f"ewma_{int(alpha*100):02d}"] = (
            base_shifted.groupby(df["item_id"])
            .apply(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
            .reset_index(drop=True)
            .astype(np.float32)
        )

    return df
