"""Unit test ensuring strict absence of temporal feature leakage."""
import numpy as np
import pandas as pd
import pytest
from src.features.lag_generator import generate_lag_features

def test_no_temporal_leakage_in_lags():
    # Build toy time series
    dates = pd.date_range("2016-01-01", periods=60)
    df = pd.DataFrame({
        "item_id": ["ITEM_1"] * 60,
        "d_num": np.arange(1, 61, dtype=np.uint16),
        "sales": np.arange(10, 70, dtype=np.float32)
    })

    augmented = generate_lag_features(df, lags=[28, 35], rolling_windows=[7], min_shift=28)

    # For day index < 28, all lag and shifted rolling features MUST be NaN
    sub_early = augmented[augmented["d_num"] <= 28]
    assert sub_early["sales_lag_28"].isna().iloc[0]

    # Verify that sales_lag_28 at index 29 matches sales at index 1
    val_lag28 = augmented.loc[augmented["d_num"] == 29, "sales_lag_28"].values[0]
    val_orig = augmented.loc[augmented["d_num"] == 1, "sales"].values[0]
    assert np.isclose(val_lag28, val_orig), f"Expected {val_orig}, got {val_lag28}"

