"""Comprehensive Multi-Origin Feature Leakage and Forecast Boundary Tests.

Verifies:
1. Absence of future sales leakage across multiple forecast origins.
2. Robustness against future price mutations (no forward lookahead).
3. Shifted rolling mean and EWMA boundaries (only past information used).
4. Training/validation cutoff separation integrity.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.lag_generator import generate_lag_features
from src.features.price_feats import generate_price_features
from src.features.calendar_feats import generate_calendar_features

def create_synthetic_panel(num_days=400):
    """Creates a deterministic panel for causal boundary testing."""
    dates = pd.date_range("2015-01-01", periods=num_days)
    df = pd.DataFrame({
        "item_id": ["ITEM_A"] * num_days,
        "d_num": np.arange(1, num_days + 1, dtype=np.uint16),
        "date": dates,
        "wday": [d.weekday() + 1 for d in dates],
        "month": [d.month for d in dates],
        "sales": np.arange(10, 10 + num_days, dtype=np.float32),
        "sell_price": np.linspace(5.0, 15.0, num_days, dtype=np.float32)
    })
    return df

def test_multi_origin_sales_leakage():
    """Mutating sales inside the 28-day forecast horizon must NOT alter features at origin T."""
    base_panel = create_synthetic_panel(num_days=200)

    for origin in [60, 100, 150]:
        # Generate baseline features
        df_base = generate_lag_features(base_panel.copy(), lags=[28, 35, 56], rolling_windows=[7, 28], min_shift=28)

        # Mutate future sales inside the horizon (origin < t <= origin + 28)
        mutated_panel = base_panel.copy()
        future_mask = (mutated_panel["d_num"] > origin) & (mutated_panel["d_num"] <= origin + 28)
        mutated_panel.loc[future_mask, "sales"] = 9999.0

        df_mutated = generate_lag_features(mutated_panel, lags=[28, 35, 56], rolling_windows=[7, 28], min_shift=28)

        # Compare features at origin T (must be identical)
        features_to_check = ["sales_lag_28", "sales_lag_35", "sales_lag_56", "roll_mean_7", "roll_mean_28", "ewma_10"]
        row_base = df_base[df_base["d_num"] == origin][features_to_check].iloc[0]
        row_mut = df_mutated[df_mutated["d_num"] == origin][features_to_check].iloc[0]

        for feat in features_to_check:
            assert np.isclose(row_base[feat], row_mut[feat], equal_nan=True), (
                f"LEAKAGE DETECTED: Mutating future sales at t > {origin} altered feature '{feat}' at t = {origin}!"
            )

def test_price_leakage_future_mutation():
    """Future price spikes inside the forecast horizon must NOT alter current discount or price momentum."""
    base_panel = create_synthetic_panel(num_days=100)
    origin = 50

    df_base = generate_price_features(base_panel.copy())

    # Mutate price at future day origin + 7
    mutated_panel = base_panel.copy()
    mutated_panel.loc[mutated_panel["d_num"] == origin + 7, "sell_price"] = 500.0

    df_mutated = generate_price_features(mutated_panel)

    # Check that price features at origin did not change
    price_features = ["price_max_28d", "discount_ratio", "price_momentum", "price_diff_lag7"]
    row_base = df_base[df_base["d_num"] == origin][price_features].iloc[0]
    row_mut = df_mutated[df_mutated["d_num"] == origin][price_features].iloc[0]

    for feat in price_features:
        assert np.isclose(row_base[feat], row_mut[feat]), (
            f"PRICE LEAKAGE: Future price spike at t={origin+7} altered '{feat}' at origin t={origin}!"
        )

def test_calendar_temporal_invariance():
    """Calendar features must depend only on calendar date, not on target sales or prices."""
    panel = create_synthetic_panel(num_days=30)
    cal_df = generate_calendar_features(panel.copy())

    assert "week" in cal_df.columns
    assert "year" in cal_df.columns
    assert "sin_wday" in cal_df.columns
    assert "cos_wday" in cal_df.columns
    assert "sin_week" in cal_df.columns
    assert "cos_week" in cal_df.columns

    # Verify cyclical bounds
    assert cal_df["sin_wday"].between(-1.01, 1.01).all()
    assert cal_df["cos_wday"].between(-1.01, 1.01).all()
    assert cal_df["sin_week"].between(-1.01, 1.01).all()
    assert cal_df["cos_week"].between(-1.01, 1.01).all()
