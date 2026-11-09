"""Unit tests for forecasting evaluation metrics and official 12-level WRMSSE."""
import numpy as np
import pandas as pd
import pytest
from src.utils.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_wape,
    calculate_rmsse,
    pinball_loss,
    M5WRMSSEMetric
)

def test_calculate_mae_and_rmse():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])
    # Absolute errors: [2, 2, 0] -> MAE = 4/3 ~ 1.3333
    mae = calculate_mae(y_true, y_pred)
    assert np.isclose(mae, 4.0 / 3.0)

    # Squared errors: [4, 4, 0] -> Mean = 8/3 -> RMSE = sqrt(8/3) ~ 1.6330
    rmse = calculate_rmse(y_true, y_pred)
    assert np.isclose(rmse, np.sqrt(8.0 / 3.0))

def test_calculate_wape():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])
    wape = calculate_wape(y_true, y_pred)
    assert np.isclose(wape, 4.0 / 60.0, atol=1e-4)

def test_pinball_loss_symmetry():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    loss = pinball_loss(y_true, y_pred, quantile=0.50)
    mae = np.mean(np.abs(y_true - y_pred))
    assert np.isclose(loss, 0.5 * mae)

def test_calculate_rmsse():
    history = np.array([10.0, 12.0, 11.0, 13.0, 12.0])
    y_true = np.array([14.0, 15.0])
    y_pred = np.array([14.0, 15.0])
    rmsse = calculate_rmsse(y_true, y_pred, history)
    assert np.isclose(rmsse, 0.0)

def test_m5_official_wrmsse_calculation():
    """Verifies that M5WRMSSEMetric precomputes scale denominators and scores 12 levels."""
    dates = pd.date_range("2016-01-01", periods=60)
    records = []
    for d in dates:
        d_num = (d - dates[0]).days + 1
        records.append({
            "date": d,
            "d_num": d_num,
            "state_id": "CA",
            "store_id": "CA_1",
            "cat_id": "FOODS",
            "dept_id": "FOODS_1",
            "item_id": "FOODS_1_001",
            "sales": float(d_num % 5 + 1),
            "sell_price": 2.50
        })
    df_history = pd.DataFrame(records)

    # Train history: first 40 days; val holdout: last 20 days
    train_df = df_history[df_history["d_num"] <= 40].copy()
    val_df = df_history[df_history["d_num"] > 40].copy()
    val_df["point_pred"] = val_df["sales"]  # Perfect forecast

    metric = M5WRMSSEMetric(train_history_df=train_df)
    score, level_scores = metric.score(val_df, pred_col="point_pred")

    # Perfect forecast must yield 0.0 WRMSSE
    assert np.isclose(score, 0.0, atol=1e-5), f"Expected 0.0 for perfect prediction, got {score}"
    assert len(level_scores) == 12, "WRMSSE must evaluate across all 12 hierarchy levels!"
