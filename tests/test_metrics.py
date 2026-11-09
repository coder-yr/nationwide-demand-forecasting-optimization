"""Unit tests for forecasting evaluation metrics."""
import numpy as np
import pytest
from src.utils.metrics import calculate_rmsse, calculate_wape, pinball_loss

def test_calculate_wape():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])
    # Absolute errors: 2 + 2 + 0 = 4. Sum true = 60. WAPE = 4 / 60 ~ 0.0667
    wape = calculate_wape(y_true, y_pred)
    assert np.isclose(wape, 4.0 / 60.0, atol=1e-4)

def test_pinball_loss_symmetry():
    # At median (q=0.5), pinball loss is 0.5 * MAE
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    loss = pinball_loss(y_true, y_pred, quantile=0.50)
    mae = np.mean(np.abs(y_true - y_pred))
    assert np.isclose(loss, 0.5 * mae)

def test_calculate_rmsse():
    history = np.array([10.0, 12.0, 11.0, 13.0, 12.0])
    y_true = np.array([14.0, 15.0])
    y_pred = np.array([14.0, 15.0])
    # Perfect forecast gives 0.0
    rmsse = calculate_rmsse(y_true, y_pred, history)
    assert np.isclose(rmsse, 0.0)

