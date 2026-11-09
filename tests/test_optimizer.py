"""Unit tests for OR-Tools inventory optimization solver."""
import numpy as np
import pytest
from src.optimization.solver import InventoryReplenishmentOptimizer

def test_or_tools_solver_feasibility():
    optimizer = InventoryReplenishmentOptimizer(time_limit_seconds=10)

    item_ids = ["SKU_01", "SKU_02"]
    forecast = np.array([
        [10.0, 15.0, 20.0],
        [5.0, 10.0, 12.0]
    ])
    unit_costs = np.array([5.0, 8.0])
    sell_prices = np.array([12.0, 18.0])
    initial_inv = np.array([10.0, 5.0])
    lead_times = np.array([1, 1])
    moqs = np.array([6, 6])
    unit_vols = np.array([0.02, 0.03])

    res = optimizer.solve_replenishment(
        item_ids=item_ids,
        forecast_demands=forecast,
        unit_costs=unit_costs,
        sell_prices=sell_prices,
        initial_inventories=initial_inv,
        lead_times=lead_times,
        moqs=moqs,
        unit_volumes=unit_vols,
        horizon_days=3
    )

    assert res["status"] in ("OPTIMAL", "FEASIBLE")
    orders = res["orders_matrix"]
    assert orders.shape == (2, 3)

    # Check MOQ constraint: whenever an order is positive, it must be >= MOQ
    for i in range(2):
        for t in range(3):
            if orders[i, t] > 0.0:
                assert orders[i, t] >= moqs[i]
