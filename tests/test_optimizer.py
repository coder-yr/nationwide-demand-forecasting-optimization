"""Exhaustive Tests for Google OR-Tools Inventory Optimization Solver."""
import numpy as np
import pytest
from src.optimization.solver import InventoryReplenishmentOptimizer

def test_or_tools_solver_feasibility_and_lost_sales():
    """Verifies that the lost-sales MIP formulation produces non-negative inventory and honors MOQ."""
    optimizer = InventoryReplenishmentOptimizer(time_limit_seconds=10)

    item_ids = ["SKU_01", "SKU_02"]
    # High demand that will exhaust starting inventory to test lost sales
    forecast = np.array([
        [20.0, 30.0, 40.0],
        [15.0, 25.0, 35.0]
    ])
    unit_costs = np.array([5.0, 8.0])
    sell_prices = np.array([12.0, 18.0])
    initial_inv = np.array([10.0, 5.0])
    lead_times = np.array([1, 1])
    moqs = np.array([6, 12])
    unit_vols = np.array([0.02, 0.03])
    max_budget = 1000.0
    max_vol = 50.0

    res = optimizer.solve_replenishment(
        item_ids=item_ids,
        forecast_demands=forecast,
        unit_costs=unit_costs,
        sell_prices=sell_prices,
        initial_inventories=initial_inv,
        lead_times=lead_times,
        moqs=moqs,
        unit_volumes=unit_vols,
        max_storage_volume=max_vol,
        budget_per_day=max_budget,
        horizon_days=3
    )

    assert res["status"] in ("OPTIMAL", "FEASIBLE")
    orders = res["orders_matrix"]
    stock = res["projected_inventory"]
    lost_sales = res["projected_lost_sales"]

    # 1. Non-negative physical inventory
    assert (stock < 0.0).sum() == 0, "Negative physical shelf inventory detected!"

    # 2. Non-negative lost sales
    assert (lost_sales < 0.0).sum() == 0, "Negative lost sales detected!"

    # 3. Integer ordering quantities
    assert np.issubdtype(orders.dtype, np.integer), "Orders matrix must be integer type!"

    # 4. Strict MOQ enforcement
    for i in range(2):
        for t in range(3):
            if orders[i, t] > 0:
                assert orders[i, t] >= moqs[i], f"Order {orders[i, t]} is below MOQ {moqs[i]}!"

    # 5. Daily budget ceiling
    for t in range(3):
        spend_t = np.sum(orders[:, t] * unit_costs)
        assert spend_t <= max_budget + 1e-3, f"Day {t} spend ${spend_t} exceeded budget ${max_budget}!"
