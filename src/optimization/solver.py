"""Google OR-Tools Mixed-Integer Programming (MIP) Inventory Replenishment Solver.

Determines optimal daily purchase orders Q(i, t) across multiple products and horizons,
balancing holding costs, stockout penalties, MOQ constraints, and budget ceilings.
"""
# CRITICAL FOR WINDOWS RUNTIME: ortools must be imported before pandas/pyarrow
import ortools
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("inventory_optimizer")

class InventoryReplenishmentOptimizer:
    """Mathematical optimization solver using Google OR-Tools CBC/SCIP MIP solver."""

    def __init__(
        self,
        holding_cost_rate_annual: float = 0.20,
        stockout_penalty_multiplier: float = 1.5,
        fixed_order_cost: float = 50.0,
        time_limit_seconds: int = 30
    ):
        self.holding_cost_rate_daily = holding_cost_rate_annual / 365.0
        self.stockout_penalty_multiplier = stockout_penalty_multiplier
        self.fixed_order_cost = fixed_order_cost
        self.time_limit_seconds = time_limit_seconds

    def solve_replenishment(
        self,
        item_ids: List[str],
        forecast_demands: np.ndarray,       # shape: (n_items, horizon_days)
        unit_costs: np.ndarray,             # shape: (n_items,)
        sell_prices: np.ndarray,            # shape: (n_items,)
        initial_inventories: np.ndarray,    # shape: (n_items,)
        lead_times: np.ndarray,             # shape: (n_items,) in days (int)
        moqs: np.ndarray,                   # shape: (n_items,)
        unit_volumes: np.ndarray,           # shape: (n_items,) in m3
        max_storage_volume: float = 1500.0,
        budget_per_day: float = 50000.0,
        horizon_days: int = 7
    ) -> Dict[str, Any]:
        """Formulates and solves the multi-period, multi-item MIP problem.

        Args:
            item_ids: List of item identifiers.
            forecast_demands: 2D array of expected demands [n_items, horizon_days].
            unit_costs: Cost to purchase each item unit.
            sell_prices: Selling price per item unit.
            initial_inventories: Starting stock on day 0.
            lead_times: Integer arrival lag per item.
            moqs: Minimum order quantity per item if order is placed.
            unit_volumes: Volume per item in cubic meters.
            max_storage_volume: Total store storage capacity.
            budget_per_day: Daily working capital expenditure limit.
            horizon_days: Days in planning horizon.

        Returns:
            Dict[str, Any]: Solver status, optimal order matrix, and cost breakdown.
        """
        n_items = len(item_ids)
        logger.info(f"Formulating OR-Tools MIP for {n_items} items over {horizon_days}-day horizon...")

        # Create CBC / SCIP linear solver
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            solver = pywraplp.Solver.CreateSolver("CBC")
        if not solver:
            raise RuntimeError("Neither SCIP nor CBC solver available in Google OR-Tools installation.")

        solver.SetTimeLimit(self.time_limit_seconds * 1000)

        # Decision variables
        # Q[i, t]: quantity ordered on day t
        # Y[i, t]: binary flag (1 if order placed, 0 otherwise)
        # I_pos[i, t]: inventory on hand at end of day t
        # I_neg[i, t]: unmet demand / lost sales on day t
        Q = {}
        Y = {}
        I_pos = {}
        I_neg = {}

        big_M = 10000.0

        for i in range(n_items):
            for t in range(horizon_days):
                Q[i, t] = solver.NumVar(0.0, big_M, f"Q_{i}_{t}")
                Y[i, t] = solver.BoolVar(f"Y_{i}_{t}")
                I_pos[i, t] = solver.NumVar(0.0, big_M, f"Ipos_{i}_{t}")
                I_neg[i, t] = solver.NumVar(0.0, big_M, f"Ineg_{i}_{t}")

        # Constraints
        for i in range(n_items):
            lt = int(lead_times[i])
            moq = float(moqs[i])

            for t in range(horizon_days):
                # 1. MOQ and Order Binary Flag
                # Q >= MOQ * Y
                # Q <= M * Y
                solver.Add(Q[i, t] >= moq * Y[i, t])
                solver.Add(Q[i, t] <= big_M * Y[i, t])

                # 2. Inventory Balance
                # Previous inventory
                prev_inv = initial_inventories[i] if t == 0 else (I_pos[i, t - 1] - I_neg[i, t - 1])

                # Arriving orders placed at (t - lead_time)
                arriving_order = Q[i, t - lt] if (t - lt) >= 0 else 0.0
                demand_t = float(forecast_demands[i, t])

                # Balance equation: I_pos[i, t] - I_neg[i, t] == prev_inv + arriving_order - demand_t
                solver.Add(I_pos[i, t] - I_neg[i, t] == prev_inv + arriving_order - demand_t)

        # 3. Capacity & Budget Constraints per day
        for t in range(horizon_days):
            # Volume capacity
            vol_expr = solver.Sum(unit_volumes[i] * I_pos[i, t] for i in range(n_items))
            solver.Add(vol_expr <= max_storage_volume)

            # Daily budget
            budget_expr = solver.Sum(unit_costs[i] * Q[i, t] for i in range(n_items))
            solver.Add(budget_expr <= budget_per_day)

        # Objective Function: Minimize Total Cost
        obj_expr = []
        for i in range(n_items):
            c_unit = float(unit_costs[i])
            h_cost = c_unit * self.holding_cost_rate_daily
            margin = max(float(sell_prices[i]) - c_unit, 0.5)
            shortage_pen = margin * self.stockout_penalty_multiplier

            for t in range(horizon_days):
                # Fixed setup cost
                obj_expr.append(self.fixed_order_cost * Y[i, t])
                # Unit purchasing cost
                obj_expr.append(c_unit * Q[i, t])
                # Holding cost
                obj_expr.append(h_cost * I_pos[i, t])
                # Stockout shortage penalty
                obj_expr.append(shortage_pen * I_neg[i, t])

        solver.Minimize(solver.Sum(obj_expr))

        # Solve
        logger.info("Executing OR-Tools optimization solve...")
        status = solver.Solve()

        is_optimal = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
        status_name = "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else ("FEASIBLE" if status == pywraplp.Solver.FEASIBLE else "INFEASIBLE")

        logger.info(f"Solver completed with status: {status_name}, Objective Value = ${solver.Objective().Value():,.2f}" if is_optimal else f"Solver failed: {status_name}")

        orders_matrix = np.zeros((n_items, horizon_days), dtype=np.float32)
        projected_stock = np.zeros((n_items, horizon_days), dtype=np.float32)
        projected_stockouts = np.zeros((n_items, horizon_days), dtype=np.float32)

        if is_optimal:
            for i in range(n_items):
                for t in range(horizon_days):
                    orders_matrix[i, t] = round(Q[i, t].solution_value())
                    projected_stock[i, t] = round(I_pos[i, t].solution_value())
                    projected_stockouts[i, t] = round(I_neg[i, t].solution_value())

        return {
            "status": status_name,
            "objective_value": solver.Objective().Value() if is_optimal else None,
            "orders_matrix": orders_matrix,
            "projected_inventory": projected_stock,
            "projected_stockouts": projected_stockouts,
            "item_ids": item_ids,
            "horizon_days": horizon_days
        }
