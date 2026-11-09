"""Google OR-Tools Mixed-Integer Programming (MIP) Inventory Replenishment Solver.

Determines optimal daily purchase orders Q(i, t) across multiple products and horizons,
balancing holding costs, lost-sales penalties, MOQ constraints, and budget ceilings.
"""
# CRITICAL FOR WINDOWS RUNTIME: ortools must be loaded before pandas/pyarrow
import ortools
from ortools.linear_solver import pywraplp

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("inventory_optimizer")

class InventoryReplenishmentOptimizer:
    """Mathematical optimization solver using Google OR-Tools CBC/SCIP MIP solver.

    Business Model: Retail Hypermarket Lost-Sales Formulation
    -------------------------------------------------------
    In this model, unmet demand is LOST (customer departs or purchases alternative SKU).
    Starting inventory for period t is strictly non-negative:
        Available Stock = I_pos[i, t-1] + Arriving Orders
        Sales[i, t] = min(Demand, Available Stock)
        Ending Inventory I_pos[i, t] = Available Stock - Sales[i, t] >= 0
        Lost Sales I_neg[i, t] = Demand[i, t] - Sales[i, t] >= 0
    Unsatisfied demand does NOT accumulate as negative backorders.
    """

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
        """Formulates and solves the lost-sales integer replenishment MIP problem."""
        n_items = len(item_ids)
        logger.info(f"Formulating OR-Tools MIP (Lost Sales) for {n_items} items over {horizon_days}-day horizon...")

        # Create CBC / SCIP linear solver
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            solver = pywraplp.Solver.CreateSolver("CBC")
        if not solver:
            raise RuntimeError("Neither SCIP nor CBC solver available in Google OR-Tools installation.")

        solver.SetTimeLimit(self.time_limit_seconds * 1000)

        # Decision variables
        # Q[i, t]: integer order quantity placed on day t
        # Y[i, t]: binary order placement indicator (1 if Q > 0, 0 otherwise)
        # S[i, t]: realized sales on day t (continuous/integer bounded)
        # I_pos[i, t]: ending physical shelf inventory on day t (non-negative)
        # I_neg[i, t]: lost sales on day t (non-negative)
        Q = {}
        Y = {}
        S = {}
        I_pos = {}
        I_neg = {}

        max_order_qty = 5000

        for i in range(n_items):
            for t in range(horizon_days):
                Q[i, t] = solver.IntVar(0, max_order_qty, f"Q_{i}_{t}")
                Y[i, t] = solver.BoolVar(f"Y_{i}_{t}")
                S[i, t] = solver.NumVar(0.0, max_order_qty, f"Sales_{i}_{t}")
                I_pos[i, t] = solver.NumVar(0.0, max_order_qty, f"Ipos_{i}_{t}")
                I_neg[i, t] = solver.NumVar(0.0, max_order_qty, f"LostSales_{i}_{t}")

        # Constraints
        for i in range(n_items):
            lt = int(lead_times[i])
            moq = int(moqs[i])

            for t in range(horizon_days):
                # 1. MOQ and Binary Order Setup Constraints
                # Q >= MOQ * Y
                # Q <= M * Y
                solver.Add(Q[i, t] >= moq * Y[i, t])
                solver.Add(Q[i, t] <= max_order_qty * Y[i, t])

                # 2. Available stock at beginning of period t
                # Beginning stock = ending physical inventory of day t-1 (strictly >= 0)
                # Arriving orders were placed at (t - lead_time)
                arriving_order = Q[i, t - lt] if (t - lt) >= 0 else 0
                if t == 0:
                    avail_stock = float(initial_inventories[i]) + arriving_order
                else:
                    avail_stock = I_pos[i, t - 1] + arriving_order

                demand_t = float(forecast_demands[i, t])

                # 3. RETAIL LOST-SALES FORMULATION:
                # Sales cannot exceed available stock
                solver.Add(S[i, t] <= avail_stock)
                # Sales cannot exceed demand
                solver.Add(S[i, t] <= demand_t)

                # Physical ending inventory balance: I_pos = available stock - sales
                solver.Add(I_pos[i, t] == avail_stock - S[i, t])

                # Lost sales definition: I_neg = demand - sales
                solver.Add(I_neg[i, t] == demand_t - S[i, t])

        # 4. Physical Volume Capacity & Working Capital Budget per day
        for t in range(horizon_days):
            vol_expr = solver.Sum(float(unit_volumes[i]) * I_pos[i, t] for i in range(n_items))
            solver.Add(vol_expr <= max_storage_volume)

            budget_expr = solver.Sum(float(unit_costs[i]) * Q[i, t] for i in range(n_items))
            solver.Add(budget_expr <= budget_per_day)

        # Objective Function: Minimize Total Operational Cost
        obj_expr = []
        for i in range(n_items):
            c_unit = float(unit_costs[i])
            h_cost = c_unit * self.holding_cost_rate_daily
            margin = max(float(sell_prices[i]) - c_unit, 0.5)
            shortage_pen = margin * self.stockout_penalty_multiplier

            for t in range(horizon_days):
                # Fixed order setup cost
                obj_expr.append(self.fixed_order_cost * Y[i, t])
                # Unit purchasing cost
                obj_expr.append(c_unit * Q[i, t])
                # Physical inventory holding carrying cost
                obj_expr.append(h_cost * I_pos[i, t])
                # Lost sales opportunity penalty
                obj_expr.append(shortage_pen * I_neg[i, t])

        solver.Minimize(solver.Sum(obj_expr))

        # Solve
        logger.info("Executing OR-Tools MIP solver...")
        status = solver.Solve()

        is_optimal = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
        status_name = "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else ("FEASIBLE" if status == pywraplp.Solver.FEASIBLE else "INFEASIBLE")

        orders_matrix = np.zeros((n_items, horizon_days), dtype=np.int32)
        projected_stock = np.zeros((n_items, horizon_days), dtype=np.float32)
        projected_lost_sales = np.zeros((n_items, horizon_days), dtype=np.float32)

        if is_optimal:
            for i in range(n_items):
                for t in range(horizon_days):
                    orders_matrix[i, t] = int(round(Q[i, t].solution_value()))
                    projected_stock[i, t] = float(max(I_pos[i, t].solution_value(), 0.0))
                    projected_lost_sales[i, t] = float(max(I_neg[i, t].solution_value(), 0.0))

        return {
            "status": status_name,
            "objective_value": solver.Objective().Value() if is_optimal else None,
            "orders_matrix": orders_matrix,
            "projected_inventory": projected_stock,
            "projected_lost_sales": projected_lost_sales,
            "item_ids": item_ids,
            "horizon_days": horizon_days
        }
