"""End-to-End Inventory Replenishment Pipeline using Google OR-Tools.

Takes calibrated forecast outputs from the ML models, computes dynamic safety stocks,
and executes the multi-item constrained MIP optimization solver.
"""
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import duckdb

# ortools is imported safely before pandas/pyarrow via solver
from src.optimization.solver import InventoryReplenishmentOptimizer
from src.optimization.safety_stock import calculate_safety_stock_and_rop
from src.utils.logger import get_logger

logger = get_logger("replenish_pipeline")

class ReplenishmentPipeline:
    """Manages the transformation of model forecast artifacts into optimized purchase orders."""

    def __init__(
        self,
        store_id: str = "CA_1",
        forecasts_dir: str = "data/outputs",
        outputs_dir: str = "data/outputs",
        horizon_days: int = 7,
        service_level: float = 0.95,
        daily_budget: float = 50000.0,
        max_storage_m3: float = 1500.0
    ):
        self.store_id = store_id
        self.forecasts_dir = Path(forecasts_dir)
        self.outputs_dir = Path(outputs_dir)
        self.horizon_days = horizon_days
        self.service_level = service_level
        self.daily_budget = daily_budget
        self.max_storage_m3 = max_storage_m3

    def run(self) -> pd.DataFrame:
        """Executes the replenishment optimization workflow."""
        forecast_file = self.forecasts_dir / f"forecast_{self.store_id}.parquet"
        if not forecast_file.exists():
            raise FileNotFoundError(f"Forecast file {forecast_file} does not exist. Run model training first.")

        logger.info(f"Loading forecast artifacts for store {self.store_id} from {forecast_file}...")
        df_forecast = duckdb.read_parquet(str(forecast_file)).df()

        # Group by item and take first horizon_days
        horizon_dates = sorted(df_forecast["date"].unique())[:self.horizon_days]
        df_sub = df_forecast[df_forecast["date"].isin(horizon_dates)].copy()

        # Select top active items for constrained multi-period MIP
        top_items = df_sub.groupby("item_id")["p50"].sum().nlargest(100).index
        df_opt_items = df_sub[df_sub["item_id"].isin(top_items)].copy()

        logger.info(f"Formulating optimization model for top {len(top_items)} velocity items over {len(horizon_dates)} days...")

        # Pivot demands to 2D matrix (n_items, horizon_days)
        pivot_demand = df_opt_items.pivot(index="item_id", columns="date", values="p50").fillna(0.0)
        item_ids = list(pivot_demand.index)
        demand_matrix = pivot_demand.values

        # Aggregate unit prices and estimate unit cost (standard 65% wholesale ratio)
        price_map = df_opt_items.groupby("item_id")["sell_price"].last().to_dict()
        sell_prices = np.array([price_map.get(iid, 4.99) for iid in item_ids], dtype=np.float32)
        unit_costs = np.round(sell_prices * 0.65, 2)

        # Dynamic safety stock parameters
        p10_agg = df_opt_items.groupby("item_id")["p10"].mean().loc[item_ids].values
        p50_agg = df_opt_items.groupby("item_id")["p50"].mean().loc[item_ids].values
        p90_agg = df_opt_items.groupby("item_id")["p90"].mean().loc[item_ids].values

        ss_results = calculate_safety_stock_and_rop(
            p10_demand=p10_agg,
            p50_demand=p50_agg,
            p90_demand=p90_agg,
            lead_time_days=3.0,
            service_level=self.service_level
        )

        np.random.seed(42)
        n = len(item_ids)
        # Operational parameters
        initial_inventories = np.maximum(ss_results["safety_stock"] * np.random.uniform(0.3, 1.2, n), 0).round()
        lead_times = np.array([2] * n)
        moqs = np.random.choice([6, 12, 24], size=n)
        unit_vols = np.array([0.02] * n)

        optimizer = InventoryReplenishmentOptimizer(time_limit_seconds=30)
        res = optimizer.solve_replenishment(
            item_ids=item_ids,
            forecast_demands=demand_matrix,
            unit_costs=unit_costs,
            sell_prices=sell_prices,
            initial_inventories=initial_inventories,
            lead_times=lead_times,
            moqs=moqs,
            unit_volumes=unit_vols,
            max_storage_volume=self.max_storage_m3,
            budget_per_day=self.daily_budget,
            horizon_days=self.horizon_days
        )

        logger.info(f"Optimization Status: {res['status']}")
        if res["objective_value"] is not None:
            logger.info(f"Optimized Supply Chain Objective Cost: ${res['objective_value']:,.2f}")

        # Assemble actionable purchase order dataframe
        order_matrix = res["orders_matrix"]
        records = []
        for i, iid in enumerate(item_ids):
            total_ordered = int(np.sum(order_matrix[i, :]))
            cost = round(total_ordered * unit_costs[i], 2)
            cur_stock = int(initial_inventories[i])
            ss = int(ss_results["safety_stock"][i])
            action = "REORDER_NOW" if cur_stock < ss else ("OPTIMAL" if total_ordered > 0 else "HOLD")

            records.append({
                "store_id": self.store_id,
                "item_id": iid,
                "on_hand_stock": cur_stock,
                "safety_stock": ss,
                "reorder_point": int(ss_results["reorder_point"][i]),
                "recommended_order_qty": total_ordered,
                "moq": int(moqs[i]),
                "unit_cost": float(unit_costs[i]),
                "estimated_order_cost": cost,
                "action": action
            })

        orders_df = pd.DataFrame(records).sort_values("estimated_order_cost", ascending=False).reset_index(drop=True)

        output_csv = self.outputs_dir / f"replenishment_orders_{self.store_id}.csv"
        orders_df.to_csv(output_csv, index=False)
        logger.info(f"Actionable purchase orders saved to {output_csv}")
        logger.info(f"Summary: {len(orders_df)} SKUs evaluated | Total Spend: ${orders_df['estimated_order_cost'].sum():,.2f}")

        return orders_df

if __name__ == "__main__":
    pipeline = ReplenishmentPipeline(store_id="CA_1")
    pipeline.run()

