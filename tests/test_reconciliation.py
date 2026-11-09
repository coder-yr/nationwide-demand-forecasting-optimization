"""Unit tests for hierarchical reconciliation and mathematical coherence."""
import numpy as np
import pandas as pd
import pytest
from src.models.reconciler import HierarchicalReconciler

def test_bottom_up_coherence():
    # Construct a synthetic multi-store, multi-dept dataset
    dates = pd.date_range("2016-01-01", periods=5)
    records = []
    for d in dates:
        for store in ["CA_1", "CA_2"]:
            for dept in ["FOODS_1", "HOBBIES_1"]:
                for item in [1, 2]:
                    records.append({
                        "date": d,
                        "state_id": "CA",
                        "store_id": store,
                        "dept_id": dept,
                        "cat_id": dept.split("_")[0],
                        "item_id": f"{dept}_{item}",
                        "p50": np.random.uniform(10, 50)
                    })
    df = pd.DataFrame(records)

    reconciled = HierarchicalReconciler.bottom_up_reconciliation(df, forecast_val_col="p50")

    # Verify that sum of store-dept equals store total
    store_dept = reconciled["level_9_store_dept"]
    store_tot = reconciled["level_3_store"]

    is_coherent = HierarchicalReconciler.verify_coherence(
        child_forecasts=store_dept,
        parent_forecasts=store_tot,
        child_group_cols=["store_id", "date"],
        parent_group_cols=["store_id", "date"],
        val_col="reconciled_sales"
    )
    assert is_coherent, "Store Department sum does not match Store Total!"

    # Verify that sum of stores equals State total
    state_tot = reconciled["level_2_state"]
    is_state_coherent = HierarchicalReconciler.verify_coherence(
        child_forecasts=store_tot,
        parent_forecasts=state_tot,
        child_group_cols=["date"],
        parent_group_cols=["date"],
        val_col="reconciled_sales"
    )
    assert is_state_coherent, "Store sum does not match State Total!"

