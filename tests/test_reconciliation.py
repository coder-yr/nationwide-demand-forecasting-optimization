"""Exhaustive Tests for Complete 12-Level Hierarchical Reconciliation and Coherence."""
import numpy as np
import pandas as pd
import pytest

from src.models.reconciler import CompleteHierarchicalReconciler

def create_synthetic_bottom_panel(num_days=10):
    """Generates a complete multi-tier Level 12 dataset across states, stores, categories, depts."""
    dates = pd.date_range("2016-01-01", periods=num_days)
    records = []
    
    # 2 States (CA, TX), 2 stores each, 2 categories (FOODS, HOBBIES), 2 depts each, 2 items each
    for d in dates:
        for state in ["CA", "TX"]:
            for store_num in [1, 2]:
                store_id = f"{state}_{store_num}"
                for cat in ["FOODS", "HOBBIES"]:
                    for dept_num in [1, 2]:
                        dept_id = f"{cat}_{dept_num}"
                        for item_num in [1, 2]:
                            item_id = f"{dept_id}_{item_num:03d}"
                            records.append({
                                "date": d,
                                "state_id": state,
                                "store_id": store_id,
                                "cat_id": cat,
                                "dept_id": dept_id,
                                "item_id": item_id,
                                "p50": float(np.random.randint(1, 20))
                            })
    return pd.DataFrame(records)

def test_full_12_level_reconciliation_coherence():
    """Verifies that all 12 hierarchical tiers are generated and parent-child coherence is exact."""
    df_bottom = create_synthetic_bottom_panel(num_days=7)
    reconciled = CompleteHierarchicalReconciler.bottom_up_reconciliation(df_bottom, val_col="p50")

    # Verify all 12 levels exist
    for lvl in range(1, 13):
        assert lvl in reconciled, f"Level {lvl} missing from reconciled output!"
        assert len(reconciled[lvl]) > 0, f"Level {lvl} has 0 records!"

    # Run exhaustive parent-child key-matched verification
    is_coherent, log = CompleteHierarchicalReconciler.verify_exhaustive_coherence(reconciled)

    for entry in log:
        assert entry["is_coherent"], (
            f"COHERENCE FAILURE: Level {entry['parent_level']} ({entry['parent_name']}) vs "
            f"Level {entry['child_level']} ({entry['child_name']}) has error {entry['max_discrepancy']}!"
        )

    assert is_coherent is True
