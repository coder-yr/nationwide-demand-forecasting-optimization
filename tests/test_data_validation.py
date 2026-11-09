"""Data Integrity, Schema, and Leakage Validation Tests."""
import numpy as np
import pandas as pd
import pytest
import duckdb
from pathlib import Path

from src.data.pipeline import M5DataPipeline

def test_data_partition_integrity_and_price_leakage():
    parquet_path = Path("data/processed/stores/store_CA_1.parquet")
    if not parquet_path.exists():
        pytest.skip("Store CA_1 parquet not yet generated.")

    df = duckdb.read_parquet(str(parquet_path)).df()

    # 1. Natural key uniqueness
    dups = df.duplicated(subset=["store_id", "item_id", "date"]).sum()
    assert dups == 0, f"Found {dups} duplicate records for (store_id, item_id, date)!"

    # 2. Non-negative unit sales
    assert (df["sales"] < 0).sum() == 0, "Found negative unit sales in partition!"

    # 3. Active vs Pre-launch item verification (Causal price leakage check)
    assert "is_active_item" in df.columns, "is_active_item column missing from partition!"
    active_items = df[df["is_active_item"] == 1]
    pre_launch_items = df[df["is_active_item"] == 0]

    # Active items MUST have valid positive selling prices
    if len(active_items) > 0:
        assert (active_items["sell_price"] <= 0).sum() == 0, "Active items have zero or negative prices!"
        assert active_items["sell_price"].isna().sum() == 0, "Active items have null prices!"

    # Pre-launch items MUST NOT leak positive launch prices via backward-fill
    # Verify that pre-launch prices are 0.0, strictly avoiding bfill
    if len(pre_launch_items) > 0:
        assert (pre_launch_items["sell_price"] > 0).sum() == 0, "Pre-launch items have positive prices (bfill leakage detected)!"

    # 4. Mandatory columns existence
    expected_cols = [
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "d_num", "sales", "date", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "has_event", "snap_current_state", "sell_price", "is_active_item"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Expected column '{col}' missing from processed schema!"

def test_downcaster_efficiency():
    df_sample = pd.DataFrame({
        "sales": [0, 5, 20, 100],
        "price": [1.99, 4.50, 12.00, 99.99],
        "category": ["FOODS", "FOODS", "HOBBIES", "HOBBIES"]
    })
    from src.data.downcaster import downcast_dataframe
    downcasted = downcast_dataframe(df_sample.copy(), verbose=False)
    assert downcasted["sales"].dtype == np.uint8
    assert downcasted["price"].dtype == np.float32
    assert isinstance(downcasted["category"].dtype, pd.CategoricalDtype)

