"""Data Ingestion, Transformation, and Parquet Lake Pipeline for M5 Dataset.

Processes the raw wide M5 format into clean, store-partitioned Parquet files
joined with calendar and pricing dynamics.
"""
import os
import gc
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import duckdb

from src.data.downcaster import downcast_dataframe
from src.utils.logger import get_logger

logger = get_logger("data_pipeline")

class M5DataPipeline:
    """Manages the end-to-end transformation of raw M5 CSVs into partitioned Parquet stores."""

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        stores_output_dir: str = "data/processed/stores",
        history_days_window: Optional[int] = 730  # Default to last 2 years for optimal training
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.stores_output_dir = Path(stores_output_dir)
        self.history_days_window = history_days_window

        self.calendar_path = self.raw_dir / "calendar.csv"
        self.sales_path = (
            self.raw_dir / "sales_train_evaluation.csv"
            if (self.raw_dir / "sales_train_evaluation.csv").exists()
            else self.raw_dir / "sales_train_validation.csv"
        )
        self.prices_path = self.raw_dir / "sell_prices.csv"

        self.stores_output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_calendar(self) -> pd.DataFrame:
        """Loads and pre-processes calendar events and SNAP indicators."""
        logger.info(f"Loading calendar from {self.calendar_path}")
        df_cal = pd.read_csv(self.calendar_path)
        df_cal["date"] = pd.to_datetime(df_cal["date"])
        df_cal["d_num"] = df_cal["d"].apply(lambda x: int(x.split("_")[1])).astype(np.uint16)

        # Fill event na with 'None'
        df_cal["event_name_1"] = df_cal["event_name_1"].fillna("None")
        df_cal["event_type_1"] = df_cal["event_type_1"].fillna("None")
        df_cal["has_event"] = (df_cal["event_name_1"] != "None").astype(np.uint8)

        # Categoricals and downcast
        for col in ["event_name_1", "event_type_1", "weekday"]:
            df_cal[col] = df_cal[col].astype("category")

        df_cal = downcast_dataframe(df_cal, verbose=False)
        return df_cal

    def load_prices(self, store_id: Optional[str] = None) -> pd.DataFrame:
        """Loads and downcasts sell_prices, optionally filtering by store_id."""
        logger.info(f"Loading prices from {self.prices_path}" + (f" for store {store_id}" if store_id else ""))
        if store_id:
            con = duckdb.connect()
            query = f"SELECT * FROM read_csv_auto('{self.prices_path.as_posix()}') WHERE store_id = '{store_id}'"
            df_prices = con.execute(query).df()
            con.close()
        else:
            df_prices = pd.read_csv(self.prices_path)

        df_prices = downcast_dataframe(df_prices, verbose=False)
        return df_prices

    def get_available_stores(self) -> List[str]:
        """Queries unique store IDs from the raw sales file header."""
        con = duckdb.connect()
        query = f"SELECT DISTINCT store_id FROM read_csv_auto('{self.sales_path.as_posix()}')"
        stores = [row[0] for row in con.execute(query).fetchall()]
        con.close()
        return sorted(stores)

    def process_store(
        self,
        store_id: str,
        df_calendar: pd.DataFrame,
        overwrite: bool = False
    ) -> Path:
        """Transforms a single store's sales into long format, joins calendar & prices,

        and saves a Snappy-compressed Parquet partition.
        """
        output_file = self.stores_output_dir / f"store_{store_id}.parquet"
        if output_file.exists() and not overwrite:
            logger.info(f"Store {store_id} parquet already exists at {output_file}. Skipping.")
            return output_file

        logger.info(f"Processing store: {store_id}...")

        # 1. Read store sales using DuckDB
        con = duckdb.connect()
        query = f"SELECT * FROM read_csv_auto('{self.sales_path.as_posix()}') WHERE store_id = '{store_id}'"
        df_store_wide = con.execute(query).df()
        con.close()

        # Identify d_ columns
        id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        d_cols = [c for c in df_store_wide.columns if c.startswith("d_")]

        # Filter recent history if window specified
        if self.history_days_window and len(d_cols) > self.history_days_window:
            d_cols = d_cols[-self.history_days_window:]

        # 2. Melt wide to long
        logger.info(f"Melting {len(df_store_wide)} items across {len(d_cols)} days for store {store_id}")
        df_long = pd.melt(
            df_store_wide,
            id_vars=id_cols,
            value_vars=d_cols,
            var_name="d",
            value_name="sales"
        )
        del df_store_wide
        gc.collect()

        df_long["d_num"] = df_long["d"].apply(lambda x: int(x.split("_")[1])).astype(np.uint16)
        df_long["sales"] = df_long["sales"].astype(np.uint16)

        # 3. Join Calendar
        state_id = store_id.split("_")[0]
        snap_col = f"snap_{state_id}"

        cal_subset = df_calendar[[
            "d_num", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
            "event_name_1", "event_type_1", "has_event", snap_col
        ]].rename(columns={snap_col: "snap_current_state"})

        df_merged = df_long.merge(cal_subset, on="d_num", how="left")
        del df_long
        gc.collect()

        # 4. Join Prices
        df_prices = self.load_prices(store_id=store_id)
        df_merged = df_merged.merge(
            df_prices[["item_id", "wm_yr_wk", "sell_price"]],
            on=["item_id", "wm_yr_wk"],
            how="left"
        )
        del df_prices
        gc.collect()

        # Fill price missing values with item mean or forward-fill
        df_merged["sell_price"] = df_merged.groupby("item_id")["sell_price"].transform(
            lambda x: x.ffill().bfill()
        ).astype(np.float32)

        # Downcast
        df_merged = downcast_dataframe(df_merged, verbose=False)

        # 5. Write to Parquet with Snappy compression
        logger.info(f"Writing {len(df_merged):,} rows to {output_file}")
        df_merged.to_parquet(output_file, engine="pyarrow", compression="snappy", index=False)

        del df_merged
        gc.collect()
        return output_file

    def run_all_stores(self, overwrite: bool = False) -> List[Path]:
        """Runs the complete ingestion and partitioning pipeline for all stores."""
        df_cal = self.load_calendar()
        stores = self.get_available_stores()
        logger.info(f"Starting ingestion pipeline for {len(stores)} stores: {stores}")

        generated_files = []
        for store in stores:
            path = self.process_store(store, df_calendar=df_cal, overwrite=overwrite)
            generated_files.append(path)

        logger.info(f"Pipeline complete. {len(generated_files)} store partitions generated in {self.stores_output_dir}")
        return generated_files

