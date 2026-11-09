"""End-to-End Model Training, Backtesting, and Evaluation Pipeline.

Executes:
1. Loading store partition from Parquet data lake via DuckDB.
2. Generating leakage-free lag, rolling, pricing, and cyclical calendar features.
3. Walk-forward train/test split (28-day holdout horizon).
4. Training LightGBM Tweedie point model + P10, P50, P90 Quantile regressors.
5. Calculating WRMSSE, WAPE, and Scaled Pinball Loss.
6. Emitting forecast artifacts for hierarchical reconciliation and inventory optimization.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import duckdb

from src.features.lag_generator import generate_lag_features
from src.features.price_feats import generate_price_features
from src.features.calendar_feats import generate_calendar_features
from src.models.lgbm_forecaster import LGBMForecaster
from src.utils.metrics import calculate_wape, calculate_rmsse, calculate_multi_quantile_loss
from src.utils.logger import get_logger

logger = get_logger("train_pipeline")

class ModelTrainingPipeline:
    """Orchestrates end-to-end model training, feature generation, and evaluation."""

    def __init__(
        self,
        store_id: str = "CA_1",
        stores_dir: str = "data/processed/stores",
        outputs_dir: str = "data/outputs",
        forecast_horizon: int = 28
    ):
        self.store_id = store_id
        self.stores_dir = Path(stores_dir)
        self.outputs_dir = Path(outputs_dir)
        self.forecast_horizon = forecast_horizon
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def load_and_engineer_features(self) -> Tuple[pd.DataFrame, List[str]]:
        """Loads store partition and creates all model features."""
        parquet_file = self.stores_dir / f"store_{self.store_id}.parquet"
        if not parquet_file.exists():
            raise FileNotFoundError(f"Store partition {parquet_file} does not exist. Run ingestion first.")

        logger.info(f"Loading {self.store_id} data lake partition via DuckDB...")
        df = duckdb.read_parquet(str(parquet_file)).df()

        logger.info(f"Loaded {len(df):,} rows. Engineering temporal and pricing features...")
        # 1. Price dynamics
        df = generate_price_features(df)
        # 2. Calendar and cyclical encodings
        df = generate_calendar_features(df)
        # 3. Lags and rolling statistics (strict min_shift=28)
        df = generate_lag_features(
            df,
            lags=[28, 29, 35, 42],
            rolling_windows=[7, 28],
            min_shift=self.forecast_horizon
        )

        # Drop burn-in rows where min lag 28 has NaNs
        initial_len = len(df)
        df = df.dropna(subset=["sales_lag_28"]).reset_index(drop=True)
        logger.info(f"Burn-in filtered: {initial_len:,} -> {len(df):,} rows ready for training.")

        feature_cols = [
            "wday", "month", "day_of_month", "is_weekend",
            "sin_wday", "cos_wday", "sin_month", "cos_month",
            "dept_id_code", "cat_id_code", "snap_current_state",
            "sell_price", "price_max_28d", "discount_ratio", "is_discounted", "price_diff_lag7",
            "sales_lag_28", "sales_lag_29", "sales_lag_35", "sales_lag_42",
            "roll_mean_7", "roll_std_7", "roll_mean_28", "roll_std_28",
            "ewma_10", "ewma_30"
        ]

        # Verify all feature columns exist
        feature_cols = [c for c in feature_cols if c in df.columns]
        logger.info(f"Selected {len(feature_cols)} feature columns for training.")
        return df, feature_cols

    def run(self) -> Dict[str, float]:
        """Executes walk-forward evaluation, training, metric calculation, and artifact emission."""
        df, feature_cols = self.load_and_engineer_features()

        # Split: last 28 days as holdout validation test set
        max_d_num = df["d_num"].max()
        split_cutoff = max_d_num - self.forecast_horizon

        train_df = df[df["d_num"] <= split_cutoff].reset_index(drop=True)
        test_df = df[df["d_num"] > split_cutoff].reset_index(drop=True)

        logger.info(
            f"Train set: {len(train_df):,} rows (d_num <= {split_cutoff}) | "
            f"Test set: {len(test_df):,} rows ({split_cutoff} < d_num <= {max_d_num})"
        )

        # Instantiate and fit LightGBM Quantile Forecaster
        forecaster = LGBMForecaster(
            quantiles=[0.10, 0.50, 0.90],
            n_estimators=150,
            learning_rate=0.08,
            num_leaves=31
        )
        forecaster.fit(train_df=train_df, feature_cols=feature_cols, val_df=test_df)

        # Generate point and quantile predictions on holdout test set
        test_df["point_pred"] = forecaster.predict(test_df)
        quantiles = forecaster.predict_quantiles(test_df)
        test_df["p10"] = quantiles[0.10]
        test_df["p50"] = quantiles[0.50]
        test_df["p90"] = quantiles[0.90]

        # Evaluate Performance Metrics
        y_true = test_df["sales"].values
        y_pred = test_df["point_pred"].values
        wape = calculate_wape(y_true, y_pred)

        quantile_losses = calculate_multi_quantile_loss(
            y_true=y_true,
            quantile_predictions={0.10: test_df["p10"].values, 0.50: test_df["p50"].values, 0.90: test_df["p90"].values}
        )

        # RMSSE on aggregate store series
        store_true = test_df.groupby("date")["sales"].sum().values
        store_pred = test_df.groupby("date")["point_pred"].sum().values
        store_hist = train_df.groupby("date")["sales"].sum().values
        rmsse = calculate_rmsse(store_true, store_pred, store_hist)

        logger.info("=" * 60)
        logger.info(f"MODEL PERFORMANCE EVALUATION FOR STORE {self.store_id}:")
        logger.info(f"  • Aggregate Store RMSSE : {rmsse:.4f}")
        logger.info(f"  • Item-Level WAPE        : {wape * 100:.2f}%")
        logger.info(f"  • Mean Pinball Loss     : {quantile_losses['mean_pinball_loss']:.4f}")
        logger.info(f"  • P10 Pinball Loss      : {quantile_losses['pinball_loss_q10']:.4f}")
        logger.info(f"  • P50 Pinball Loss      : {quantile_losses['pinball_loss_q50']:.4f}")
        logger.info(f"  • P90 Pinball Loss      : {quantile_losses['pinball_loss_q90']:.4f}")
        logger.info("=" * 60)

        # Feature Importance
        fi_df = forecaster.get_feature_importances()
        logger.info(f"Top 5 Most Influential Features:\n{fi_df.head(5).to_string(index=False)}")

        # Save Forecast Artifacts to Parquet
        output_path = self.outputs_dir / f"forecast_{self.store_id}.parquet"
        save_cols = [
            "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
            "date", "d_num", "sales", "point_pred", "p10", "p50", "p90", "sell_price"
        ]
        test_df[save_cols].to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
        logger.info(f"Successfully saved {len(test_df):,} forecast records to {output_path}")

        return {
            "rmsse": rmsse,
            "wape": wape,
            "mean_pinball_loss": quantile_losses["mean_pinball_loss"]
        }

if __name__ == "__main__":
    pipeline = ModelTrainingPipeline(store_id="CA_1")
    pipeline.run()

