"""Temporal Fusion Transformer (TFT) Forecaster using Darts and PyTorch Forecasting.

Applied to high-velocity Pareto Class-A products and category clusters with
multi-horizon attention and quantile loss outputs.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression

from src.utils.logger import get_logger

logger = get_logger("tft_forecaster")

class DartsTFTForecaster:
    """TFT model wrapper for multi-horizon probabilistic deep learning forecasting."""

    def __init__(
        self,
        input_chunk_length: int = 56,
        output_chunk_length: int = 28,
        hidden_size: int = 32,
        lstm_layers: int = 1,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        batch_size: int = 64,
        n_epochs: int = 10,
        random_state: int = 42
    ):
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.quantiles = [0.10, 0.50, 0.90]

        # PyTorch Forecasting likelihood with calibrated quantiles
        likelihood = QuantileRegression(quantiles=self.quantiles)

        self.model = TFTModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            batch_size=batch_size,
            n_epochs=n_epochs,
            likelihood=likelihood,
            random_state=random_state,
            pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False}
        )

    def fit_from_dataframe(
        self,
        df: pd.DataFrame,
        time_col: str = "date",
        target_col: str = "sales",
        series_id_col: str = "item_id",
        past_covariates_cols: Optional[List[str]] = None
    ) -> "DartsTFTForecaster":
        """Fits TFT on a collection of time series (e.g. top revenue drivers)."""
        logger.info(f"Preparing Darts TimeSeries from DataFrame for TFT training...")

        # Ensure daily regularity
        series_list = []
        for s_id, s_df in df.groupby(series_id_col):
            s_sorted = s_df.sort_values(time_col).drop_duplicates(time_col)
            ts = TimeSeries.from_dataframe(
                s_sorted,
                time_col=time_col,
                value_cols=target_col,
                freq="D",
                fill_missing_dates=True,
                fillna_value=0.0
            )
            series_list.append(ts)

        logger.info(f"Fitting TFT model across {len(series_list)} time series...")
        self.model.fit(series=series_list, verbose=False)
        logger.info("TFT fitting complete.")
        return self

    def predict_quantiles(
        self,
        series: List[TimeSeries],
        n_days: int = 28,
        num_samples: int = 200
    ) -> List[Dict[str, np.ndarray]]:
        """Emits quantile predictions (p10, p50, p90) for target time series."""
        raw_preds = self.model.predict(n=n_days, series=series, num_samples=num_samples)
        if not isinstance(raw_preds, list):
            raw_preds = [raw_preds]

        results = []
        for pred in raw_preds:
            p10 = pred.quantile_timeseries(0.10).values().flatten()
            p50 = pred.quantile_timeseries(0.50).values().flatten()
            p90 = pred.quantile_timeseries(0.90).values().flatten()
            results.append({
                "p10": np.maximum(p10, 0.0),
                "p50": np.maximum(p50, 0.0),
                "p90": np.maximum(p90, 0.0),
            })
        return results

