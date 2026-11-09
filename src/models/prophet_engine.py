"""Prophet Additive Forecasting Engine for Macro Hierarchical Aggregates (Levels 1 to 5).

Decomposes annual, weekly, and holiday seasonal components for high-level series.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from prophet import Prophet
from src.utils.logger import get_logger

logger = get_logger("prophet_engine")

class HierarchicalProphetEngine:
    """Manages structural Prophet models fitted on aggregated tiers (National, State, Category, Dept)."""

    def __init__(self, interval_width: float = 0.80):
        self.interval_width = interval_width
        self.models: Dict[str, Prophet] = {}

    def fit_series(
        self,
        series_id: str,
        df_series: pd.DataFrame,
        date_col: str = "date",
        sales_col: str = "sales"
    ) -> "HierarchicalProphetEngine":
        """Fits a single Prophet model for an aggregated time series."""
        logger.info(f"Fitting Prophet on aggregate series '{series_id}' ({len(df_series)} observations)...")

        # Prophet expects columns 'ds' and 'y'
        pdf = pd.DataFrame({
            "ds": pd.to_datetime(df_series[date_col]),
            "y": df_series[sales_col].astype(float)
        }).sort_values("ds").reset_index(drop=True)

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=self.interval_width
        )
        m.add_country_holidays(country_name="US")
        m.fit(pdf)

        self.models[series_id] = m
        return self

    def predict_series(
        self,
        series_id: str,
        horizon_days: int = 28
    ) -> pd.DataFrame:
        """Emits point and confidence interval forecasts for an aggregate series."""
        if series_id not in self.models:
            raise ValueError(f"Series '{series_id}' has not been fitted.")

        m = self.models[series_id]
        future = m.make_future_dataframe(periods=horizon_days, freq="D")
        forecast = m.predict(future)

        # Tail horizon
        res = forecast.tail(horizon_days)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        res["yhat"] = np.maximum(res["yhat"], 0.0)
        res["yhat_lower"] = np.maximum(res["yhat_lower"], 0.0)
        res["yhat_upper"] = np.maximum(res["yhat_upper"], 0.0)
        res["series_id"] = series_id
        return res

