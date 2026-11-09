"""LightGBM Quantile & Tweedie Hierarchical Forecaster.

Supports both zero-inflated continuous point forecasting (Tweedie objective)
and calibrated probabilistic demand intervals (Pinball / Quantile loss for P10, P50, P90).
"""
import os
from typing import Dict, List, Optional
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.models.base import BaseForecaster
from src.utils.logger import get_logger

logger = get_logger("lgbm_forecaster")

class LGBMForecaster(BaseForecaster):
    """Production-grade LightGBM forecaster for store-item panel data."""

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        min_data_in_leaf: int = 50,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        tweedie_variance_power: float = 1.15,
        random_state: int = 42
    ):
        self.quantiles = quantiles or [0.10, 0.50, 0.90]
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.tweedie_variance_power = tweedie_variance_power
        self.random_state = random_state

        self.point_model: Optional[lgb.Booster] = None
        self.quantile_models: Dict[float, lgb.Booster] = {}
        self.feature_cols: List[str] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "sales",
        val_df: Optional[pd.DataFrame] = None
    ) -> "LGBMForecaster":
        """Fits both point Tweedie model and quantile regressors."""
        self.feature_cols = feature_cols
        logger.info(f"Training LightGBM on {len(train_df):,} rows with {len(feature_cols)} features...")

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        # 1. Train Primary Point Forecast (Tweedie objective)
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        valid_sets = [train_data]
        if val_df is not None:
            val_data = lgb.Dataset(val_df[feature_cols], label=val_df[target_col], reference=train_data, free_raw_data=False)
            valid_sets.append(val_data)

        point_params = {
            "objective": "tweedie",
            "tweedie_variance_power": self.tweedie_variance_power,
            "metric": "rmse",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": 1,
            "seed": self.random_state,
            "verbose": -1,
            "n_jobs": -1
        }

        self.point_model = lgb.train(
            point_params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            callbacks=[lgb.log_evaluation(period=0)]
        )

        # 2. Train Quantile Regressors for Uncertainty Intervals (P10, P50, P90)
        for q in self.quantiles:
            logger.info(f"Fitting quantile regressor for alpha={q:.2f}...")
            q_params = {
                "objective": "quantile",
                "alpha": q,
                "metric": "quantile",
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "min_data_in_leaf": self.min_data_in_leaf,
                "feature_fraction": self.feature_fraction,
                "bagging_fraction": self.bagging_fraction,
                "bagging_freq": 1,
                "seed": self.random_state,
                "verbose": -1,
                "n_jobs": -1
            }
            self.quantile_models[q] = lgb.train(
                q_params,
                train_data,
                num_boost_round=max(self.n_estimators // 2, 100),
                callbacks=[lgb.log_evaluation(period=0)]
            )

        logger.info("LightGBM fitting completed successfully.")
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Emits Tweedie point predictions."""
        if self.point_model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        preds = self.point_model.predict(test_df[self.feature_cols])
        return np.maximum(preds, 0.0)

    def predict_quantiles(
        self,
        test_df: pd.DataFrame,
        quantiles: Optional[List[float]] = None
    ) -> Dict[float, np.ndarray]:
        """Emits predictions for each calibrated quantile."""
        active_quantiles = quantiles or self.quantiles
        results = {}
        for q in active_quantiles:
            if q in self.quantile_models:
                raw_pred = self.quantile_models[q].predict(test_df[self.feature_cols])
                results[q] = np.maximum(raw_pred, 0.0)
            else:
                # Fallback to point prediction
                results[q] = self.predict(test_df)
        return results

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns gain-based feature importances."""
        if self.point_model is None:
            return pd.DataFrame()
        importances = self.point_model.feature_importance(importance_type="gain")
        return pd.DataFrame({
            "feature": self.feature_cols,
            "importance": importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)

