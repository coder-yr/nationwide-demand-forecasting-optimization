"""LightGBM Quantile & Tweedie Hierarchical Forecaster.

Supports zero-inflated continuous point forecasting (Tweedie objective),
calibrated probabilistic demand intervals (Pinball / Quantile loss for P10, P50, P90),
quantile crossing correction, and model serialization.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
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
        self.quantiles = sorted(quantiles or [0.10, 0.50, 0.90])
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
        self.metadata: Dict[str, Any] = {}

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
                num_boost_round=max(self.n_estimators // 2, 80),
                callbacks=[lgb.log_evaluation(period=0)]
            )

        self.metadata = {
            "model_type": "LightGBM_Tweedie_Quantile",
            "quantiles": self.quantiles,
            "feature_cols": self.feature_cols,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "tweedie_power": self.tweedie_variance_power
        }

        logger.info("LightGBM fitting completed successfully.")
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Emits non-negative Tweedie point predictions."""
        if self.point_model is None:
            raise RuntimeError("Model has not been fitted. Call fit() or load_model() first.")
        preds = self.point_model.predict(test_df[self.feature_cols])
        return np.maximum(preds, 0.0)

    def predict_quantiles(
        self,
        test_df: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        correct_crossing: bool = True
    ) -> Dict[float, np.ndarray]:
        """Emits calibrated predictions for each quantile with monotonic sorting correction."""
        active_quantiles = sorted(quantiles or self.quantiles)
        raw_results = {}
        for q in active_quantiles:
            if q in self.quantile_models:
                raw_pred = self.quantile_models[q].predict(test_df[self.feature_cols])
                raw_results[q] = np.maximum(raw_pred, 0.0)
            else:
                raw_results[q] = self.predict(test_df)

        if not correct_crossing or len(active_quantiles) < 2:
            return raw_results

        # Check and correct quantile crossing across active quantiles
        # Stack into matrix [n_samples, n_quantiles]
        q_cols = [raw_results[q] for q in active_quantiles]
        q_matrix = np.column_stack(q_cols)

        # Monotonic sort per row guarantees P10 <= P50 <= P90
        q_sorted = np.sort(q_matrix, axis=1)

        # Count crossing instances
        inversions = np.sum(q_matrix != q_sorted)
        if inversions > 0:
            logger.info(f"Quantile crossing: corrected {inversions:,} coordinate inversions across {len(test_df):,} rows.")

        corrected_results = {}
        for idx, q in enumerate(active_quantiles):
            corrected_results[q] = q_sorted[:, idx]

        return corrected_results

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns gain-based feature importances."""
        if self.point_model is None:
            return pd.DataFrame()
        importances = self.point_model.feature_importance(importance_type="gain")
        return pd.DataFrame({
            "feature": self.feature_cols,
            "importance": importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def save_model(self, save_dir: str):
        """Serializes point and quantile boosters along with feature metadata."""
        out_path = Path(save_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.point_model:
            self.point_model.save_model(str(out_path / "point_model.txt"))

        for q, model in self.quantile_models.items():
            model.save_model(str(out_path / f"quantile_{int(q*100):02d}.txt"))

        with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Successfully serialized LightGBM models to {out_path}")

    @classmethod
    def load_model(cls, save_dir: str) -> "LGBMForecaster":
        """Loads serialized boosters and metadata."""
        in_path = Path(save_dir)
        if not (in_path / "metadata.json").exists():
            raise FileNotFoundError(f"Model metadata not found at {in_path / 'metadata.json'}")

        with open(in_path / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(
            quantiles=meta.get("quantiles", [0.10, 0.50, 0.90]),
            learning_rate=meta.get("learning_rate", 0.05),
            num_leaves=meta.get("num_leaves", 63),
            tweedie_variance_power=meta.get("tweedie_power", 1.15)
        )
        instance.feature_cols = meta.get("feature_cols", [])
        instance.metadata = meta

        if (in_path / "point_model.txt").exists():
            instance.point_model = lgb.Booster(model_file=str(in_path / "point_model.txt"))

        for q in instance.quantiles:
            q_file = in_path / f"quantile_{int(q*100):02d}.txt"
            if q_file.exists():
                instance.quantile_models[q] = lgb.Booster(model_file=str(q_file))

        logger.info(f"Loaded LGBMForecaster from {in_path} ({len(instance.quantile_models)} quantiles)")
        return instance
