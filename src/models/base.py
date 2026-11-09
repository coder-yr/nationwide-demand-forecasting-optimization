"""Abstract Base Forecaster Interface."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

class BaseForecaster(ABC):
    """Abstract base class that all forecasting model implementations must inherit from."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, feature_cols: List[str], target_col: str = "sales") -> "BaseForecaster":
        """Fits the model on the historical training DataFrame."""
        pass

    @abstractmethod
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Returns point demand forecasts for the given feature matrix."""
        pass

    @abstractmethod
    def predict_quantiles(
        self,
        test_df: pd.DataFrame,
        quantiles: Optional[List[float]] = None
    ) -> Dict[float, np.ndarray]:
        """Returns probabilistic forecast intervals (e.g. {0.10: p10_array, 0.50: p50_array, 0.90: p90_array})."""
        pass

