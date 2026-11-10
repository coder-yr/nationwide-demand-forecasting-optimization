"""Dynamic Safety Stock and Reorder Point (ROP) Calculator.

Transforms probabilistic quantile spreads (P10, P50, P90) into calibrated
safety stock buffers and reorder triggers.
"""
from typing import Dict, Optional
import numpy as np
import scipy.stats as stats
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("safety_stock")

def calculate_safety_stock_and_rop(
    p10_demand: np.ndarray,
    p50_demand: np.ndarray,
    p90_demand: np.ndarray,
    lead_time_days: float = 3.0,
    lead_time_std_days: float = 0.5,
    service_level: float = 0.95
) -> Dict[str, np.ndarray]:
    """Calculates dynamic safety stock and Reorder Point (ROP).

    Formula:
      sigma_D ~ (P90 - P10) / (2 * Z_0.90) = (P90 - P10) / 2.5631
      SS = Z_service * sqrt(L * sigma_D^2 + D_mean^2 * sigma_L^2)
      ROP = D_mean * L + SS

    Returns:
        Dict[str, np.ndarray]: Calculated metrics for each item in the array.
    """
    service_level = float(service_level)
    z_service = stats.norm.ppf(service_level)
    z_90 = stats.norm.ppf(0.90)  # ~1.28155

    # Estimate standard deviation of daily demand from quantile spread
    sigma_demand = np.maximum((p90_demand - p10_demand) / (2.0 * z_90), 0.1)

    # Lead time variance contribution
    lt_demand_variance = lead_time_days * (sigma_demand ** 2)
    lt_duration_variance = (p50_demand ** 2) * (lead_time_std_days ** 2)

    total_std_during_lead_time = np.sqrt(lt_demand_variance + lt_duration_variance)
    safety_stock = z_service * total_std_during_lead_time
    lead_time_expected_demand = p50_demand * lead_time_days
    reorder_point = lead_time_expected_demand + safety_stock

    return {
        "sigma_demand": sigma_demand,
        "safety_stock": np.ceil(safety_stock),
        "reorder_point": np.ceil(reorder_point),
        "lead_time_demand": lead_time_expected_demand,
    }

