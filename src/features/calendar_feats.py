"""Calendar, cyclical temporal, and holiday event feature generator."""
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("calendar_feats")

def generate_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts cyclical calendar representations and event indicators.

    Args:
        df: DataFrame with date, wday, month, event_name_1, event_type_1.

    Returns:
        pd.DataFrame: Augmented with cyclical and event flags.
    """
    logger.info("Generating calendar and event features...")

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df["day_of_month"] = df["date"].dt.day.astype(np.uint8)
    df["is_weekend"] = df["wday"].isin([1, 2]).astype(np.uint8)  # M5 wday 1=Saturday, 2=Sunday

    # Cyclical day of week encoding
    df["sin_wday"] = np.sin(2 * np.pi * df["wday"] / 7.0).astype(np.float32)
    df["cos_wday"] = np.cos(2 * np.pi * df["wday"] / 7.0).astype(np.float32)

    # Cyclical month encoding
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0).astype(np.float32)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0).astype(np.float32)

    # Categorical codes for LightGBM
    for cat_col in ["dept_id", "cat_id", "event_name_1", "event_type_1"]:
        if cat_col in df.columns:
            df[f"{cat_col}_code"] = df[cat_col].astype("category").cat.codes.astype(np.int16)

    return df

