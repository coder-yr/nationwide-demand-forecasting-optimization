"""Calendar, cyclical temporal, and holiday event feature generator."""
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("calendar_feats")

def generate_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts cyclical calendar representations and multi-event indicators.

    Args:
        df: DataFrame with date, wday, month, year, event columns.

    Returns:
        pd.DataFrame: Augmented with calendar and event features.
    """
    logger.info("Generating calendar, cyclical temporal, and holiday features...")

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df["day_of_month"] = df["date"].dt.day.astype(np.uint8)
    df["week"] = df["date"].dt.isocalendar().week.astype(np.uint8)
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year.astype(np.uint16)

    # Weekend flag (M5 wday: 1=Saturday, 2=Sunday)
    df["is_weekend"] = df["wday"].isin([1, 2]).astype(np.uint8)

    # Cyclical day of week encoding (period = 7)
    df["sin_wday"] = np.sin(2 * np.pi * df["wday"] / 7.0).astype(np.float32)
    df["cos_wday"] = np.cos(2 * np.pi * df["wday"] / 7.0).astype(np.float32)

    # Cyclical month encoding (period = 12)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0).astype(np.float32)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0).astype(np.float32)

    # Cyclical week encoding (period = 52)
    df["sin_week"] = np.sin(2 * np.pi * df["week"] / 52.0).astype(np.float32)
    df["cos_week"] = np.cos(2 * np.pi * df["week"] / 52.0).astype(np.float32)

    # Event indicators: Primary and Secondary
    has_event_1 = (df["event_name_1"] != "None") if "event_name_1" in df.columns else pd.Series(False, index=df.index)
    has_event_2 = (df["event_name_2"] != "None") if "event_name_2" in df.columns else pd.Series(False, index=df.index)
    df["has_event"] = (has_event_1 | has_event_2).astype(np.uint8)

    # Categorical codes for LightGBM
    for cat_col in ["dept_id", "cat_id", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        if cat_col in df.columns:
            df[f"{cat_col}_code"] = df[cat_col].astype("category").cat.codes.astype(np.int16)

    return df
