"""Price elasticity, promotional discount, and price momentum feature generator."""
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("price_feats")

def generate_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates discount depth, price changes, and department-relative pricing.

    Args:
        df: DataFrame containing item_id, dept_id, wm_yr_wk, and sell_price.

    Returns:
        pd.DataFrame: Augmented with price elasticity features.
    """
    logger.info("Generating price elasticity and discount features...")
    df = df.sort_values(["item_id", "d_num"]).reset_index(drop=True)

    # 1. Rolling 4-week maximum price per item
    item_price_grp = df.groupby("item_id")["sell_price"]
    df["price_max_28d"] = item_price_grp.transform(lambda s: s.rolling(28, min_periods=1).max()).astype(np.float32)

    # 2. Discount ratio: current price vs 4-week historical high (discount depth)
    df["discount_ratio"] = (df["sell_price"] / np.maximum(df["price_max_28d"], 1e-4)).clip(0.0, 1.5).astype(np.float32)
    df["is_discounted"] = (df["discount_ratio"] < 0.98).astype(np.uint8)

    # 3. Price change week-over-week (difference from lag 7)
    df["price_diff_lag7"] = (df["sell_price"] - item_price_grp.shift(7)).fillna(0.0).astype(np.float32)

    return df

