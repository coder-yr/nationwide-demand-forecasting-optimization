"""Price elasticity, promotional discount, and price momentum feature generator."""
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("price_feats")

def generate_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates discount depth, price volatility, and price momentum features.

    All rolling price features are computed strictly from historical prices up to time t,
    without using future price changes.

    Args:
        df: DataFrame containing item_id, dept_id, sell_price, and d_num.

    Returns:
        pd.DataFrame: Augmented with price elasticity and momentum features.
    """
    logger.info("Generating price elasticity, discount depth, and momentum features...")
    df = df.sort_values(["item_id", "d_num"]).reset_index(drop=True)

    item_price_grp = df.groupby("item_id")["sell_price"]

    # 1. Rolling 28-day maximum price per item
    df["price_max_28d"] = item_price_grp.transform(lambda s: s.rolling(28, min_periods=1).max()).astype(np.float32)

    # 2. Rolling 28-day minimum and mean price
    df["price_min_28d"] = item_price_grp.transform(lambda s: s.rolling(28, min_periods=1).min()).astype(np.float32)
    df["price_mean_28d"] = item_price_grp.transform(lambda s: s.rolling(28, min_periods=1).mean()).astype(np.float32)
    df["price_std_28d"] = item_price_grp.transform(lambda s: s.rolling(28, min_periods=1).std()).fillna(0.0).astype(np.float32)

    # 3. Discount ratio: current price vs 28-day historical high
    safe_max = np.maximum(df["price_max_28d"], 1e-4)
    df["discount_ratio"] = (df["sell_price"] / safe_max).clip(0.0, 1.5).astype(np.float32)
    df["is_discounted"] = (df["discount_ratio"] < 0.98).astype(np.uint8)

    # 4. Price Momentum: current price relative to 28-day rolling average
    safe_mean = np.maximum(df["price_mean_28d"], 1e-4)
    df["price_momentum"] = (df["sell_price"] / safe_mean).astype(np.float32)

    # 5. Price change week-over-week (difference from lag 7)
    df["price_diff_lag7"] = (df["sell_price"] - item_price_grp.shift(7)).fillna(0.0).astype(np.float32)

    return df
