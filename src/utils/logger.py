"""Structured logger for the nationwide demand forecasting and inventory platform."""
import logging
import sys
from typing import Optional

def get_logger(name: str = "demand_forecast", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a consistent console logger with standard formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

