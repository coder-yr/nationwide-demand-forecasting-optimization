"""Inventory optimization and replenishment engine modules."""
from .safety_stock import calculate_safety_stock_and_rop

def get_optimizer(*args, **kwargs):
    """Lazy loader for InventoryReplenishmentOptimizer to avoid Windows DLL conflicts."""
    from .solver import InventoryReplenishmentOptimizer
    return InventoryReplenishmentOptimizer(*args, **kwargs)

__all__ = ["calculate_safety_stock_and_rop", "get_optimizer"]
