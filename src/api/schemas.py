"""Pydantic schemas and data transfer objects for the FastAPI Gateway."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    data_lake_ready: bool = True
    available_stores: List[str]

class HierarchyNode(BaseModel):
    state_id: str
    stores: List[str]
    categories: List[str]

class ForecastRequest(BaseModel):
    store_id: str = Field(default="CA_1", description="Store code (e.g. CA_1)")
    category: Optional[str] = Field(default="ALL", description="Category or ALL")
    horizon_days: int = Field(default=28, ge=7, le=56, description="Forecast horizon")
    model_name: str = Field(default="lgbm", description="lgbm, tft, or prophet")
    reconcile: bool = Field(default=True, description="Enforce hierarchical reconciliation")

class ForecastResponse(BaseModel):
    store_id: str
    category: str
    horizon_days: int
    model_name: str
    reconciled: bool
    dates: List[str]
    p10: List[float]
    p50: List[float]
    p90: List[float]

class ReplenishmentRequest(BaseModel):
    store_id: str = Field(default="CA_1", description="Store to optimize")
    horizon_days: int = Field(default=7, ge=1, le=28)
    service_level: float = Field(default=0.95, ge=0.80, le=0.99)
    daily_budget: float = Field(default=50000.0, ge=1000.0)
    max_storage_m3: float = Field(default=1500.0, ge=100.0)

class ReplenishmentItem(BaseModel):
    item_id: str
    dept_id: str
    on_hand_stock: int
    safety_stock: int
    reorder_point: int
    recommended_order: int
    moq: int
    order_cost: float
    action: str

class ReplenishmentResponse(BaseModel):
    status: str
    store_id: str
    total_spend: float
    total_units_ordered: int
    items_analyzed: int
    items: List[ReplenishmentItem]

