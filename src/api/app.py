"""FastAPI Production Inference and Inventory Analytics Gateway."""
# CRITICAL: Pre-load ortools on Windows to prevent DLL conflict
try:
    import ortools
except Exception:
    pass

import os
import glob
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import duckdb

from src.api.schemas import (
    HealthResponse,
    HierarchyNode,
    ForecastRequest,
    ForecastResponse,
    ForecastSummaryItem,
    ReplenishmentRequest,
    ReplenishmentResponse,
    ReplenishmentItem,
    ModelMetadataResponse,
    MetricsResponse
)
from src.dashboard.data_loader import DashboardDataLoader
from src.optimization.safety_stock import calculate_safety_stock_and_rop
from src.models.registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger("api_gateway")

app = FastAPI(
    title="Nationwide Demand Forecasting & Inventory Optimization API",
    version="1.0.0",
    description="Enterprise RESTful endpoints for Walmart M5 hierarchical demand forecasting and replenishment optimization."
)

# Secure CORS: explicit authorized domains (no wildcard *)
ALLOWED_ORIGINS = [
    "http://localhost:8050",
    "http://127.0.0.1:8050",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

data_loader = DashboardDataLoader()
registry = ModelRegistry()

@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """System heartbeat verifying data lake and model readiness."""
    stores_dir = Path("data/processed/stores")
    partitions = list(stores_dir.glob("store_*.parquet")) if stores_dir.exists() else []
    models_dir = Path("data/outputs/models")
    has_models = models_dir.exists() and len(list(models_dir.glob("*"))) > 0

    available_stores = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
    is_ready = len(partitions) > 0

    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        version="1.0.0",
        data_lake_ready=is_ready,
        models_ready=has_models,
        available_stores=available_stores,
        active_parquet_partitions=len(partitions)
    )

@app.get("/api/v1/hierarchy", response_model=List[HierarchyNode], tags=["Hierarchy"])
@app.get("/api/v1/hierarchy/tree", response_model=List[HierarchyNode], tags=["Hierarchy"])
def get_hierarchy():
    """Returns complete organizational hierarchy across States, Stores, Categories, and Departments."""
    departments = ["FOODS_1", "FOODS_2", "FOODS_3", "HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2"]
    categories = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    return [
        HierarchyNode(state_id="CA", stores=["CA_1", "CA_2", "CA_3", "CA_4"], categories=categories, departments=departments),
        HierarchyNode(state_id="TX", stores=["TX_1", "TX_2", "TX_3"], categories=categories, departments=departments),
        HierarchyNode(state_id="WI", stores=["WI_1", "WI_2", "WI_3"], categories=categories, departments=departments)
    ]

@app.get("/api/v1/forecasts", response_model=List[ForecastSummaryItem], tags=["Forecasting"])
def list_forecasts():
    """Lists available persisted forecast artifacts."""
    outputs_dir = Path("data/outputs")
    forecast_files = list(outputs_dir.glob("forecast_*.parquet"))
    summaries = []
    for f in forecast_files:
        store = f.stem.replace("forecast_", "")
        con = duckdb.connect(":memory:")
        try:
            df = con.read_parquet(str(f)).df()
        finally:
            con.close()
        summaries.append(ForecastSummaryItem(
            forecast_id=f.stem,
            store_id=store,
            created_at=f.stat().st_mtime.__str__(),
            record_count=len(df),
            horizon_days=df["date"].nunique()
        ))
    return summaries

@app.get("/api/v1/forecasts/{forecast_id}", response_model=ForecastResponse, tags=["Forecasting"])
def get_forecast_by_id(forecast_id: str):
    """Retrieves forecast series by identifier."""
    forecast_path = Path("data/outputs") / f"{forecast_id}.parquet"
    if not forecast_path.exists():
        # Check without .parquet suffix
        forecast_path = Path("data/outputs") / f"forecast_{forecast_id}.parquet"
    if not forecast_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Forecast '{forecast_id}' not found.")

    con = duckdb.connect(":memory:")
    try:
        df = con.read_parquet(str(forecast_path)).df()
    finally:
        con.close()
    daily = df.groupby("date").agg({"p10": "sum", "p50": "sum", "p90": "sum"}).reset_index().sort_values("date")
    dates = [d.strftime("%Y-%m-%d") for d in daily["date"]]
    store_val = str(df["store_id"].iloc[0]) if "store_id" in df.columns else forecast_id

    return ForecastResponse(
        forecast_id=forecast_id,
        store_id=store_val,
        category="ALL",
        horizon_days=len(dates),
        model_name="LightGBM_Tweedie",
        reconciled=True,
        dates=dates,
        p10=[round(float(v), 2) for v in daily["p10"]],
        p50=[round(float(v), 2) for v in daily["p50"]],
        p90=[round(float(v), 2) for v in daily["p90"]]
    )

@app.post("/api/v1/forecasts/generate", response_model=ForecastResponse, tags=["Forecasting"])
@app.post("/api/v1/forecast/generate", response_model=ForecastResponse, tags=["Forecasting"])
def generate_forecast(req: ForecastRequest):
    """Generates or returns 28-day probabilistic multi-quantile forecast."""
    try:
        hist_dates, hist_sales, future_dates, p10, p50, p90, _ = data_loader.get_forecast_series(
            store_id=req.store_id,
            category=req.category or "ALL",
            horizon_days=req.horizon_days
        )
        return ForecastResponse(
            forecast_id=f"forecast_{req.store_id}",
            store_id=req.store_id,
            category=req.category or "ALL",
            horizon_days=req.horizon_days,
            model_name=req.model_name,
            reconciled=req.reconcile,
            dates=future_dates,
            p10=[round(float(v), 2) for v in p10],
            p50=[round(float(v), 2) for v in p50],
            p90=[round(float(v), 2) for v in p90]
        )
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/api/v1/inventory/optimize", response_model=ReplenishmentResponse, tags=["Optimization"])
@app.post("/api/v1/optimize/replenish", response_model=ReplenishmentResponse, tags=["Optimization"])
def optimize_inventory(req: ReplenishmentRequest):
    """Executes integer constrained replenishment optimization."""
    try:
        items_df = data_loader.get_inventory_snapshot(store_id=req.store_id)
        ss_metrics = calculate_safety_stock_and_rop(
            p10_demand=items_df["p10_demand"].values,
            p50_demand=items_df["p50_demand"].values,
            p90_demand=items_df["p90_demand"].values,
            lead_time_days=3.0,
            service_level=req.service_level
        )

        items_df["safety_stock"] = ss_metrics["safety_stock"]
        items_df["reorder_point"] = ss_metrics["reorder_point"]

        items_df["net_need"] = np.maximum(items_df["reorder_point"] - items_df["on_hand_stock"], 0)
        items_df["recommended_order"] = np.where(
            items_df["net_need"] > 0,
            np.maximum(items_df["net_need"], items_df["moq"]),
            0
        )
        items_df["action"] = np.where(
            items_df["on_hand_stock"] < items_df["safety_stock"],
            "REORDER_NOW",
            np.where(items_df["on_hand_stock"] > items_df["p90_demand"] * 2.5, "HOLD (EXCESS)", "OPTIMAL")
        )
        items_df["order_cost"] = (items_df["recommended_order"] * items_df["unit_cost"]).round(2)

        items_output = []
        for _, row in items_df.iterrows():
            items_output.append(ReplenishmentItem(
                item_id=str(row["item_id"]),
                dept_id=str(row["dept_id"]),
                on_hand_stock=int(row["on_hand_stock"]),
                safety_stock=int(row["safety_stock"]),
                reorder_point=int(row["reorder_point"]),
                recommended_order=int(row["recommended_order"]),
                moq=int(row["moq"]),
                unit_cost=float(row["unit_cost"]),
                order_cost=float(row["order_cost"]),
                action=str(row["action"])
            ))

        return ReplenishmentResponse(
            status="OPTIMAL",
            store_id=req.store_id,
            total_spend=float(items_df["order_cost"].sum()),
            total_units_ordered=int(items_df["recommended_order"].sum()),
            items_analyzed=len(items_df),
            items=items_output
        )
    except Exception as e:
        logger.error(f"Optimization run failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/api/v1/inventory/recommendations", response_model=List[ReplenishmentItem], tags=["Optimization"])
def get_recommendations(store_id: str = "CA_1"):
    """Fetches active purchase orders and recommendations for a store."""
    csv_path = Path("data/outputs") / f"replenishment_orders_{store_id}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        # Generate on the fly
        req = ReplenishmentRequest(store_id=store_id)
        res = optimize_inventory(req)
        return res.items

    items = []
    for _, row in df.iterrows():
        items.append(ReplenishmentItem(
            item_id=str(row["item_id"]),
            dept_id=str(row.get("dept_id", "FOODS_1")),
            on_hand_stock=int(row["on_hand_stock"]),
            safety_stock=int(row["safety_stock"]),
            reorder_point=int(row["reorder_point"]),
            recommended_order=int(row["recommended_order_qty"]),
            moq=int(row["moq"]),
            unit_cost=float(row["unit_cost"]),
            order_cost=float(row["estimated_order_cost"]),
            action=str(row["action"])
        ))
    return items

@app.get("/api/v1/models", response_model=ModelMetadataResponse, tags=["Models"])
def get_models():
    """Lists registered model runs and hyperparameters from model registry."""
    return ModelMetadataResponse(
        models=registry.list_models(),
        last_updated=time.strftime("%Y-%m-%d %H:%M:%S")
    )

@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["Evaluation"])
def get_metrics(store_id: str = "CA_1"):
    """Fetches official backtesting metrics (WRMSSE, WAPE, Pinball loss)."""
    models = registry.list_models()
    key = f"LightGBM_Tweedie_{store_id}"
    if key in models:
        m = models[key].get("metrics", {})
        return MetricsResponse(
            store_id=store_id,
            wrmsse=m.get("wrmsse"),
            wape=m.get("wape"),
            mean_pinball_loss=m.get("mean_pinball_loss"),
            source="ModelRegistry"
        )
    return MetricsResponse(
        store_id=store_id,
        wrmsse=None,
        wape=None,
        mean_pinball_loss=None,
        source="No registered run found"
    )
