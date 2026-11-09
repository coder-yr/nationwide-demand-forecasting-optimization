"""API Integration Tests for FastAPI Gateway."""
import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("healthy", "degraded")
    assert "available_stores" in data
    assert len(data["available_stores"]) == 10

def test_hierarchy_endpoint():
    res = client.get("/api/v1/hierarchy")
    assert res.status_code == 200
    nodes = res.json()
    assert len(nodes) == 3  # CA, TX, WI
    states = [n["state_id"] for n in nodes]
    assert "CA" in states and "TX" in states and "WI" in states

def test_forecast_generate_and_list():
    # Generate forecast
    res_gen = client.post("/api/v1/forecasts/generate", json={
        "store_id": "CA_1",
        "horizon_days": 14,
        "model_name": "lgbm"
    })
    assert res_gen.status_code == 200
    f_data = res_gen.json()
    assert len(f_data["dates"]) == 14
    assert len(f_data["p50"]) == 14

    # List forecasts
    res_list = client.get("/api/v1/forecasts")
    assert res_list.status_code == 200
    assert isinstance(res_list.json(), list)

def test_inventory_optimize_and_recommendations():
    # Optimize
    res_opt = client.post("/api/v1/inventory/optimize", json={
        "store_id": "CA_1",
        "horizon_days": 7,
        "service_level": 0.95
    })
    assert res_opt.status_code == 200
    opt_data = res_opt.json()
    assert opt_data["status"] == "OPTIMAL"
    assert opt_data["items_analyzed"] > 0
    assert len(opt_data["items"]) > 0

    # Get recommendations
    res_rec = client.get("/api/v1/inventory/recommendations?store_id=CA_1")
    assert res_rec.status_code == 200
    rec_items = res_rec.json()
    assert len(rec_items) > 0
    assert "recommended_order" in rec_items[0]

def test_models_and_metrics_endpoints():
    res_models = client.get("/api/v1/models")
    assert res_models.status_code == 200
    assert "models" in res_models.json()

    res_metrics = client.get("/api/v1/metrics?store_id=CA_1")
    assert res_metrics.status_code == 200
    m_data = res_metrics.json()
    assert m_data["store_id"] == "CA_1"

def test_missing_forecast_404():
    res = client.get("/api/v1/forecasts/NON_EXISTENT_ID_9999")
    assert res.status_code == 404

