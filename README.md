# Nationwide Demand Forecasting & Inventory Optimization Platform

An industry-grade supply chain intelligence platform combining **hierarchical probabilistic time-series forecasting** and **constrained mathematical optimization** on the **Walmart M5 forecasting dataset**.

---

## 🌟 Key Capabilities

1. **Massive-Scale Time Series Ingestion**:
   * Downcasts 64-bit Pandas types to `category`, `uint16`, `int16`, and `float32`, reducing 58M+ row memory usage by **70% to 85%**.
   * Emits store-partitioned Snappy Parquet files with embedded DuckDB querying.
2. **Multi-Model Forecasting Tiering**:
   * **LightGBM**: Tweedie point forecasts and multi-quantile regressors ($P_{10}, P_{50}, P_{90}$) across all 30,490 bottom-level store-item series.
   * **Prophet**: Macro-level additive models for National, State, Department, and Category aggregates (Levels 1–5).
   * **Temporal Fusion Transformer (TFT)**: Multi-horizon attention architecture for high-velocity Pareto Class-A clusters via Darts and PyTorch Forecasting.
3. **Hierarchical Reconciliation**:
   * Guarantees mathematical coherence between bottom-level and national totals via **Bottom-Up** and **Structural MinT-WLS** reconciliation.
4. **Constrained Inventory Optimization**:
   * Formulates mixed-integer programs (MIP) in **Google OR-Tools** balancing holding costs, lost-sales shortage penalties, supplier lead times, minimum order quantities (MOQ), and working capital limits.
5. **Interactive Decision Support Control Tower**:
   * **Plotly Dash** portal featuring forecast fan charts with 80% confidence intervals ($P_{10}-P_{90}$), inventory risk matrices, purchase order grids, and what-if parameter sliders.
   * **FastAPI Gateway**: High-throughput REST endpoints with OpenAPI/Swagger interactive documentation.

---

## 🏗️ Project Architecture

```
nationwide-demand-forecast/
├── config/
│   └── base_config.yaml         # Centralized hyperparameters & constraints
├── data/
│   ├── raw/                     # Original M5 CSVs (calendar, prices, sales)
│   ├── processed/               # Store-partitioned Snappy Parquet lake
│   └── outputs/                 # Forecast dumps & purchase orders
├── src/
│   ├── data/
│   │   ├── downcaster.py        # Type downcasting & memory optimization
│   │   └── pipeline.py          # Ingestion & wide-to-long ETL
│   ├── features/
│   │   ├── lag_generator.py     # Leakage-free lags & EWMA (>=28 days)
│   │   ├── price_feats.py       # Elasticity, discounts & momentum
│   │   └── calendar_feats.py    # Events, cyclical month/wday encodings
│   ├── models/
│   │   ├── base.py              # Base Forecaster ABC interface
│   │   ├── lgbm_forecaster.py   # Tweedie & Quantile LightGBM
│   │   ├── prophet_engine.py    # Macro aggregate Prophet models
│   │   ├── tft_forecaster.py    # PyTorch Forecasting / Darts TFT
│   │   └── reconciler.py        # Bottom-Up & MinT reconciliation
│   ├── optimization/
│   │   ├── safety_stock.py      # Dynamic safety stock & ROP calculator
│   │   └── solver.py            # Google OR-Tools MIP replenishment solver
│   ├── api/
│   │   ├── app.py               # FastAPI service gateway
│   │   └── schemas.py           # Pydantic request/response models
│   ├── dashboard/
│   │   ├── app.py               # Dash application entrypoint
│   │   ├── layout.py            # Control tower layout & KPI cards
│   │   ├── callbacks.py         # Dynamic Plotly visualizations
│   │   └── data_loader.py       # Parquet DuckDB bridge
│   └── utils/
│       ├── logger.py            # Structured logging
│       └── metrics.py           # WRMSSE, WAPE, and Scaled Pinball Loss
├── tests/
│   ├── test_feature_leakage.py  # Zero-leakage verification
│   ├── test_metrics.py          # Metric accuracy tests
│   ├── test_optimizer.py        # OR-Tools constraint tests
│   └── test_reconciliation.py   # Mathematical coherence tests
├── conftest.py                  # Pytest environment bootstrap
├── pyproject.toml               # Build metadata & pytest configuration
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Ingest Data & Generate Parquet Lake
To process raw M5 CSVs into partitioned Parquet files:
```python
from src.data.pipeline import M5DataPipeline

pipeline = M5DataPipeline(history_days_window=365)
# Processes store CA_1 (1.1M rows downcasted to ~5 MB)
pipeline.process_store("CA_1")
```

### 2. Run Test Suite
Verify that all unit tests and invariant checks pass:
```bash
python -m pytest tests/ -p no:dash
```

### 3. Launch Plotly Dash Control Tower
Start the interactive operations portal:
```bash
python -m src.dashboard.app
```
Then navigate to: `http://localhost:8050`

### 4. Launch FastAPI REST Gateway
Run the API service:
```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs available at: `http://localhost:8000/docs`

