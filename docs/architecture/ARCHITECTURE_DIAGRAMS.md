# Nationwide Demand Forecasting & Inventory Optimization
## Comprehensive System Architecture Diagrams

This document contains 11 verified Mermaid architectural diagrams modeling the exact data flow, machine learning lifecycle, mathematical reconciliation, optimization mechanics, API routing, and deployment topologies implemented in the platform.

---

### 1. Overall System Architecture

```mermaid
flowchart TD
    subgraph Data_Layer["1. Data Ingestion & Storage"]
        Raw["Raw M5 CSVs (Sales, Calendar, Prices)"] --> Ingest["M5DataPipeline Ingestion Engine"]
        Ingest --> Downcast["Memory Downcaster (85.7% RAM Reduction)"]
        Downcast --> ParquetLake["Store-Partitioned Snappy Parquet Lake"]
    end

    subgraph Feature_Layer["2. Causal Feature Engineering"]
        ParquetLake --> DuckDBEngine["DuckDB Query Engine"]
        DuckDBEngine --> Lags["Lags (28..364) & Shifted Rolling Means"]
        DuckDBEngine --> Prices["Causal Price Imputation & Elasticity"]
        DuckDBEngine --> Calendar["Cyclical Encodings, SNAP & Events"]
        Lags & Prices & Calendar --> FeatMatrix["Leakage-Free Feature Matrix (T >= 28)"]
    end

    subgraph Model_Layer["3. Multi-Tier Forecasting"]
        FeatMatrix --> LGBM["LightGBM Tweedie & Multi-Quantile"]
        FeatMatrix --> Prophet["Macro Prophet Models (Levels 1-5)"]
        FeatMatrix --> TFT["Darts TFT (Pareto Class-A Series)"]
        LGBM --> QCross["Monotonic Quantile Sorter (P10 <= P50 <= P90)"]
    end

    subgraph Reconcile_Layer["4. Coherence Reconciliation"]
        QCross --> Reconciler["12-Level Hierarchical Reconciler"]
        Reconciler --> CoherentForecasts["Mathematically Coherent Forecasts"]
    end

    subgraph Optimization_Layer["5. Prescriptive Optimization"]
        CoherentForecasts --> ROP["Dynamic Safety Stock & Lead-Time Demand"]
        ROP --> ORTools["Google OR-Tools Lost-Sales MIP Solver"]
        ORTools --> Orders["Constrained Replenishment Orders (Q in N)"]
    end

    subgraph Serving_Layer["6. Serving & Analytics"]
        CoherentForecasts & Orders --> FastAPI["FastAPI REST Service (9 Endpoints)"]
        CoherentForecasts & Orders --> Dash["Plotly Dash Control Tower UI"]
    end
```

---

### 2. Data Ingestion Pipeline

```mermaid
flowchart LR
    CSV_Sales["sales_train_evaluation.csv"] --> FilterStore["Filter Target Store (e.g., CA_1)"]
    CSV_Cal["calendar.csv"] --> CleanCal["Date Indexing & SNAP Flags"]
    CSV_Price["sell_prices.csv"] --> CausalPrice["Zero-Leakage Price Imputation"]

    FilterStore --> Melt["Melt Wide Days (d_1..d_1941) to Long Format"]
    Melt --> MergeCal["Join Calendar on d_id"]
    MergeCal --> MergePrice["Join Prices on (item_id, wm_yr_wk)"]

    CausalPrice --> MergePrice
    CleanCal --> MergeCal

    MergePrice --> DowncastLogic["Downcast: float64 to float32, int64 to uint16/category"]
    DowncastLogic --> WriteParquet["Write Store Parquet (Snappy Compressed)"]
```

---

### 3. Feature Engineering Workflow

```mermaid
flowchart TD
    RawStore["Input Store Data (t = 1..T)"] --> TimeBoundary["Strict Forecast Horizon Boundary (t <= T - 28)"]
    
    subgraph Lag_Engine["Lag & Rolling Aggregators"]
        TimeBoundary --> Lags["Lags: [28, 29, 30, 35, 42, 56, 84, 364]"]
        TimeBoundary --> ShiftedRoll["Shift(28) -> Rolling Mean & Std [7, 14, 28, 56]"]
        TimeBoundary --> EWMA["Shift(28) -> Exponential Moving Averages (0.1, 0.3)"]
    end

    subgraph Price_Engine["Price Signal Extractor"]
        TimeBoundary --> PriceDiff["Price 7-Day & 28-Day Differences"]
        TimeBoundary --> RelDiscount["Relative Discount vs 28-Day Max"]
        TimeBoundary --> PriceMomentum["Price Momentum Ratio"]
    end

    subgraph Calendar_Engine["Calendar Signals"]
        TimeBoundary --> Cyclical["Sine/Cosine Transforms (Day-of-Week, Month, Week)"]
        TimeBoundary --> Events["Binary Primary & Secondary Event Flags"]
        TimeBoundary --> SNAP["State SNAP Eligibility Indicator"]
    end

    Lags & ShiftedRoll & EWMA & PriceDiff & RelDiscount & PriceMomentum & Cyclical & Events & SNAP --> Concat["Assemble Training Matrix (Zero Look-Ahead)"]
```

---

### 4. Forecast Model Training Lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant Pipeline as Train Pipeline
    participant Feats as Feature Generator
    participant LGBM as LightGBM Regressor
    participant Sorter as Monotonic Sorter
    participant Registry as Model Registry
    participant Storage as Disk (data/outputs/models/)

    Pipeline->>Feats: Request train matrix (T <= origin)
    Feats-->>Pipeline: Return (X_train, y_train, X_val, y_val)
    Pipeline->>LGBM: Train Tweedie Point Model (p=1.15)
    LGBM-->>Pipeline: Point Model Booster
    Pipeline->>LGBM: Train Quantile Models (alpha = 0.10, 0.50, 0.90)
    LGBM-->>Pipeline: Quantile Boosters
    Pipeline->>Sorter: Evaluate Validation Predictions
    Sorter-->>Pipeline: Enforce P10 <= P50 <= P90
    Pipeline->>Registry: Register Model Version, Hyperparams, Metrics
    Registry->>Storage: Serialize Boosters & registry.json
```

---

### 5. Rolling-Origin Walk-Forward Backtesting

```mermaid
flowchart TD
    FullTimeline["M5 Historical Timeline (d_1 to d_1941)"]
    
    subgraph Fold_1["Cutoff 1: Origin d_1885"]
        Train1["Train Window: d_1 to d_1885"] --> Forecast1["28-Day Test Horizon: d_1886 to d_1913"]
        Forecast1 --> Eval1["Compute WRMSSE, WAPE, Pinball"]
    end

    subgraph Fold_2["Cutoff 2: Origin d_1913"]
        Train2["Train Window: d_1 to d_1913"] --> Forecast2["28-Day Test Horizon: d_1914 to d_1941"]
        Forecast2 --> Eval2["Compute WRMSSE, WAPE, Pinball"]
    end

    FullTimeline --> Fold_1
    FullTimeline --> Fold_2
    Eval1 & Eval2 --> AggregateMetrics["Aggregate Mean Rolling Metrics across Origins"]
```

---

### 6. Official M5 12-Level Hierarchy

```mermaid
flowchart TD
    L1["Level 1: Total (National)"]
    
    L2["Level 2: State (CA, TX, WI)"]
    L3["Level 3: Store (10 Stores)"]
    
    L4["Level 4: Category (HOBBIES, HOUSEHOLD, FOODS)"]
    L5["Level 5: Department (7 Departments)"]
    
    L6["Level 6: State x Category (9 Series)"]
    L7["Level 7: State x Department (21 Series)"]
    
    L8["Level 8: Store x Category (30 Series)"]
    L9["Level 9: Store x Department (70 Series)"]
    
    L10["Level 10: Item (3,049 Series)"]
    L11["Level 11: State x Item (9,147 Series)"]
    L12["Level 12: Store x Item (30,490 Series - Bottom Tier)"]

    L1 --> L2 --> L3
    L1 --> L4 --> L5
    L2 & L4 --> L6
    L2 & L5 --> L7
    L3 & L4 --> L8
    L3 & L5 --> L9
    L5 --> L10
    L2 & L10 --> L11
    L3 & L10 --> L12
```

---

### 7. Hierarchical Reconciliation Workflow

```mermaid
flowchart TD
    BottomForecasts["Bottom-Level Forecasts (Level 12: Store x Item)"] --> BottomUpAgg["Summation Operator S (Child-to-Parent Aggregation)"]
    
    BottomUpAgg --> L11["Level 11: State x Item"]
    BottomUpAgg --> L10["Level 10: Item"]
    BottomUpAgg --> L9["Level 9: Store x Department"]
    BottomUpAgg --> L8["Level 8: Store x Category"]
    BottomUpAgg --> L7["Level 7: State x Department"]
    BottomUpAgg --> L6["Level 6: State x Category"]
    BottomUpAgg --> L5["Level 5: Department"]
    BottomUpAgg --> L4["Level 4: Category"]
    BottomUpAgg --> L3["Level 3: Store"]
    BottomUpAgg --> L2["Level 2: State"]
    BottomUpAgg --> L1["Level 1: National Total"]

    L11 & L10 & L9 & L8 & L7 & L6 & L5 & L4 & L3 & L2 & L1 --> CoherenceCheck["Exhaustive Coherence Validation (Level-by-Level Key Match)"]
    CoherenceCheck --> VerifiedOutputs["Coherent Multi-Tier Forecast Matrices (Discrepancy = 0.000000)"]
```

---

### 8. Inventory Optimization Workflow

```mermaid
flowchart TD
    subgraph Inputs["1. Supply Chain Input Parameters"]
        P50["Demand Forecast P50 (D_it)"]
        LeadTime["Supplier Lead Time (L_i days)"]
        Costs["Holding Cost (h_i) & Shortage Penalty (p_i)"]
        MOQ["Minimum Order Quantity (MOQ_i)"]
        Budget["Daily Capital Budget (B_t)"]
        Capacity["Warehouse Volumetric Limit (C_vol)"]
        Inv0["Current Physical Shelf Stock (I_i0)"]
    end

    subgraph MIP["2. OR-Tools CBC Mixed-Integer Formulation"]
        VarQ["Integer Order Quantities: Q_it in N"]
        VarY["Binary Order Setup: Y_it in {0, 1}"]
        InvPos["Physical Stock Non-Negativity: I_it >= 0"]
        LostSales["Lost Sales Shortage: S_it >= 0"]
        
        Obj["Minimize Total Cost = Holding + Shortage + Fixed Setup"]
        Cons1["Inventory Balance: I_it = I_i,t-1 + Q_i,t-L - D_it + S_it"]
        Cons2["MOQ Enforcement: Q_it >= MOQ_i * Y_it"]
        Cons3["Budget Cap: Sum(cost_i * Q_it) <= B_t"]
        Cons4["Capacity Cap: Sum(vol_i * I_it) <= C_vol"]
    end

    subgraph Outputs["3. Prescriptive Decisions"]
        Solve["MIP Optimization Engine Solve"]
        Orders["Recommended Replenishment Orders (Q_it)"]
        Trajectories["Projected Inventory Trajectories (I_it)"]
        Stockouts["Predicted Stockout Risk Events (S_it > 0)"]
    end

    Inputs --> MIP
    MIP --> Solve
    Solve --> Orders & Trajectories & Stockouts
```

---

### 9. FastAPI Architecture

```mermaid
flowchart TD
    Client["Client / Front-End / Integration Services"] --> Router["FastAPI Gateway (Uvicorn / Port 8000)"]
    
    subgraph Endpoints["REST API Routing"]
        Router --> EP1["GET /health"]
        Router --> EP2["GET /api/v1/hierarchy"]
        Router --> EP3["GET /api/v1/forecast/{store_id}"]
        Router --> EP4["POST /api/v1/forecast/generate"]
        Router --> EP5["POST /api/v1/inventory/optimize"]
        Router --> EP6["GET /api/v1/inventory/recommendations"]
        Router --> EP7["GET /api/v1/models"]
        Router --> EP8["GET /api/v1/metrics"]
        Router --> EP9["GET /docs (Swagger / OpenAPI UI)"]
    end

    subgraph Service_Logic["Execution Engines"]
        EP2 & EP3 & EP4 --> DataLoader["DuckDB Data Loader & Registry"]
        EP5 & EP6 --> Solver["OR-Tools CBC Inventory Solver"]
        EP7 & EP8 --> RegMetrics["Model Registry & Metrics Provider"]
    end

    DataLoader --> ParquetLake["data/processed/stores/*.parquet"]
    Solver --> BaseConfig["config/base_config.yaml"]
```

---

### 10. Dashboard Data Flow

```mermaid
flowchart LR
    User["Operations User"] --> DashUI["Plotly Dash Control Tower UI"]
    
    subgraph Controls["Reactive UI Filters"]
        StoreSelect["Store Dropdown"]
        CatSelect["Category Dropdown"]
        DeptSelect["Department Dropdown"]
        HorizonSlider["Horizon Slider (7..28 days)"]
    end

    DashUI --> Controls
    Controls --> Callbacks["Dash Reactive Callbacks"]

    subgraph Engine["DuckDB Analytical Backend"]
        Callbacks --> DuckDB["DuckDB SQL Reader"]
        DuckDB --> Parquet["Store Parquet Files"]
        Callbacks --> Solver["MIP Inventory Solver"]
    end

    Engine --> Outputs["Reactive Dashboard Visualizations"]
    Outputs --> FanChart["Forecast Fan Chart (P10, P50, P90 Bands)"]
    Outputs --> RiskMatrix["Stockout Risk Matrix (Scatter Plot)"]
    Outputs --> ReorderTable["Replenishment Purchase Order Grid"]
    Outputs --> ProvBadge["Live vs Demo Provenance Indicator"]
```

---

### 11. Production Containerized Deployment Architecture

```mermaid
flowchart TD
    subgraph Host["Docker Host (Linux / Windows / Cloud VM)"]
        DockerCompose["docker-compose.yml"]
        
        subgraph Net["Internal Bridge Network"]
            API_Container["Container: nationwide-forecast-api (Port 8000)"]
            Dash_Container["Container: nationwide-forecast-dashboard (Port 8050)"]
        end

        subgraph Volumes["Persistent Volume Mounts"]
            VolData["data/ (Parquet Lake & Models)"]
            VolConfig["config/ (base_config.yaml)"]
        end

        DockerCompose --> API_Container
        DockerCompose --> Dash_Container
        VolData & VolConfig -.-> API_Container
        VolData & VolConfig -.-> Dash_Container
        Dash_Container -->|HTTP REST| API_Container
    end

    InternetUsers["Supply Chain Planners / Enterprise Systems"] -->|HTTP / 8050| Dash_Container
    ExternalAPIs["ERP / Automated Reordering Systems"] -->|HTTP REST / 8000| API_Container
```

