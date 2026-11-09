# Nationwide Demand Forecasting & Inventory Optimization
## Executive Walkthrough & Demonstration Guide

This guide provides end-to-end instructions for demonstrating the platform, capturing UI assets, and presenting the business value and engineering architecture to recruiters, engineering leaders, and supply chain stakeholders.

---

## 1. Quickstart: Launching the Control Tower

Ensure the virtual environment is activated and dependencies are installed.

```bash
# Terminal 1: Launch FastAPI Gateway
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Launch Plotly Dash Control Tower
python -m src.dashboard.app
```

Access Points:
- **Interactive Control Tower:** `http://localhost:8050`
- **Swagger REST API Documentation:** `http://localhost:8000/docs`

---

## 2. Dashboard UI Blueprint & Visual Assets Guide

When preparing presentation decks or GitHub screenshots, capture genuine views from the running Dash instance (`http://localhost:8050`) into `docs/assets/`:

### View 1: Executive Overview & Header
- **Elements to highlight:**
  - Header Title: *Nationwide Demand Forecasting & Inventory Optimization*
  - Provenance Badge: `⚡ Data Source: Active M5 Lake` (or `Demo Fallback` if raw data is unmounted)
  - Coherence Badge: `Hierarchical Coherence: Verified (12 Levels)`
  - Operational Assumptions Alert: Transparently discloses that inventory, MOQs, and lead times are simulated retail parameters.
- **Recommended Asset Path:** `docs/assets/01_dashboard_header.png`

### View 2: Global Hierarchical Filter Controls
- **Elements to highlight:**
  - `Target Store` dropdown (e.g., `CA_1`, `CA_2`, `TX_1`, `WI_1`)
  - `Category` dropdown (`ALL`, `FOODS`, `HOBBIES`, `HOUSEHOLD`)
  - `Department` dropdown (`FOODS_1` through `HOUSEHOLD_2`)
  - `Forecast Horizon` selector (7 days operational, 14 days tactical, 28 days monthly)
  - `Forecasting Model` toggle (LightGBM Tweedie+Quantile vs. Prophet Macro Aggregate)
  - Blue `Recompute Optimization` action button
- **Recommended Asset Path:** `docs/assets/02_filter_controls.png`

### View 3: Executive KPI Tiles
- **Elements to highlight:**
  - **Projected Demand:** Real-time sum of $P_{50}$ forecast over the selected horizon (e.g., `42,850 units`)
  - **Stockout Risk Exposure:** Critical items where $P_{90} > \text{On-Hand Stock}$
  - **Working Capital Requirement:** Total estimated cost for optimal replenishment orders
  - **Service Level Target:** Default 95% cycle-service target
- **Recommended Asset Path:** `docs/assets/03_kpi_cards.png`

### View 4: Hierarchical Forecast Fan Chart
- **Elements to highlight:**
  - Historical unit sales trajectory (solid blue line)
  - Point forecast $P_{50}$ (dashed black line)
  - Calibrated 80% prediction interval band ($P_{10}$ to $P_{90}$ light blue shading)
  - Demonstrates strict quantile monotonicity: $P_{10} \le P_{50} \le P_{90}$
- **Recommended Asset Path:** `docs/assets/04_forecast_fan_chart.png`

### View 5: Inventory Risk Matrix
- **Elements to highlight:**
  - 2D Scatter plot: X-axis represents current physical shelf stock; Y-axis represents 28-day $P_{90}$ peak demand.
  - Red markers indicating `High Stockout Risk` items (Demand exceeding on-hand inventory).
  - Green markers indicating `Balanced` stock.
  - Hover tooltips showing Item ID, Current Stock, $P_{90}$ Demand, and Recommended Reorder Quantity.
- **Recommended Asset Path:** `docs/assets/05_inventory_risk_matrix.png`

### View 6: Optimal Replenishment Table & What-If Simulator
- **Elements to highlight:**
  - Dynamic DataTable listing items, current stock, reorder point (ROP), safety stock, and recommended integer purchase orders ($Q \in \mathbb{N}$).
  - Highlighted status tags: `REORDER_NOW` (salmon red background) and `OPTIMAL` (soft green background).
  - What-If Sliders allowing live adjustment of Target Service Level (85% to 99%) and Daily Procurement Budget (\$10k to \$100k).
- **Recommended Asset Path:** `docs/assets/06_replenishment_orders.png`

---

## 3. 3–5 Minute Executive Demo Script

Use this verbatim script during technical portfolio presentations or interview walkthroughs:

### Segment 1: The Business Problem (0:00 – 0:45)
> *"In enterprise retail, inventory managers face a persistent trade-off: overstocking ties up millions in working capital and leads to spoilage, while understocking triggers stockouts and lost revenue. Most legacy retail systems make replenishment decisions using simple moving averages or unconstrained safety stock formulas.*
>
> *This platform solves both challenges by closing the loop between machine learning forecasting and operations research optimization. It ingests nationwide point-of-sale data, predicts future demand across all 12 hierarchy levels, and solves a constrained mixed-integer programming model to output exact, supplier-compliant purchase orders."*

### Segment 2: Data Engineering & Zero-Leakage Features (0:45 – 1:30)
> *"We benchmarked the system using the Walmart M5 dataset—encompassing 30,490 products across 10 stores in California, Texas, and Wisconsin. To process over 58 million transactional rows on standard hardware, I engineered a zero-copy type downcaster that reduces memory footprint by 85.7% and persists data into store-partitioned Snappy Parquets with DuckDB embedded querying.*
>
> *Crucially, we eliminated look-ahead bias: historical prices prior to product launches are causally imputed without using future price changes, and all lag and rolling statistical features are shifted by at least 28 days to match our direct forecasting horizon."*

### Segment 3: Multi-Tier Forecasting & 12-Level Coherence (1:30 – 2:30)
> *"Demand in retail is intermittent and highly skewed. We deploy a multi-tiered architecture:*
> 1. *For store-item series, we use LightGBM with Tweedie deviance for point estimates alongside pinball loss for $P_{10}, P_{50}$, and $P_{90}$ quantiles, backed by monotonic sorting to prevent quantile crossing.*
> 2. *For macro aggregates (national and state categories), we implement Prophet additive decomposition.*
> 3. *For high-velocity Class-A Pareto drivers, we provide Temporal Fusion Transformer (TFT) modeling via Darts.*
>
> *To evaluate models, we implemented the official M5 WRMSSE across all 12 hierarchy levels. Our LightGBM pipeline achieved a WRMSSE of 0.6869, outperforming naive baselines by 16%. We then pass these predictions through our 12-level hierarchical reconciler, mathematically verifying that child forecasts sum to their parent series with zero discrepancy."*

### Segment 4: Mathematical Inventory Optimization (2:30 – 3:30)
> *"A forecast is only as good as the decision it drives. Instead of unrealistic backorder models, our prescriptive engine solves a retail lost-sales mixed-integer program using Google OR-Tools.*
>
> *The optimizer balances inventory holding costs against stockout penalties while enforcing hard business constraints: supplier lead times, minimum order quantities (MOQ), warehouse volumetric capacity, and daily store procurement budgets. The decision variables are strictly integer order batches, ensuring actionable purchase orders."*

### Segment 5: Control Tower Walkthrough (3:30 – 4:30)
> *"Finally, everything surfaces in this Plotly Dash Control Tower and FastAPI backend. Planners can slice by store, category, and department, inspect calibrated prediction intervals on the fan chart, pinpoint at-risk items on the risk matrix, and perform what-if scenario simulations to evaluate how changing target service levels impacts working capital.*
>
> *The entire system is containerized with Docker, verified by an 18-test automated suite, and tracked with continuous integration."*

---

## 4. API Verification via cURL / Swagger

You can demonstrate the backend headless capabilities directly:

```bash
# Check system health
curl -s http://localhost:8000/health | jq

# Retrieve 12-level hierarchy structure
curl -s http://localhost:8000/api/v1/hierarchy | jq

# Request 28-day probabilistic forecast for store CA_1 (FOODS category)
curl -X POST http://localhost:8000/api/v1/forecast/generate \
  -H "Content-Type: application/json" \
  -d '{"store_id": "CA_1", "category": "FOODS", "horizon_days": 28, "model_name": "lgbm", "reconcile": true}' | jq

# Execute MIP inventory replenishment solver
curl -X POST http://localhost:8000/api/v1/inventory/optimize \
  -H "Content-Type: application/json" \
  -d '{"store_id": "CA_1", "horizon_days": 28, "budget_limit": 50000, "service_level": 0.95}' | jq
```

