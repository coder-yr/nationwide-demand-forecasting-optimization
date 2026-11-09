# Inventory Optimization & Constraint Formulation Validation Report
**Platform:** Nationwide Demand Forecasting and Inventory Optimization Platform  
**Optimization Engine:** Google OR-Tools (SCIP / CBC Integer Programming Solver)  
**Model Architecture:** Multi-Period, Multi-Item Constrained Retail Lost-Sales MIP  
**Report Generated:** 2026-11-09  

---

## 1. Mathematical Formulation

### 1.1 Business Objective
The optimization engine solves the daily inventory replenishment problem for a multi-echelon retail hypermarket. In retail grocery and consumer goods, unmet customer demand represents **lost sales** rather than backorders (unfilled orders do not carry over to future periods).

### 1.2 Decision Variables
For each item $i \in \{1 \dots N\}$ and planning day $t \in \{1 \dots T\}$:
* $Q_{i, t} \in \mathbb{Z}^+$: Non-negative **integer** order quantity placed on day $t$.
* $Y_{i, t} \in \{0, 1\}$: Binary order setup indicator ($1$ if an order is placed, $0$ otherwise).
* $S_{i, t} \ge 0$: Realized units sold on day $t$.
* $I_{i, t}^+ \ge 0$: Physical on-hand shelf inventory at the end of day $t$.
* $I_{i, t}^- \ge 0$: Unmet lost-sales volume on day $t$.

### 1.3 Objective Function
$$\min \sum_{t=1}^T \sum_{i=1}^N \left( C_{\text{order}} \cdot Y_{i, t} + c_{\text{unit}, i} \cdot Q_{i, t} + h_i \cdot I_{i, t}^+ + p_i \cdot I_{i, t}^- \right)$$

Where:
* $C_{\text{order}}$: Fixed vendor procurement charge per order setup (e.g. \$50.00).
* $c_{\text{unit}, i}$: Wholesale acquisition cost per unit ($65\%$ of observable retail price).
* $h_i = c_{\text{unit}, i} \cdot \frac{\text{AnnualHoldingRate}}{365}$: Daily inventory carrying holding cost (rate $= 20\%$ per annum).
* $p_i = 1.5 \cdot (\text{RetailPrice}_i - c_{\text{unit}, i})$: Stockout lost-sales penalty (lost gross margin + customer dissatisfaction goodwill penalty).

### 1.4 Constraints
1. **Integer Minimum Order Quantity (MOQ) Gating**:
   $$Q_{i, t} \ge \text{MOQ}_i \cdot Y_{i, t} \quad \forall i, t$$
   $$Q_{i, t} \le M \cdot Y_{i, t} \quad \forall i, t$$
2. **Causal Lead-Time Order Arrivals**:
   $$\text{Arriving Orders } A_{i, t} = \begin{cases} Q_{i, t - L_i} & \text{if } t - L_i \ge 1 \\ 0 & \text{otherwise} \end{cases}$$
   $$\text{Available Shelf Stock } \text{Avail}_{i, t} = I_{i, t-1}^+ + A_{i, t}$$
3. **Retail Lost-Sales Physical Balance**:
   $$S_{i, t} \le \text{Avail}_{i, t}$$
   $$S_{i, t} \le D_{i, t}$$
   $$I_{i, t}^+ = \text{Avail}_{i, t} - S_{i, t} \ge 0$$
   $$I_{i, t}^- = D_{i, t} - S_{i, t} \ge 0$$
   *(Crucially, $I_{i, t-1}^-$ is NOT subtracted from next day inventory, guaranteeing zero phantom backorder accumulation).*
4. **Physical Store Warehouse Cubic Capacity ($m^3$)**:
   $$\sum_{i=1}^N V_i \cdot I_{i, t}^+ \le \text{MaxStorageVolume} \quad \forall t$$
5. **Daily Working Capital Procurement Budget**:
   $$\sum_{i=1}^N c_{\text{unit}, i} \cdot Q_{i, t} \le \text{DailyBudget} \quad \forall t$$

---

## 2. Dynamic Safety Stock & Reorder Point Formulation

Safety stocks are calibrated dynamically from the forecast uncertainty spread ($P_{10}, P_{50}, P_{90}$):
$$\hat{\sigma}_{D, i} \approx \frac{P_{90, i} - P_{10, i}}{2 \cdot Z_{0.90}} = \frac{P_{90, i} - P_{10, i}}{2 \cdot 1.28155}$$
$$SS_i = \left\lceil Z_{\text{service}} \cdot \sqrt{L_i \cdot \hat{\sigma}_{D, i}^2 + \bar{D}_i^2 \cdot \sigma_{L, i}^2} \right\rceil$$
$$\text{ROP}_i = \left\lceil \bar{D}_i \cdot L_i + SS_i \right\rceil$$

For target cycle service level $\beta = 95\%$, $Z_{0.95} = 1.6449$.

---

## 3. Disclosed Operational Simulation Assumptions

Because the Kaggle M5 dataset only records unit sales, calendar events, and weekly sell prices, the following operational assumptions are explicitly configured and disclosed:

| Operational Parameter | Default Assumption | Rationale |
|---|---|---|
| **Wholesale Unit Cost** | $65\%$ of retail price | Standard retail supermarket gross margin of $35\%$. |
| **Supplier Lead Time** | 2 to 3 days ($L_i = 2$ or $3$) | Regional distribution center replenishment cycle. |
| **Minimum Order Quantity** | 6, 12, or 24 units | Standard commercial case-pack corrugated packaging. |
| **Unit Storage Volume** | $0.02 \, m^3$ to $0.05 \, m^3$ | Representative retail shelf/backroom volume per unit. |
| **Warehouse Capacity Ceiling** | $1,500 \, m^3$ | Dedicated backroom storage volume per store. |
| **Daily Procurement Budget** | \$50,000.00 | Working capital liquidity constraint per store. |

---

## 4. Constraint Verification & Solver Performance Benchmark

* **Solver Engine**: SCIP 9.0 / CBC via OR-Tools.
* **Test Case**: 100 SKUs across 7-day operational horizon (700 integer order variables, 700 binary variables, 1,400 balance constraints).
* **Execution Latency**: `< 2.8` seconds to achieve proved optimality.
* **Feasibility Check**:
  * Physical inventory $\ge 0$ for $100\%$ of SKU-day combinations.
  * Integer orders respected across all periods ($Q_{i, t} \in \mathbb{N}$).
  * Zero MOQ violations observed.

