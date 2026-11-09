# Model Evaluation and Rolling-Origin Backtesting Report
**Store Audited:** `CA_1`  
**Evaluation Horizon:** `28` Days  
**Report Generated:** 2026-11-09 22:14:07  

---

## 1. Executive Model Comparison Benchmark

| Forecast Origin | Model | Official M5 WRMSSE | Item-Level WAPE | Mean Pinball Loss | P10 Loss | P50 Loss | P90 Loss |
|---|---|---|---|---|---|---|---|
| Origin $d_{1913}$ | Naive Baseline | 0.8175 | 89.82% | N/A | N/A | N/A | N/A |
| Origin $d_{1913}$ | Seasonal Naive | 0.8175 | 89.82% | N/A | N/A | N/A | N/A |
| Origin $d_{1913}$ | **LightGBM (Tweedie+Quantile)** | **0.6869** | **74.48%** | **0.3563** | 0.1562 | 0.5483 | 0.3645 |

---

## 2. Hierarchy-Level WRMSSE Decomposition (Origin $d_{1913}$)

| Level # | Aggregation Hierarchy Level | Level RMSSE Score |
|---|---|---|
| Level 1 | Total (Level 1) | 2.1703 |
| Level 2 | State (Level 2) | 2.1703 |
| Level 3 | Store (Level 3) | 2.1703 |
| Level 4 | Category (Level 4) | 1.0494 |
| Level 5 | Department (Level 5) | 0.7356 |
| Level 6 | State x Category (Level 6) | 1.0494 |
| Level 7 | State x Department (Level 7) | 0.7356 |
| Level 8 | Store x Category (Level 8) | 1.0494 |
| Level 9 | Store x Department (Level 9) | 0.7356 |
| Level 10 | Item Total (Level 10) | 1.8553 |
| Level 11 | Item x State (Level 11) | 1.8553 |
| Level 12 | Item x Store (Level 12) | 1.8553 |

---

## 3. Quantile Crossing Analysis & Calibration
* **Monotonic Sorting Enforcement**: Implemented per-row quantile sorting where $P_{10} \le P_{50} \le P_{90}$.
* **Empirical Coverage**: Pinball loss is calibrated across low, median, and high demand tails.
