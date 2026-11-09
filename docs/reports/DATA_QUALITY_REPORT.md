# Data Quality and Ingestion Audit Report
**Store Audited:** `CA_1`  
**Dataset Source:** Walmart M5 Forecasting (Kaggle)  
**Report Generated:** 2026-11-09 22:03:47  

---

## 1. Executive Summary & Ingestion Benchmark

| Metric | Measured Value |
|---|---|
| **Total Rows in Partition** | 1,112,885 |
| **Unique Items** | 3,049 |
| **Date Range** | 2015-05-24 to 2016-05-22 |
| **Total Days Tracked** | 365 days |
| **Compressed Parquet Size** | 1.58 MB (Snappy compressed) |
| **Uncompressed Memory Footprint** | ~749.39 MB |
| **Downcasting Compression Ratio** | >85% memory reduction |

---

## 2. Data Validation & Integrity Checks

| Integrity Test | Result | Expected | Status |
|---|---|---|---|
| **Duplicate Keys (`store_id`, `item_id`, `date`)** | 0 | 0 | PASSED |
| **Critical Nulls (`store_id`, `item_id`, `date`, `sales`)** | 0 | 0 | PASSED |
| **Negative Unit Sales (`sales < 0`)** | 0 | 0 | PASSED |
| **Active Item Missing Prices** | 0 | 0 | PASSED |
| **Active Item Non-Positive Prices (`price <= 0`)** | 0 | 0 | PASSED |
| **Overall Validation Status** | VALID | VALID | PASSED |

---

## 3. Sparsity & Intermittent Demand Profile

* **Overall Zero-Sales Ratio**: `55.35%` of total rows have `sales == 0`.
* **Active-Lifecycle Zero-Sales Ratio**: `55.28%` of days after item launch have zero sales.
* **Interpretation**: The M5 series exhibits extreme intermittent and zero-inflated demand, confirming the requirement for a **Tweedie objective** ($p \in [1.1, 1.3]$) and **Quantile Pinball losses** over standard Gaussian assumptions.

---

## 4. Causal Price Leakage Remediation Documentation

* **Previous Defect**: Pre-launch prices were backward-filled (`.bfill()`), leaking future introductory prices into periods prior to item catalog entry.
* **Remediation Implemented**:
  1. Identified each item's earliest recorded sell price date (`first_price_d`).
  2. Flagged periods prior to `first_price_d` with `is_active_item = 0` and `sell_price = 0.0`.
  3. Applied **forward-fill (`.ffill()`) strictly forward in time** for subsequent active price maintenance.
  4. Zero future prices are leaked into historical training matrices.

---

## 5. Parquet Schema Specification

| Column Name | Physical Data Type | Logical Type | Description |
|---|---|---|---|
| `id` | `category` | String ID | Unique series identifier (`item_id`_`store_id`_evaluation) |
| `item_id` | `category` | Categorical | Specific product SKU |
| `dept_id` | `category` | Categorical | Department (`FOODS_1` to `HOUSEHOLD_2`) |
| `cat_id` | `category` | Categorical | Category (`FOODS`, `HOBBIES`, `HOUSEHOLD`) |
| `store_id` | `category` | Categorical | Store identifier (e.g. `CA_1`) |
| `state_id` | `category` | Categorical | State (`CA`, `TX`, `WI`) |
| `d` | `category` | String | Day label (`d_1` to `d_1941`) |
| `d_num` | `uint16` | Integer | Numeric day sequence (1 to 1941) |
| `sales` | `uint16` | Integer Count | Daily unit sales volume |
| `date` | `timestamp[us]` | Datetime | Calendar date |
| `wm_yr_wk` | `uint16` | Integer | Walmart calendar week index |
| `weekday` | `category` | String | Name of weekday |
| `wday` | `uint8` | Integer | Day index (1=Sat, 2=Sun, ..., 7=Fri) |
| `month` | `uint8` | Integer | Month (1 to 12) |
| `year` | `uint16` | Integer | Calendar year |
| `event_name_1` | `category` | Categorical | Primary holiday/event name |
| `event_type_1` | `category` | Categorical | Primary holiday category |
| `event_name_2` | `category` | Categorical | Secondary holiday name |
| `event_type_2` | `category` | Categorical | Secondary holiday category |
| `has_event` | `uint8` | Boolean Flag | Indicator (1 if holiday on date, 0 otherwise) |
| `snap_current_state`| `uint8` | Boolean Flag | SNAP food stamp disbursement indicator |
| `sell_price` | `float32` | Currency ($) | Weekly observable retail selling price |
| `is_active_item` | `uint8` | Boolean Flag | 1 if item launched/active; 0 if pre-launch |
