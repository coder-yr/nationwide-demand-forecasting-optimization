# Repository Presentation & Portfolio-Readiness Audit

**Date:** November 2026  
**Auditor:** Senior Software Engineer, Technical Writer & Developer Advocate  
**Target Repository:** `nationwide-demand-forecasting-optimization`  

---

## 1. Executive Summary

This forensic presentation audit evaluates the repository against tier-1 open-source standards, recruiter legibility, and technical credibility. The core application logic—including causal price imputation, 12-level M5 hierarchical reconciliation, mixed-integer lost-sales inventory replenishment, and FastAPI/Dash services—is fully implemented and passes all 18 automated tests. 

However, several documentation, GitHub presentation, and asset gaps were identified that detract from its production credibility and portfolio impact.

---

## 2. Forensic Audit Findings

| Category | Finding | Current State | Remediation Action |
| :--- | :--- | :--- | :--- |
| **1. Missing Documentation** | Contributing guidelines, issue templates, PR templates, and visual demo guides are missing. | Only basic README and technical reports exist in `docs/reports/`. | Create `CONTRIBUTING.md`, `.github/pull_request_template.md`, `.github/ISSUE_TEMPLATE/`, and `docs/DEMO_GUIDE.md`. |
| **2. Setup Commands** | Previous instructions lacked OS-specific instructions for Windows PowerShell vs. Linux/macOS. | Generic bash snippets only. | Provide explicitly tested Windows PowerShell and Linux/macOS bash commands for virtualenvs, CLI execution, and services. |
| **3. Outdated Descriptions** | Early design blueprints mentioned synthetic components that have since been replaced with empirical pipelines. | `docs/system_architecture_blueprint.md` has historical planning text. | Ensure `README.md` and `docs/architecture/` strictly document real implementations. |
| **4. Feature Claims** | Prophet and TFT deep-learning tiers were previously referenced without disclosing computational boundaries. | Modules `prophet_engine.py` and `tft_forecaster.py` exist, but full training on 30,490 series is unfeasible on CPU. | Explicitly classify capabilities: **Core Production** (LightGBM Tweedie/Quantile, MIP Solver, 12-Level Reconciler), **Macro Tier** (Prophet on Levels 1–5), and **Experimental/Pareto** (Darts TFT on Class-A). |
| **5. Architecture Diagrams** | Missing dedicated per-domain Mermaid diagrams for ingestion, feature pipelines, rolling backtests, and MIP optimization. | Only high-level diagrams in `CURRENT_ARCHITECTURE.md`. | Author a dedicated multi-view architecture document with 11 diagrams in `docs/architecture/ARCHITECTURE_DIAGRAMS.md`. |
| **6. Screenshots & Visual Assets** | No screenshots in repository root or `docs/assets/`. | Dash UI is fully functional but visually unrepresented in markdown previews. | Provide `docs/DEMO_GUIDE.md` detailing exact screen layouts, KPIs, walk-through script, and instructions for capturing assets. |
| **7. File References** | Minor path discrepancies in documentation (referencing `data/outputs/forecasts.csv` vs `data/outputs/models/`). | Some docs reference planned file names. | Standardize all paths to real workspace directories (`data/outputs/models/`, `data/processed/stores/`). |
| **8. Incomplete Installation** | Missing clear dependency prerequisites (e.g., C++ redistributable requirements on Windows for OpenMP/OR-Tools). | Missing troubleshooting notes for Windows DLL loading order. | Add platform prerequisite callouts to `README.md`. |
| **9. Test Documentation** | README did not break down the 18-test suite coverage or explain synthetic testing fixtures vs live data. | Generic `pytest` command only. | Document test suite structure, assertions, and execution benchmarks in README and test reports. |
| **10. Terminology** | Inconsistent usage between "backorders" vs "lost sales". | Mathematical engine uses retail lost-sales shelf dynamics ($I_{pos} \ge 0$), while early blueprints mentioned backorders. | Standardize on **retail lost-sales inventory model with non-negative physical stock**. |

---

## 3. Disclosed Operational Boundaries & Assumptions

To maintain absolute technical credibility, the following aspects are clearly disclosed across the repository:
1. **Walmart M5 Dataset Scope:** The M5 dataset strictly contains daily unit sales, calendar events, and weekly retail prices.
2. **Synthetic Inventory Fixtures:** Opening inventory, procurement unit costs, supplier lead times, minimum order quantities (MOQ), warehouse volumetric limits ($m^3$), and working capital constraints are configured via `config/base_config.yaml` as empirical supply chain simulation assumptions.
3. **Hardware & Execution Constraints:** Full 30,490-series LightGBM training requires partitioned execution. Single-store `CA_1` (1,114,845 records) is the benchmark standard.

---

## 4. Implementation Plan for Portfolio Hardening

1. **Phase 2:** Redesign `README.md` with modern layout, technology badges, Mermaid architecture, installation, evaluation metrics, and API/dashboard walkthrough.
2. **Phase 3:** Author 11 comprehensive Mermaid diagrams in `docs/architecture/ARCHITECTURE_DIAGRAMS.md`.
3. **Phase 4:** Author `docs/DEMO_GUIDE.md` with a structured 3–5 minute executive demo script.
4. **Phase 5:** Add `.github/ISSUE_TEMPLATE/` (bug & feature), `.github/pull_request_template.md`, and `CONTRIBUTING.md`.
5. **Phase 6:** Validate CI workflow compatibility on multi-Python matrix.
6. **Phase 7:** Confirm clean git hygiene and absence of secrets or dangling files.
7. **Phase 8:** Generate `docs/reports/FINAL_REPOSITORY_READINESS_REPORT.md`.

