# Contributing to Nationwide Demand Forecasting & Inventory Optimization

Thank you for your interest in contributing to the **Nationwide Demand Forecasting and Inventory Optimization Platform**! We welcome improvements, bug reports, and optimizations from the data science, operations research, and software engineering communities.

---

## 1. Code of Conduct & Standards

To ensure a high standard of engineering quality:
- **Zero Look-Ahead Bias:** Any new feature engineering pipelines must strictly enforce the $t - 28$ day information boundary to prevent data leakage.
- **Mathematical Integrity:** Reconcilers and metrics must strictly uphold parent-child coherence and official M5 calculation formulas.
- **Test-Driven Delivery:** All PRs must include pytest coverage for new functionality without introducing look-ahead leakage or breaks to existing tests.
- **Type Safety & Style:** Code should conform to PEP 8, include docstrings, and leverage explicit type annotations.

---

## 2. Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/coder-yr/nationwide-demand-forecasting-optimization.git
   cd nationwide-demand-forecasting-optimization
   ```

2. **Create and activate a virtual environment:**
   - **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the test suite:**
   ```bash
   python -m pytest tests/ -v -p no:dash
   ```

---

## 3. Pull Request Guidelines

1. Create a feature branch off `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Implement your changes, ensuring code is formatted and tested.
3. Verify that all 18 automated tests pass locally.
4. Commit your changes with clear, semantic commit messages (e.g. `feat: add wls reconciliation weightings`, `fix: handle edge case in lead time calendar join`).
5. Open a Pull Request referencing any related issues using our standard [Pull Request Template](.github/pull_request_template.md).

---

## 4. Reporting Issues

Please use our GitHub Issue templates:
- **[Bug Report](.github/ISSUE_TEMPLATE/bug_report.md):** For reproducibility issues, data pipeline crashes, or metric discrepancies.
- **[Feature Request](.github/ISSUE_TEMPLATE/feature_request.md):** For proposing new forecasting models, optimization solvers, or UI widgets.

