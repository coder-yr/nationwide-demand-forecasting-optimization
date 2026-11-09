## Pull Request Description

### Summary of Changes
<!-- Provide a concise description of what this PR introduces, fixes, or refactors. -->

### Problem Statement / Context
<!-- Why is this change necessary? Reference linked issues if applicable (e.g., Fixes #12). -->

---

## Type of Change
- [ ] 🐛 Bug fix (non-breaking change fixing an issue)
- [ ] ✨ New feature (non-breaking change adding functionality)
- [ ] ⚡ Performance improvement / Memory optimization
- [ ] 📈 Forecasting model / Optimization algorithm enhancement
- [ ] 📝 Documentation update / Diagram improvement
- [ ] 🧪 Test suite addition or invariant verification

---

## Invariant & Quality Checklist
- [ ] **Zero Data Leakage:** Confirmed no features utilize data past the $t-28$ forecast horizon.
- [ ] **Lost-Sales Dynamics:** Physical shelf inventory formulation remains strictly non-negative ($I \ge 0$).
- [ ] **Hierarchical Coherence:** Parent-child reconciliation aggregates without mathematical discrepancy.
- [ ] **Test Coverage:** All unit and integration tests pass (`python -m pytest tests/ -p no:dash`).
- [ ] **No Committed Secrets or Large Datasets:** No raw `.csv` or generated `.parquet` files are tracked.

