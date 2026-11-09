"""Pytest session configuration."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Pre-load ortools before pytest plugins or pandas/pyarrow load
try:
    import ortools
    from ortools.linear_solver import pywraplp
except Exception:
    pass

