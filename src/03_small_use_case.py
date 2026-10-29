"""
==============================================================================
STEP 3 — SMALL USE CASE: Solve, Validate & Visualize
==============================================================================

REFERENCE
Llenas et al. (2026) "PepsiCo Deploys AI-Driven Pricing and Promotion
Optimization at Scale", INFORMS Journal on Applied Analytics.

This script:
  1. Solves the base-case MILP (from 02_formulation_and_model.py)
  2. Checks linearization accuracy vs. the exact exponential model
  3. Runs three sanity checks to validate our simplifications
  4. Compares optimal solution vs. a naive "+5% uniform" strategy
  5. Produces three publication-quality charts

==============================================================================
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import model builder
from importlib.machinery import SourceFileLoader
step2 = SourceFileLoader("step2", "02_formulation.py").load_module()

products     = step2.products
E            = step2.E
candidates   = step2.candidates
PEPSICO      = step2.PEPSICO
COMPETITOR   = step2.COMPETITOR
ALL_PRODUCTS = step2.ALL_PRODUCTS
BASELINE_REV = step2.BASELINE_REV
BASELINE_PROF = step2.BASELINE_PROF
build_model  = step2.build_model
solve_model  = step2.solve_model
extract_results = step2.extract_results

# ==============================================================
# 3.1  SOLVE THE BASE CASE
# ==============================================================

model = build_model()
result = solve_model(model)

opt_rev = pyo.value(model.total_rev)
opt_prof = pyo.value(model.total_prof)

print(f"\n  Solver status     : {result.solver.status}")
print(f"  Optimal revenue   : {opt_rev:,.1f} MXN/week")
print(f"  Optimal profit    : {opt_prof:,.1f} MXN/week")
print(f"  Optimal margin    : {opt_prof/opt_rev:.1%}")
print(f"  Revenue uplift    : {(opt_rev/BASELINE_REV - 1)*100:+.2f}%")

df_results = extract_results(model)
print(f"\n{df_results.to_string(index=False)}")

# ==============================================================
# 3.2  LINEARIZATION ACCURACY CHECK
# ==============================================================

print("\n  Comparing our linear approximation vs. the exact exponential model (from the paper) at the optimal prices:\n")

delta_vals = {j: pyo.value(model.delta[j]) for j in ALL_PRODUCTS}

max_error = 0.0
for i in ALL_PRODUCTS:
    # Our linear approximation
    v_linear = products.loc[i, "baseline_volume_units_wk"] * (
        1 + sum(E[i-1, j-1] * delta_vals[j] for j in ALL_PRODUCTS))

    # Exact exponential (paper's Eq. 24)
    exponent = sum(E[i-1, j-1] * np.log(1 + delta_vals[j])
                   for j in ALL_PRODUCTS)
    v_exact = products.loc[i, "baseline_volume_units_wk"] * np.exp(exponent)

    error = abs(v_linear - v_exact) / v_exact * 100
    max_error = max(max_error, error)
    print(f"  {products.loc[i, 'name']:25s}  "
          f"Linear={v_linear:7.1f}  Exact={v_exact:7.1f}  "
          f"Error={error:.2f}%")

print(f"\n  Maximum linearization error: {max_error:.2f}%")
if max_error < 5.0:
    print("  :) All errors < 5% — linearization is valid for this price range")
else:
    print("  :( Some errors exceed 5% — consider tighter price bounds")