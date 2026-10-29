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


# ==============================================================
# 3.3  SANITY CHECKS
# ==============================================================

print("\n" + "=" * 60)
print("  3.3  SANITY CHECKS")
print("=" * 60)

# ── Check 1: Zero elasticities -> max prices ──
print("\n  [Check 1] Zero elasticities -> optimizer should pick max prices")
E_zero = np.zeros((7, 7))
m1 = build_model(E_matrix=E_zero, min_rev_frac=0.0, min_margin=0.0)
solve_model(m1)
all_max = True
for i in PEPSICO:
    chosen_k = [k for k in range(5) if pyo.value(m1.y[i, k]) > 0.5][0]
    pk = candidates[i][chosen_k]
    max_p = max(candidates[i])
    ok = pk == max_p
    all_max = all_max and ok
    print(f"    {products.loc[i, 'name']:25s}  chosen={pk}  max={max_p}  "
          f"{'PASS' if ok else 'FAIL'}")
print(f"  -> {'PASS' if all_max else 'FAIL'}")

# ── Check 2: Very high own-elasticity -> lower prices ──
print("\n  [Check 2] High own-elasticity (-3.0) -> should keep or lower prices")
E_high = E.copy()
for i in range(5):
    E_high[i, i] = -3.0
m2 = build_model(E_matrix=E_high, min_rev_frac=0.0, min_margin=0.0)
solve_model(m2)
all_lower = True
for i in PEPSICO:
    chosen_k = [k for k in range(5) if pyo.value(m2.y[i, k]) > 0.5][0]
    pk = candidates[i][chosen_k]
    p0 = products.loc[i, "current_price_mxn"]
    direction = "UP" if pk > p0 else ("UP" if pk < p0 else "=")
    ok = pk <= p0
    all_lower = all_lower and ok
    print(f"    {products.loc[i, 'name']:25s}  {p0}→{pk} {direction}  "
          f"{'PASS' if ok else 'FAIL'}")
print(f"  -> {'PASS' if all_lower else 'PARTIAL (some raised)'}")

# ── Check 3: Price ladders satisfied ──
print("\n  [Check 3] Price ladders respected?")
ladder_ok = True
for big, small in [(1, 2), (3, 4)]:
    kb = [k for k in range(5) if pyo.value(model.y[big, k]) > 0.5][0]
    ks = [k for k in range(5) if pyo.value(model.y[small, k]) > 0.5][0]
    ppg_big = candidates[big][kb] / products.loc[big, "size_g"]
    ppg_sm  = candidates[small][ks] / products.loc[small, "size_g"]
    ok = ppg_big <= ppg_sm + 1e-6
    ladder_ok = ladder_ok and ok
    brand = products.loc[big, "brand"]
    print(f"    {brand}: 170g={ppg_big:.4f} MXN/g  ≤  "
          f"62g={ppg_sm:.4f} MXN/g  {'PASS' if ok else 'FAIL'}")
print(f"  -> {'PASS' if ladder_ok else 'FAIL'}")

# ── Check 4: Revenue floor binding? ──
print(f"\n  [Check 4] Revenue floor: {opt_rev:.1f} ≥ "
      f"{0.95*BASELINE_REV:.1f}  "
      f" -> {'PASS' if opt_rev >= 0.95*BASELINE_REV - 0.01 else 'FAIL'}")

# ── Check 5: Margin floor binding? ──
print(f"  [Check 5] Margin floor: {opt_prof/opt_rev:.1%} ≥ 20%  "
      f" -> {'PASS' if opt_prof/opt_rev >= 0.20 - 0.001 else 'FAILa'}")

# ==============================================================
# 3.4  COMPARISON: OPTIMAL vs. NAIVE "+5% UNIFORM"
# ==============================================================

# Compute naive revenue: raise all prices by 5%
naive_rev = 0.0
for i in PEPSICO:
    delta_naive = {j: 0.05 if j in PEPSICO else 0.025 for j in ALL_PRODUCTS}
    vol_i = products.loc[i, "baseline_volume_units_wk"] * (
        1 + sum(E[i-1, j-1] * delta_naive[j] for j in ALL_PRODUCTS))
    rev_i = (products.loc[i, "baseline_volume_units_wk"] *
             products.loc[i, "current_price_mxn"] /
             products.loc[i, "size_g"]) * (
        1 + delta_naive[i] + sum(E[i-1, j-1] * delta_naive[j]
                                  for j in ALL_PRODUCTS))
    naive_rev += rev_i

print(f"\n  Baseline revenue   : {BASELINE_REV:,.1f} MXN/week")
print(f"  Naive +5% revenue  : {naive_rev:,.1f} MXN/week  "
      f"({(naive_rev/BASELINE_REV - 1)*100:+.2f}%)")
print(f"  MILP optimal rev   : {opt_rev:,.1f} MXN/week  "
      f"({(opt_rev/BASELINE_REV - 1)*100:+.2f}%)")
print(f"\n  Optimizer advantage : {opt_rev - naive_rev:+.1f} MXN/week "
      f"over naive strategy")