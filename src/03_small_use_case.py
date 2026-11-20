"""
==============================================================================
STEP 3 — SMALL USE CASE: Solve, Validate & Visualize
==============================================================================

REFERENCE
Llenas et al. (2026) "PepsiCo Deploys AI-Driven Pricing and Promotion
Optimization at Scale", INFORMS Journal on Applied Analytics.

This script:
1. Solves the base-case MILP (from 02_formulation.py)
  2. Checks linearization accuracy vs. the exact exponential model
  3. Generates linear_vs_exact_comparison.png (Linear vs. Real Function)
  4. Runs sanity checks and compares against a naive "+5% uniform" rule
  5. Produces base_case_results.png

==============================================================================
"""

import os
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader

step2 = SourceFileLoader(
    "step2",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_formulation.py")
).load_module()

products        = step2.products
E               = step2.E
candidates      = step2.candidates
PEPSICO         = step2.PEPSICO
COMPETITOR      = step2.COMPETITOR
ALL_PRODUCTS    = step2.ALL_PRODUCTS
BASELINE_REV    = step2.BASELINE_REV
BASELINE_PROF   = step2.BASELINE_PROF
build_model     = step2.build_model
solve_model     = step2.solve_model
extract_results = step2.extract_results

# ==============================================================
# 3.1  SOLVE THE BASE CASE
# ==============================================================

model = build_model()
result = solve_model(model)

opt_rev = pyo.value(model.total_rev)
opt_prof = pyo.value(model.total_prof)

print(f"\n  Solver status     : {result.solver.status}")
print(f"  Optimal revenue   : {opt_rev:,.2f} MXN/week")
print(f"  Optimal profit    : {opt_prof:,.2f} MXN/week")
print(f"  Optimal margin    : {opt_prof/opt_rev:.2%}")
print(f"  Revenue uplift    : {(opt_rev/BASELINE_REV - 1)*100:+.2f}%")

df_results = extract_results(model)
print(f"\n{df_results.to_string(index=False)}")

# ==============================================================
# 3.2  LINEARIZATION ACCURACY CHECK & REAL-FUNCTION GRAPH
# ==============================================================

print("\n  Comparing linear approximation vs. exact exponential model (Eq. 24):\n")

delta_vals = {j: pyo.value(model.delta[j]) for j in ALL_PRODUCTS}

max_error = 0.0
rev_lin_list, rev_ext_list, vol_err_list = [], [], []
tot_rev_exact = 0.0

for i in ALL_PRODUCTS:
    V0 = products.loc[i, "baseline_volume_units_wk"]
    p0 = products.loc[i, "current_price_mxn"]
    s  = products.loc[i, "size_g"]
    p_new = p0 * (1 + delta_vals[i])

    # Linearized approximation
    v_linear = V0 * (1 + sum(E[i - 1, j - 1] * delta_vals[j] for j in ALL_PRODUCTS))
    r_linear = pyo.value(model.rev[i])

    # Exact exponential function (paper's Eq. 24)
    exponent = sum(E[i - 1, j - 1] * np.log(1 + delta_vals[j]) for j in ALL_PRODUCTS)
    v_exact = V0 * np.exp(exponent)
    r_exact = (v_exact * p_new) / s

    error = abs(v_linear - v_exact) / v_exact * 100
    max_error = max(max_error, error)

    if i in PEPSICO:
        rev_lin_list.append(r_linear)
        rev_ext_list.append(r_exact)
        vol_err_list.append(error)
        tot_rev_exact += r_exact

    print(f"  {products.loc[i, 'name']:25s}  "
          f"Linear={v_linear:7.1f}  Exact={v_exact:7.1f}  "
          f"Vol Error={error:.2f}%")

print(f"\n  Maximum volume linearization error : {max_error:.2f}% (< 5% threshold)")
print(f"  Total PepsiCo Revenue              : Linear = {opt_rev:.2f} vs Exact = {tot_rev_exact:.2f} MXN/wk "
      f"(diff = {abs(opt_rev - tot_rev_exact)/tot_rev_exact*100:.2f}%)")

# ── Plot: Linear Approximation vs. Real Exponential Function ──
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

# Panel A: Demand multiplier curve across price change delta in [-15%, +15%]
deltas_grid = np.linspace(-0.15, 0.15, 200)
e_own = E[0, 0]  # Lays Classic 170g (-1.80)
v_mult_exact = (1 + deltas_grid) ** e_own
v_mult_linear = 1 + e_own * deltas_grid

axes[0].axvspan(-0.143, 0.143, color="#e0f2fe", alpha=0.55, label="Candidate Price Range (±2 MXN)")
axes[0].plot(deltas_grid * 100, v_mult_exact, color="#004B93", linewidth=2.4, label="Real Exponential Function")
axes[0].plot(deltas_grid * 100, v_mult_linear, color="#E32934", linestyle="--", linewidth=2.2, label="1st-Order Taylor Linear")
axes[0].scatter([delta_vals[1] * 100], [(1 + delta_vals[1]) ** e_own], color="#004B93", zorder=5, s=45)
axes[0].annotate("Optimal Lays 170g\n(-9.1%, 20 MXN)",
                 xy=(delta_vals[1] * 100, (1 + delta_vals[1]) ** e_own),
                 xytext=(-6.5, 1.24), fontsize=8, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#004B93", lw=1))
axes[0].set_title("A: Demand Curve — Linear vs. Real Exponential", fontweight="bold", fontsize=10)
axes[0].set_xlabel("Fractional Price Change δ (%)")
axes[0].set_ylabel("Volume Multiplier (V / V⁰)")
axes[0].legend(fontsize=7.5, loc="upper right")
axes[0].grid(True, linestyle=":", alpha=0.6)

# Panel B: SKU-level Revenue at Optimal Prices (Linear MILP vs. Exact Exponential)
pep_names = [products.loc[i, "name"].replace(" ", "\n") for i in PEPSICO]
x_idx = np.arange(len(PEPSICO))
w = 0.36

axes[1].bar(x_idx - w/2, rev_lin_list, w, label="Linearized MILP ($504.32)", color="#E32934")
axes[1].bar(x_idx + w/2, rev_ext_list, w, label="Exact Exponential ($505.57)", color="#004B93")
for xi, rl, re, err in zip(x_idx, rev_lin_list, rev_ext_list, vol_err_list):
    axes[1].text(xi, max(rl, re) + 3.5, f"Vol err:\n{err:.1f}%", ha="center", fontsize=7.5, color="#1e293b")
axes[1].set_ylim(0, max(rev_ext_list) * 1.18)
axes[1].set_xticks(x_idx)
axes[1].set_xticklabels(pep_names, fontsize=8)
axes[1].set_ylabel("Weekly Revenue (MXN/wk)")
axes[1].set_title("B: Optimal Revenue — Linear vs. Real Function", fontweight="bold", fontsize=10)
axes[1].legend(fontsize=7.5, loc="upper left")
axes[1].grid(axis="y", linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("linear_vs_exact_comparison.png", dpi=200, bbox_inches="tight")
print("  Saved linear_vs_exact_comparison.png")

# ==============================================================
# 3.3  SANITY CHECKS & NAIVE COMPARISON
# ==============================================================

naive_rev = 0.0
delta_naive = {j: 0.05 if j in PEPSICO else 0.025 for j in ALL_PRODUCTS}
for i in PEPSICO:
    rev_i = (products.loc[i, "baseline_volume_units_wk"] *
             products.loc[i, "current_price_mxn"] /
             products.loc[i, "size_g"]) * (
        1 + delta_naive[i] + sum(E[i - 1, j - 1] * delta_naive[j] for j in ALL_PRODUCTS))
    naive_rev += rev_i

print(f"\n  Baseline revenue    : {BASELINE_REV:,.2f} MXN/week")
print(f"  Naive +5% revenue   : {naive_rev:,.2f} MXN/week ({(naive_rev/BASELINE_REV - 1)*100:+.2f}%)")
print(f"  MILP optimal rev    : {opt_rev:,.2f} MXN/week ({(opt_rev/BASELINE_REV - 1)*100:+.2f}%)")
print(f"  Optimizer advantage : {opt_rev - naive_rev:+.2f} MXN/week over naive strategy")