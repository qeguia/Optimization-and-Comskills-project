"""
==============================================================================
STEP 4 - REALISTIC USE CASE: Pyomo MILP Solver Execution & Graphs
==============================================================================

REFERENCE
Llenas et al. (2026) "PepsiCo Deploys AI-Driven Pricing and Promotion
Optimization at Scale", INFORMS Journal on Applied Analytics.

==============================================================================
This script:
  1. Loads the Mexican retail channel synthetic data
  2. Formulates the Mixed-Integer Linear Program (MILP) in Pyomo
  3. Solves it using an open-source solver (HiGHS / GLPK / CBC)
  4. Outputs comparative visual charts

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

# ==============================================================
# 4.1 IMPORT FORMULATION & DATA FROM STEP 2
# ==============================================================
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

print("=" * 60)
print("  STEP 4: REALISTIC USE CASE & PYOMO SOLVER EXECUTION")
print("=" * 60)

# ==============================================================
# 4.2 SOLVE BASELINE REALISTIC MILP MODEL
# ==============================================================
model = build_model(min_rev_frac=0.95, min_margin=0.20, max_avg_inc=0.10)
result = solve_model(model, solver_name="glpk")

opt_rev = pyo.value(model.total_rev)
opt_prof = pyo.value(model.total_prof)
opt_margin = opt_prof / opt_rev
rev_uplift_pct = ((opt_rev - BASELINE_REV) / BASELINE_REV) * 100.0
prof_uplift_pct = ((opt_prof - BASELINE_PROF) / BASELINE_PROF) * 100.0

print(f"\n  Solver Status        : {result.solver.status}")
print(f"  Baseline Revenue     : ${BASELINE_REV:,.2f} MXN/wk")
print(f"  Optimal Revenue      : ${opt_rev:,.2f} MXN/wk ({rev_uplift_pct:+.2f}%)")
print(f"  Baseline Profit      : ${BASELINE_PROF:,.2f} MXN/wk (Margin: {BASELINE_PROF/BASELINE_REV:.1%})")
print(f"  Optimal Profit       : ${opt_prof:,.2f} MXN/wk (Margin: {opt_margin:.1%}, {prof_uplift_pct:+.2f}%)")

# Detailed Product Results
df_results = extract_results(model)
print("\nProduct-Level Optimization Decisions:")
print(df_results.to_string(index=False))

# ==============================================================
# 4.3 NAIVE BENCHMARK COMPARISON (+5% Price Increase Strategy)
# ==============================================================
naive_rev = 0.0
naive_prof = 0.0
delta_naive = {j: 0.05 if j in PEPSICO else 0.025 for j in ALL_PRODUCTS}

for i in PEPSICO:
    vol_i = products.loc[i, "baseline_volume_units_wk"] * (
        1 + sum(E[i-1, j-1] * delta_naive[j] for j in ALL_PRODUCTS)
    )
    rev_i = (products.loc[i, "baseline_volume_units_wk"] * products.loc[i, "current_price_mxn"] / products.loc[i, "size_g"]) * (
        1 + delta_naive[i] + sum(E[i-1, j-1] * delta_naive[j] for j in ALL_PRODUCTS)
    )
    prof_i = rev_i - products.loc[i, "unit_cost_mxn"] * (vol_i / products.loc[i, "size_g"])
    naive_rev += rev_i
    naive_prof += prof_i

print(f"\nNaive Strategy (+5% Uniform Raise):")
print(f"  Revenue : ${naive_rev:,.2f} MXN/wk ({(naive_rev/BASELINE_REV - 1)*100:+.2f}%)")
print(f"  Profit  : ${naive_prof:,.2f} MXN/wk ({(naive_prof/BASELINE_PROF - 1)*100:+.2f}%)")
print(f"  Optimizer Advantage over Naive: +${opt_rev - naive_rev:,.2f} MXN/wk")

# ==============================================================
# 4.4 VISUALIZATIONS: Strategy Comparison & Product Breakdown
# ==============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Chart 1: Financial Performance Across Strategies
strategies = ["Baseline\n(Status Quo)", "Naive Rule\n(+5% Uniform)", "MILP Optimal\n(Pyomo)"]
rev_values = [BASELINE_REV, naive_rev, opt_rev]
prof_values = [BASELINE_PROF, naive_prof, opt_prof]

x = np.arange(len(strategies))
width = 0.35

rects1 = axes[0].bar(x - width/2, rev_values, width, label='Revenue', color='#2563eb')
rects2 = axes[0].bar(x + width/2, prof_values, width, label='Gross Profit', color='#059669')

axes[0].set_ylabel("MXN / Week", fontweight="bold")
axes[0].set_title("Strategy Financial Comparison", fontweight="bold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(strategies)
axes[0].legend()
axes[0].grid(axis='y', linestyle='--', alpha=0.3)

for rect in rects1:
    h = rect.get_height()
    pct = ((h - BASELINE_REV) / BASELINE_REV) * 100
    axes[0].text(rect.get_x() + rect.get_width()/2., h + 1000, f"${h:,.0f}\n({pct:+.1f}%)" if pct!=0 else f"${h:,.0f}", ha='center', va='bottom', fontsize=8)

# Chart 2: Product Volume vs Revenue Impact
names = [products.loc[i, "name"].replace(" ", "\n") for i in PEPSICO]
vol_changes = []
rev_changes = []

for i in PEPSICO:
    v_old = products.loc[i, "baseline_volume_units_wk"]
    v_new = pyo.value(model.vol[i])
    vol_changes.append((v_new - v_old) / v_old * 100)
    
    r_old = (v_old * products.loc[i, "current_price_mxn"] / products.loc[i, "size_g"])
    r_new = pyo.value(model.rev[i])
    rev_changes.append(r_new - r_old)

x_prod = np.arange(len(PEPSICO))
axes[1].bar(x_prod - width/2, vol_changes, width, label='Volume % Δ', color='#d97706')
axes[1].set_ylabel("Volume Change (%)", color='#d97706', fontweight="bold")
axes[1].set_xticks(x_prod)
axes[1].set_xticklabels(names, fontsize=8)

ax_twin = axes[1].twinx()
ax_twin.plot(x_prod, rev_changes, color='#dc2626', marker='o', linewidth=2, label='Revenue Δ (MXN)')
ax_twin.set_ylabel("Revenue Δ (MXN/wk)", color='#dc2626', fontweight="bold")

axes[1].set_title("Product-Level Volume & Revenue Response", fontweight="bold")
axes[1].grid(axis='y', linestyle='--', alpha=0.3)

plt.suptitle("Step 4: Realistic Portfolio Optimization Results (PepsiCo Mexico)", fontsize=13, fontweight="bold")
plt.savefig("realistic_use_case_results.png", dpi=200)
print("\nSaved realistic_use_case_results.png")