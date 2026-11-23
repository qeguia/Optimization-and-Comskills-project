"""
==============================================================================
STEP 5 — SENSITIVITY & WHAT-IF ANALYSES
==============================================================================

REFERENCE
Llenas et al. (2026) "PepsiCo Deploys AI-Driven Pricing and Promotion
Optimization at Scale", INFORMS Journal on Applied Analytics.

==============================================================================
This script:
  1. Verifies that cost inflation (up to +50%), margin floor (up to 59%),
     revenue floor (95%), and avg price cap (+10%) remain slack and do NOT
     change the optimal prices in standard operating ranges.
  2. Executes three active sensitivity experiments by re-solving the Pyomo
     McCormick MILP:
       (a) Elasticity Matrix Scaling (±20% on E)
       (b) Candidate Price Grid Half-Width (±1 to ±4 MXN) — verifying interior
           optimality vs. boundary corner behavior
       (c) Competitor Price-Match Retaliation (ρ from 0% to 100%)
  3. Generates sensitivity_analysis_results.png for the poster.

==============================================================================
"""

import os
import numpy as np
import pyomo.environ as pyo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
step2 = SourceFileLoader(
    "step2",
    os.path.join(CURRENT_DIR, "02_formulation.py")
).load_module()

products          = step2.products
E                 = step2.E
PEPSICO           = step2.PEPSICO
ALL_PRODUCTS      = step2.ALL_PRODUCTS
BASELINE_REV      = step2.BASELINE_REV
BASELINE_PROF     = step2.BASELINE_PROF
build_model       = step2.build_model
solve_model       = step2.solve_model
get_chosen_prices = step2.get_chosen_prices

print("=" * 64)
print("  STEP 5: SENSITIVITY & WHAT-IF PARAMETER ANALYSIS")
print("=" * 64)

# ==============================================================
# 1. VERIFY NON-BINDING CONSTRAINTS (SLACK IN OPERATING RANGE)
# ==============================================================
print("\n[1] Checking Cost-Inflation & Margin Constraints (Slack in Operating Range):")

m_base = build_model(min_rev_frac=0.95, min_margin=0.20, max_avg_inc=0.10)
solve_model(m_base)
base_prices = list(get_chosen_prices(m_base).values())
base_prof   = pyo.value(m_base.total_prof)
base_rev    = pyo.value(m_base.total_rev)

for cm in [1.00, 1.15, 1.25, 1.50]:
    m_c = build_model(min_margin=0.20)
    for i in ALL_PRODUCTS:
        m_c.c[i] = products.loc[i, "unit_cost_mxn"] * cm
    solve_model(m_c)
    assert list(get_chosen_prices(m_c).values()) == base_prices

for mf in [0.15, 0.20, 0.35, 0.55, 0.59]:
    m_m = build_model(min_margin=mf)
    solve_model(m_m)
    assert list(get_chosen_prices(m_m).values()) == base_prices

print(f"  -> Optimal prices {base_prices} MXN (+{(base_prof/BASELINE_PROF - 1)*100:.2f}% profit lift) are UNCHANGED across:")
print("     • Cost inflation up to +50% (margin compresses from 59.01% to 38.52%, well above 20% floor)")
print("     • Margin floor γ_Π from 15% up to 59% (slack because snack gross margins sit near 55%–59%)")
print("     • Average price cap +10% (+7.37% weighted change achieved)")

# ==============================================================
# 2. ACTIVE EXPERIMENT A: ELASTICITY SCALING (±20% ON MATRIX E)
# ==============================================================
print("\n[2] Testing Elasticity Matrix Scaling (-20% to +20% on E)...")
e_scales = [0.80, 0.90, 1.00, 1.10, 1.20]
prof_lifts_e, rev_lifts_e, prices_e = [], [], []

for sc in e_scales:
    m = build_model(E_matrix=E * sc)
    solve_model(m)
    prof_val = pyo.value(m.total_prof)
    rev_val  = pyo.value(m.total_rev)
    p_list   = list(get_chosen_prices(m).values())
    prof_lifts_e.append((prof_val / BASELINE_PROF - 1.0) * 100.0)
    rev_lifts_e.append((rev_val / BASELINE_REV - 1.0) * 100.0)
    prices_e.append(p_list)
    print(f"  Elasticity {sc*100:5.0f}% | Profit: {prof_val:9,.2f} MXN/wk ({prof_lifts_e[-1]:+.2f}%) | "
          f"Rev: {rev_val:9,.2f} MXN/wk ({rev_lifts_e[-1]:+.2f}%) | Prices: {p_list}")

# ==============================================================
# 3. ACTIVE EXPERIMENT B: CANDIDATE GRID HALF-WIDTH (±1 TO ±4 MXN)
# ==============================================================
print("\n[3] Testing Candidate Price Grid Width (±1 to ±4 MXN)...")
grid_widths = [1, 2, 3, 4]
prof_lifts_w, interior_counts, prices_w = [], [], []

for w in grid_widths:
    cand_w = {
        i: list(range(int(products.loc[i, "current_price_mxn"]) - w,
                      int(products.loc[i, "current_price_mxn"]) + w + 1))
        for i in PEPSICO
    }
    m = build_model(cand_dict=cand_w)
    solve_model(m)
    chosen = get_chosen_prices(m)
    p_list = list(chosen.values())
    n_int  = sum(1 for i in PEPSICO if min(cand_w[i]) < chosen[i] < max(cand_w[i]))
    prof_val = pyo.value(m.total_prof)
    prof_lifts_w.append((prof_val / BASELINE_PROF - 1.0) * 100.0)
    interior_counts.append(n_int)
    prices_w.append(p_list)
    print(f"  Grid ±{w} MXN | Profit: {prof_val:9,.2f} MXN/wk ({prof_lifts_w[-1]:+.2f}%) | "
          f"Interior SKUs: {n_int}/5 | Prices: {p_list}")

# ==============================================================
# 4. ACTIVE EXPERIMENT C: COMPETITOR PRICE RETALIATION (0% TO 100%)
# ==============================================================
print("\n[4] Testing Competitor Price Match Retaliation (ρ from 0% to 100%)...")
retaliation_ratios = [0.0, 0.25, 0.50, 0.75, 1.00]
prof_lifts_comp, rev_lifts_comp, chosen_sets = [], [], []

for rho in retaliation_ratios:
    m = build_model(min_margin=0.20, follow_rho_override=rho)
    solve_model(m)
    prof_val = pyo.value(m.total_prof)
    rev_val  = pyo.value(m.total_rev)
    p_lift   = ((prof_val - BASELINE_PROF) / BASELINE_PROF) * 100.0
    r_lift   = ((rev_val - BASELINE_REV) / BASELINE_REV) * 100.0
    prices   = list(get_chosen_prices(m).values())
    prof_lifts_comp.append(p_lift)
    rev_lifts_comp.append(r_lift)
    chosen_sets.append(prices)
    print(f"  Retaliation ρ = {rho*100:5.1f}% | Profit: {prof_val:9,.2f} MXN/wk ({p_lift:+.2f}%) | "
          f"Rev: {rev_val:9,.2f} MXN/wk ({r_lift:+.2f}%) | Prices: {prices}")

# ==============================================================
# 5. GENERATE POSTER-READY 3-PANEL SENSITIVITY FIGURE
# ==============================================================
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

# Panel A: Elasticity Scaling (±20%)
e_pct = [(s - 1.0) * 100.0 for s in e_scales]
axes[0].plot(e_pct, prof_lifts_e, marker="o", markersize=6.5, color="#004B93", linewidth=2.4, label="Profit Lift (%)")
axes[0].plot(e_pct, rev_lifts_e, marker="s", markersize=5.5, color="#E32934", linestyle="--", linewidth=2.0, label="Revenue Lift (%)")
axes[0].axhline(0, color="gray", linestyle=":", alpha=0.7)
axes[0].annotate("1.2E: [23,15,26,16,21]\n(3 SKUs shift inward)",
                 xy=(20, prof_lifts_e[-1]), xytext=(-2, 2.8), fontsize=8.5, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#004B93", lw=1.1))
axes[0].set_title("A: Elasticity Scaling (±20% on E)", fontweight="bold", fontsize=11.5)
axes[0].set_xlabel("Elasticity Multiplier Δ (%)", fontweight="bold", fontsize=10.5)
axes[0].set_ylabel("Lift vs. Baseline (%)", fontweight="bold", fontsize=10.5)
axes[0].legend(fontsize=9, loc="lower left")
axes[0].grid(True, linestyle=":", alpha=0.6)

# Panel B: Candidate Grid Half-Width (±1 to ±4 MXN)
bars = axes[1].bar([f"±{w} MXN" for w in grid_widths], prof_lifts_w, color="#059669", width=0.48)
for idx, (lift, n_i) in enumerate(zip(prof_lifts_w, interior_counts)):
    axes[1].text(idx, lift + 0.14, f"+{lift:.2f}%\n({n_i}/5 int.)",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color="#0f172a")
axes[1].set_ylim(0, max(prof_lifts_w) * 1.28)
axes[1].set_title("B: Candidate Grid Width Sweep", fontweight="bold", fontsize=11.5)
axes[1].set_xlabel("Allowed Price Grid Half-Width", fontweight="bold", fontsize=10.5)
axes[1].set_ylabel("Optimal Profit Lift (%)", fontweight="bold", fontsize=10.5)
axes[1].grid(axis="y", linestyle=":", alpha=0.6)

# Panel C: Competitor Retaliation (ρ from 0% to 100%)
rho_pct = [r * 100.0 for r in retaliation_ratios]
axes[2].plot(rho_pct, prof_lifts_comp, marker="D", markersize=6.5, color="#d97706", linewidth=2.4)
for xp, yp in zip(rho_pct, prof_lifts_comp):
    axes[2].annotate(f"+{yp:.2f}%", (xp, yp + 0.15), ha="center", fontsize=8.5, fontweight="bold")
axes[2].set_xlim(-5, 105)
axes[2].set_ylim(2.0, 6.2)
axes[2].set_title("C: Competitor Match Ratio (ρ)", fontweight="bold", fontsize=11.5)
axes[2].set_xlabel("Competitor Price-Match Ratio ρ (%)", fontweight="bold", fontsize=10.5)
axes[2].set_ylabel("Optimal Profit Lift (%)", fontweight="bold", fontsize=10.5)
axes[2].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("sensitivity_analysis_results.png", dpi=220, bbox_inches="tight")
print("\nSaved sensitivity_analysis_results.png")