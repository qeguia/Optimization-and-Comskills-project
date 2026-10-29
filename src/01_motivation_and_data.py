"""
==============================================================================
01 MOTIVATION, BUSINESS CASE & DATA PREPARATION
==============================================================================

REFERENCE
Llenas et al. (2026) "PepsiCo Deploys AI-Driven Pricing and Promotion
Optimization at Scale", INFORMS Journal on Applied Analytics.
DOI: 10.1287/inte.2025.0302

TEAM
Matisse Aaftink-De Luca · Fernando Cordero · Qin Eguía · Tamara Shopova

==============================================================================
BUSINESS CONTEXT
==============================================================================

PepsiCo Mexico sells snack products (Lay's, Doritos, Sabritas) through 
convenience-store chains like OXXO. Every pricing cycle, the Revenue
Management team must set new shelf prices for the entire portfolio.

THE CHALLENGE:
  • Cross-price effects: Raising the price of Lay's pushes consumers to
    Doritos (own-brand cannibalization) or to Barcel, a competitor.
  • Price ladders: A 170g bag cannot cost less per gram than a 62g bag
    of the same brand — consumers expect bulk discounts.
  • Financial floors: Total revenue and profit margins must stay above
    minimum thresholds.
  • Discrete prices: In Mexico, shelf prices are integer Pesos — the
    decision space is inherently discrete, not continuous.

WHAT THE PAPER DOES (PricingAI):
  The paper uses a Bayesian hierarchical model to estimate elasticities,
  then feeds them into a Differential Evolution (DE) metaheuristic.
  DE is a population-based optimizer that gives NO optimality guarantee.

OUR CONTRIBUTION:
  Because prices are discrete, we linearize the nonlinear demand model
  and formulate the problem as a Mixed-Integer Linear Program (MILP).
  This gives us:
    - A provably optimal solution (bounded gap via branch-and-bound)
    - Compatibility with standard solvers (CBC, GLPK, HiGHS)
    - The ability to do sensitivity analysis with solver infrastructure

==============================================================================
DATA DESCRIPTION
==============================================================================

We use three CSV files with synthetic data (tried to be realistic) inspired by the PepsiCo Mexico snack market:

  1. products.csv
     - 7 products: 5 PepsiCo (Lays 170g/62g, Doritos 170g/62g, Sabritas
       170g) and 2 competitors (Barcel 170g/62g)
     - Columns: id, name, brand, size_g, current_price_mxn, unit_cost_mxn,
       baseline_volume_units_wk, is_pepsico, follows_product, follow_ratio
     - Competitor products "follow" a PepsiCo hero product: if PepsiCo
       raises price by 10%, the competitor raises by 5% (follow_ratio=0.5)

  2. elasticity_matrix.csv
     - 7×7 cross-price elasticity matrix E
     - Diagonal entries (own-price): between -1.4 and -1.9 (negative,
       consistent with the paper's validation range of -2.5 to 0)
     - Off-diagonal entries (cross-price): between 0 and 0.25 (positive
       for substitutes, meaning a price increase in product j pushes
       demand toward product i)

  3. candidate_prices.csv
     - For each PepsiCo product, 5 candidate discrete prices (±2 MXN
       from the current price)
     - These are the only prices the optimizer can choose from — reflecting
       the real-world constraint that prices must be integer Pesos

==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─────────────────────────────────────────────
# 1.1  LOAD DATA FROM CSV
# ─────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
elasticity_raw = pd.read_csv(os.path.join(DATA_DIR, "elasticity_matrix.csv"))
candidates_raw = pd.read_csv(os.path.join(DATA_DIR, "candidate_prices.csv"))

# Parse elasticity into a numpy matrix
E = elasticity_raw.drop(columns=["product_id"]).values

# Parse candidate prices into a dict: {product_id: [prices]}
candidates = {}
for pid, group in candidates_raw.groupby("product_id"):
    candidates[pid] = sorted(group["price_option"].tolist())

# Set product id as index for easy lookup
products = products.set_index("id")

print("=" * 60)
print("  PRODUCTS")
print("=" * 60)
print(products.to_string())

print("\n" + "=" * 60)
print("  CROSS-PRICE ELASTICITY MATRIX")
print("=" * 60)
print(f"\n{E}\n")
print("Diagonal (own-price elasticities):", np.diag(E).tolist())
print("All own-price elasticities are negative ✓" 
      if all(np.diag(E) < 0) else "WARNING: positive own-price elasticity!")

print("\n" + "=" * 60)
print("  CANDIDATE PRICES (MXN)")
print("=" * 60)
for pid, prices in candidates.items():
    name = products.loc[pid, "name"]
    current = products.loc[pid, "current_price_mxn"]
    print(f"  {name:25s}  current={current}  options={prices}")

# ─────────────────────────────────────────────
# 1.2  COMPUTE BASELINE KPIs
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  BASELINE KPIs (before optimization)")
print("=" * 60)

pepsico_ids = products[products["is_pepsico"] == 1].index.tolist()

total_rev, total_prof, total_vol = 0.0, 0.0, 0.0
for i in products.index:
    p = products.loc[i]
    units = p["baseline_volume_units_wk"] / p["size_g"]
    rev_i = units * p["current_price_mxn"]
    prof_i = rev_i - p["unit_cost_mxn"] * units
    if p["is_pepsico"]:
        total_rev += rev_i
        total_prof += prof_i
        total_vol += p["baseline_volume_units_wk"]

print(f"  PepsiCo weekly revenue : {total_rev:,.1f} MXN")
print(f"  PepsiCo weekly profit  : {total_prof:,.1f} MXN")
print(f"  PepsiCo margin         : {total_prof/total_rev:.1%}")
print(f"  PepsiCo total volume   : {total_vol:,.0f} units")