"""
==============================================================================
STEP 2 — FORMULATION, ASSUMPTIONS, SIMPLIFICATIONS & MILP MODEL
==============================================================================

REFERENCE
Llenas et al. (2026) "PepsiCo Deploys AI-Driven Pricing and Promotion
Optimization at Scale", INFORMS Journal on Applied Analytics.

==============================================================================
ASSUMPTIONS & SIMPLIFICATIONS
==============================================================================

A1. Small portfolio: 5 PepsiCo products + 2 competitor products (article handles 40+ products across multiple retailers).

A2. Single retailer, single period; no multi-period dynamics. (article considers semi-annual/annual planning horizons).

A3. Linearization of the volume model.
    The paper uses:
        V_i = V0_i · exp( Σ_j E_ij · [ln(p_j) - ln(p0_j)] )

    For small fractional price changes  δ_j = (p_j - p0_j) / p0_j,
    a first-order Taylor expansion gives:
        V_i ≈ V0_i · (1 + Σ_j E_ij · δ_j)

    This is linear in the δ variables. Valid when |δ| ≤ ~10-15%.

A4. Revenue simplification.
    Rev_i = (V_i / s_i) · p_i
    Substituting linearized volume:
        Rev_i ≈ (V0_i · p0_i / s_i) · (1 + δ_i + Σ_j E_ij · δ_j)
    We drop the second-order cross-term δ_i · (Σ E_ij δ_j), which is negligible for small price changes. Result: linear in δ.

A5. Competitor prices follow PepsiCo "hero" products proportionally:
        δ_competitor = ρ · δ_hero     (ρ = 0.5 in our data)
    Same mechanism as in paper (Eq. 20).

A6. Psychological price points (PPP) omitted for simplicity. (the paper includes step-function PPP effects)

A7. Discrete candidate prices -> binary selection variables.
    Since Mexican prices are integer Pesos, we define a finite set of
    candidate prices for each product and use binary y_{i,k} variables.

==============================================================================
FORMULATION
==============================================================================

--- Sets ---
  I       = {1,...,7}     all products
  I_P     = {1,...,5}     PepsiCo products (we optimize these)
  I_C     = {6, 7}        competitor products (prices follow PepsiCo)
  K_i     = {0,...,4}     candidate price indices for product i ∈ I_P

--- Parameters ---
  p0_i     : current baseline shelf price [MXN]
  V0_i     : baseline weekly sales volume [units]
  s_i      : pack size [grams]
  c_i      : unit cost [MXN/unit]
  E_ij     : cross-price elasticity (entry i,j of matrix E)
  pk_{i,k} : k-th candidate price for product i [MXN]
  ρ_i      : competitor follow ratio (for i ∈ I_C)
  h(i)     : PepsiCo hero product that competitor i follows
  γ_R      : minimum revenue as fraction of baseline
  γ_Π      : minimum profit margin (fraction of revenue)
  θ_max    : maximum average price increase (fraction)

--- Decision Variables ---
  y_{i,k} ∈ {0,1}   1 if candidate price k is selected for product i

--- Derived Variables (all linear in y) ---
  δ_i = Σ_k [(pk_{i,k} - p0_i) / p0_i] · y_{i,k}        for i ∈ I_P
  δ_i = ρ_i · δ_{h(i)}                                  for i ∈ I_C
  V_i = V0_i · (1 + Σ_j E_ij · δ_j)                     predicted volume
  Rev_i = (V0_i · p0_i / s_i) · (1 + δ_i + Σ_j E_ij · δ_j)
  Profit_i = Rev_i - c_i · V_i / s_i

--- Objective ---
  max  Σ_{i ∈ I_P}  Rev_i       (maximize total PepsiCo revenue)

--- Constraints ---
  (C1) Selection: Σ_k y_{i,k} = 1 ∀ i ∈ I_P -> Exactly one price per product

  (C2) Revenue floor: Σ_i Rev_i ≥ γ_R · Rev_baseline -> Total revenue stays above 95% of current

  (C3) Margin floor: Σ_i Profit_i ≥ γ_Π · Σ_i Rev_i -> Aggregate profit margin ≥ 20%

  (C4) Price ladder: p_new_large / s_large ≤ p_new_small / s_small -> Larger bags have lower price-per-gram (within same brand)

  (C5) Non-neg volume: V_i ≥ 0 ∀ i -> Predicted volumes must be non-negative

  (C6) Avg increase: Σ_i(p0_i · δ_i) / Σ_i(p0_i) ≤ θ_max -> Average weighted price increase ≤ 10%

==============================================================================
"""

import pandas as pd
import numpy as np
import pyomo.environ as pyo
import os

# ─────────────────────────────────────────────
# 2.1  LOAD DATA
# ─────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

products = pd.read_csv(os.path.join(DATA_DIR, "products.csv")).set_index("id")
elasticity_raw = pd.read_csv(os.path.join(DATA_DIR, "elasticity_matrix.csv"))
candidates_raw = pd.read_csv(os.path.join(DATA_DIR, "candidate_prices.csv"))

E = elasticity_raw.drop(columns=["product_id"]).values

candidates = {}
for pid, group in candidates_raw.groupby("product_id"):
    candidates[pid] = sorted(group["price_option"].tolist())

PEPSICO = products[products["is_pepsico"] == 1].index.tolist()
COMPETITOR = products[products["is_pepsico"] == 0].index.tolist()
ALL_PRODUCTS = list(products.index)
NUM_CANDIDATES = 5

# Price-ladder pairs: (large_product, small_product) within same brand
LADDER_PAIRS = [(1, 2), (3, 4)]  # Lays 170g/62g, Doritos 170g/62g

# ─────────────────────────────────────────────
# 2.2  COMPUTE BASELINE
# ─────────────────────────────────────────────

def compute_baseline():
    rev, prof = 0.0, 0.0
    for i in PEPSICO:
        p = products.loc[i]
        units = p["baseline_volume_units_wk"] / p["size_g"]
        r = units * p["current_price_mxn"]
        rev += r
        prof += r - p["unit_cost_mxn"] * units
    return rev, prof

BASELINE_REV, BASELINE_PROF = compute_baseline()