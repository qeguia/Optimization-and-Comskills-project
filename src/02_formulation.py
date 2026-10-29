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

# ─────────────────────────────────────────────
# 2.3  PYOMO MILP MODEL
# ─────────────────────────────────────────────

def build_model(min_rev_frac=0.95, min_margin=0.20, max_avg_inc=0.10, E_matrix=None):
    """
    Build the linearized discrete-price MILP.

    Parameters
    ----------
    min_rev_frac : float  — γ_R, minimum revenue as fraction of baseline
    min_margin   : float  — γ_Π, minimum profit margin
    max_avg_inc  : float  — θ_max, max average price increase
    E_matrix     : ndarray — 7×7 elasticity matrix (default: from CSV)

    Returns
    -------
    Pyomo ConcreteModel.
    """
    if E_matrix is None:
        E_matrix = E

    m = pyo.ConcreteModel("PepsiCo_Pricing_MILP")

    # ── Sets ──
    m.I_P = pyo.Set(initialize=PEPSICO)
    m.I_C = pyo.Set(initialize=COMPETITOR)
    m.I   = pyo.Set(initialize=ALL_PRODUCTS)
    m.K   = pyo.Set(initialize=range(NUM_CANDIDATES))

    # ── Parameters ──
    m.p0 = pyo.Param(m.I, initialize={
        i: products.loc[i, "current_price_mxn"] for i in ALL_PRODUCTS})
    m.V0 = pyo.Param(m.I, initialize={
        i: products.loc[i, "baseline_volume_units_wk"] for i in ALL_PRODUCTS})
    m.s  = pyo.Param(m.I, initialize={
        i: products.loc[i, "size_g"] for i in ALL_PRODUCTS})
    m.c  = pyo.Param(m.I, initialize={
        i: products.loc[i, "unit_cost_mxn"] for i in ALL_PRODUCTS})

    # Candidate fractional price changes
    delta_cand = {}
    price_cand = {}
    for i in PEPSICO:
        for k_idx, pk in enumerate(candidates[i]):
            delta_cand[i, k_idx] = (pk - products.loc[i, "current_price_mxn"]) / products.loc[i, "current_price_mxn"]
            price_cand[i, k_idx] = pk
    m.delta_cand = pyo.Param(m.I_P, m.K, initialize=delta_cand)
    m.price_cand = pyo.Param(m.I_P, m.K, initialize=price_cand)

    # Competitor follow parameters
    follow_hero = {i: int(products.loc[i, "follows_product"]) for i in COMPETITOR}
    follow_rho  = {i: products.loc[i, "follow_ratio"] for i in COMPETITOR}

    # ── Decision Variables ──
    m.y = pyo.Var(m.I_P, m.K, within=pyo.Binary)

    # ── Derived Expressions (all linear in y) ──

    # δ for PepsiCo products
    def delta_pep_rule(m, i):
        return sum(m.delta_cand[i, k] * m.y[i, k] for k in m.K)
    m.delta_pep = pyo.Expression(m.I_P, rule=delta_pep_rule)

    # δ for all products (competitors follow hero)
    def delta_all_rule(m, i):
        if i in PEPSICO:
            return m.delta_pep[i]
        else:
            return follow_rho[i] * m.delta_pep[follow_hero[i]]
    m.delta = pyo.Expression(m.I, rule=delta_all_rule)

    # New price
    def pnew_rule(m, i):
        return m.p0[i] * (1 + m.delta[i])
    m.pnew = pyo.Expression(m.I, rule=pnew_rule)

    # Predicted volume (linearized: V0 * (1 + Σ E_ij δ_j))
    def volume_rule(m, i):
        return m.V0[i] * (1 + sum(E_matrix[i-1, j-1] * m.delta[j]
                                  for j in ALL_PRODUCTS))
    m.vol = pyo.Expression(m.I, rule=volume_rule)

    # Revenue per product (linearized)
    def rev_rule(m, i):
        return (m.V0[i] * m.p0[i] / m.s[i]) * \
               (1 + m.delta[i] + sum(E_matrix[i-1, j-1] * m.delta[j]
                                     for j in ALL_PRODUCTS))
    m.rev = pyo.Expression(m.I, rule=rev_rule)

    # Profit per product
    def prof_rule(m, i):
        return m.rev[i] - m.c[i] * m.vol[i] / m.s[i]
    m.prof = pyo.Expression(m.I, rule=prof_rule)

    # Totals for PepsiCo
    m.total_rev  = pyo.Expression(expr=sum(m.rev[i]  for i in PEPSICO))
    m.total_prof = pyo.Expression(expr=sum(m.prof[i] for i in PEPSICO))

    # ── Objective: maximize PepsiCo revenue ──
    m.obj = pyo.Objective(expr=m.total_rev, sense=pyo.maximize)

    # ── Constraints ──

    # (C1) Exactly one price per PepsiCo product
    def selection_rule(m, i):
        return sum(m.y[i, k] for k in m.K) == 1
    m.c_selection = pyo.Constraint(m.I_P, rule=selection_rule)

    # (C2) Revenue floor: total revenue ≥ fraction of baseline
    m.c_rev_floor = pyo.Constraint(
        expr=m.total_rev >= min_rev_frac * BASELINE_REV)

    # (C3) Margin floor: total profit ≥ fraction of total revenue
    m.c_margin = pyo.Constraint(
        expr=m.total_prof >= min_margin * m.total_rev)

    # (C4) Price ladders (larger pack ≤ per-gram price of smaller pack)
    def ladder_rule(m, pair_idx):
        big, small = LADDER_PAIRS[pair_idx]
        return m.pnew[big] / m.s[big] <= m.pnew[small] / m.s[small]
    m.c_ladder = pyo.Constraint(range(len(LADDER_PAIRS)), rule=ladder_rule)

    # (C5) Non-negative volume
    def vol_pos_rule(m, i):
        return m.vol[i] >= 0
    m.c_vol_pos = pyo.Constraint(m.I, rule=vol_pos_rule)

    # (C6) Maximum average price increase
    total_p0 = sum(products.loc[i, "current_price_mxn"] for i in PEPSICO)
    m.c_avg_increase = pyo.Constraint(
        expr=sum(m.p0[i] * m.delta_pep[i] for i in PEPSICO)
             <= max_avg_inc * total_p0)

    return m

# ─────────────────────────────────────────────
# 2.4  SOLVER
# ─────────────────────────────────────────────

def solve_model(m, solver_name="glpk"):
    """Solve the Pyomo model with the given solver."""
    solver = pyo.SolverFactory(solver_name)
    result = solver.solve(m, tee=False)
    return result


def extract_results(m):
    """Extract a results DataFrame from a solved model."""
    rows = []
    for i in PEPSICO:
        chosen_k = [k for k in range(NUM_CANDIDATES)
                     if pyo.value(m.y[i, k]) > 0.5][0]
        new_p = candidates[i][chosen_k]
        old_p = products.loc[i, "current_price_mxn"]
        delta_pct = (new_p - old_p) / old_p * 100
        vol_new = pyo.value(m.vol[i])
        vol_old = products.loc[i, "baseline_volume_units_wk"]
        rev_new = pyo.value(m.rev[i])
        rows.append({
            "Product": products.loc[i, "name"],
            "Old Price": old_p,
            "New Price": new_p,
            "Price Δ%": f"{delta_pct:+.1f}%",
            "Old Volume": vol_old,
            "New Volume": f"{vol_new:.0f}",
            "Volume Δ%": f"{(vol_new - vol_old)/vol_old*100:+.1f}%",
            "Revenue (MXN)": f"{rev_new:.1f}",
        })
    return pd.DataFrame(rows)