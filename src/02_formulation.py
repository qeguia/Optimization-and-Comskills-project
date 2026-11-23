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

A1. Small portfolio: 5 PepsiCo products + 2 competitor products (article
    handles 40+ product promotion groups across multiple retailers).

A2. Single representative store, single planning period; no multi-period
    dynamics (article considers semi-annual/annual planning horizons).

A3. Linearization of the volume model.
    The paper uses:
        V_i = V0_i · exp( Σ_j E_ij · [ln(p_j) - ln(p0_j)] )

    For small fractional price changes  δ_j = (p_j - p0_j) / p0_j,
    a first-order Taylor expansion gives:
        V_i ≈ V0_i · (1 + Σ_j E_ij · δ_j)

    This is linear in the δ (and binary y) variables. Valid when |δ| ≤ ~15%.

A4. Exact discrete price-volume product via McCormick envelopes.
    Substituting p_i = Σ_k pk_{i,k} y_{i,k} and linearized volume V_i(y)
    into Rev_i = p_i · V_i produces binary products w_{i,k,j,l} = y_{i,k} · y_{j,l}
    (for i < j), which are linearized exactly via McCormick inequalities:
        w ≤ y_{i,k},   w ≤ y_{j,l},   w ≥ y_{i,k} + y_{j,l} - 1
    This removes second-order Taylor revenue approximation error.

A5. Competitor prices follow PepsiCo "hero" products proportionally:
        δ_competitor = ρ · δ_hero     (ρ = 0.50 in our base data)
    Same mechanism as in paper (Eq. 20).

A6. Psychological price points (PPP) omitted for simplicity (the paper
    includes step-function PPP effects).

A7. Discrete candidate prices -> binary selection variables.
    Since Mexican shelf prices are integer Pesos, we define a finite set of
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
  p0_i     : current baseline shelf price [MXN/pack]
  V0_i     : baseline weekly sales volume [packs/week]
  s_i      : pack size [grams]
  c_i      : unit cost [MXN/pack]
  E_ij     : cross-price elasticity (entry i,j of matrix E)
  pk_{i,k} : k-th candidate price for product i [MXN/pack]
  ρ_i      : competitor follow ratio (for i ∈ I_C)
  h(i)     : PepsiCo hero product that competitor i follows
  γ_R      : minimum revenue as fraction of baseline (0.95)
  γ_Π      : minimum profit margin as fraction of revenue (0.20)
  θ_max    : maximum weighted average price increase (0.10)

--- Decision Variables ---
  y_{i,k} ∈ {0,1}         1 if candidate price k is selected for product i ∈ I_P
  w_{i,k,j,l} ∈ [0,1]     McCormick auxiliary variable for y_{i,k} · y_{j,l} (i < j)

--- Derived Expressions ---
  p_i      = Σ_k pk_{i,k} · y_{i,k}                           for i ∈ I_P
  δ_i      = Σ_k [(pk_{i,k} - p0_i) / p0_i] · y_{i,k}         for i ∈ I_P
  δ_i      = ρ_i · δ_{h(i)}                                   for i ∈ I_C
  V_i      = V0_i · (1 + Σ_{j ∈ I} E_ij · δ_j)                predicted volume [packs/wk]
  Rev_i    = p_i · V_i  (exact via McCormick w_{i,k,j,l})     weekly revenue [MXN/wk]
  Profit_i = Rev_i - c_i · V_i                                weekly gross profit [MXN/wk]

--- Objective ---
  max  Σ_{i ∈ I_P} Profit_i     (maximize total PepsiCo weekly gross profit)

--- Constraints ---
  (C1) Selection      : Σ_k y_{i,k} = 1 ∀ i ∈ I_P -> Exactly one price per SKU
  (C2) Revenue floor  : Σ_{i ∈ I_P} Rev_i ≥ γ_R · Rev_baseline -> Revenue ≥ 95% of baseline
  (C3) Margin floor   : Σ_{i ∈ I_P} Profit_i ≥ γ_Π · Σ_{i ∈ I_P} Rev_i -> Gross margin ≥ 20%
  (C4) Price ladder   : p_large / s_large ≤ p_small / s_small -> Lower price/gram on 170g vs 62g
  (C5) Non-neg volume : V_i ≥ 0 ∀ i ∈ I_P -> Non-negative PepsiCo predicted volumes
  (C6) Avg price cap  : Σ_{i ∈ I_P}(p0_i · δ_i) / Σ_{i ∈ I_P}(p0_i) ≤ θ_max -> Avg price change ≤ +10%
  (C7) McCormick env. : w ≤ y_{i,k}, w ≤ y_{j,l}, w ≥ y_{i,k} + y_{j,l} - 1 ∀ (i,k,j,l), i < j

==============================================================================
"""

import os
import pandas as pd
import numpy as np
import pyomo.environ as pyo

# ─────────────────────────────────────────────
# 2.1  LOAD DATA
# ─────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

products       = pd.read_csv(os.path.join(DATA_DIR, "products.csv")).set_index("id")
elasticity_raw = pd.read_csv(os.path.join(DATA_DIR, "elasticity_matrix.csv"))
candidates_raw = pd.read_csv(os.path.join(DATA_DIR, "candidate_prices.csv"))

E = elasticity_raw.drop(columns=["product_id"]).values

candidates = {}
for pid, group in candidates_raw.groupby("product_id"):
    candidates[pid] = sorted(group["price_option"].tolist())

PEPSICO      = products[products["is_pepsico"] == 1].index.tolist()
COMPETITOR   = products[products["is_pepsico"] == 0].index.tolist()
ALL_PRODUCTS = list(products.index)
NUM_CANDIDATES = len(candidates[PEPSICO[0]])

# Price-ladder pairs: (large_product, small_product) within same brand
LADDER_PAIRS = [(1, 2), (3, 4)]  # Lays 170g/62g, Doritos 170g/62g

# ─────────────────────────────────────────────
# 2.2  COMPUTE BASELINE
# ─────────────────────────────────────────────

def compute_baseline():
    rev, prof = 0.0, 0.0
    for i in PEPSICO:
        p = products.loc[i]
        v0 = float(p["baseline_volume_units_wk"])
        r = p["current_price_mxn"] * v0
        rev += r
        prof += (p["current_price_mxn"] - p["unit_cost_mxn"]) * v0
    return rev, prof

BASELINE_REV, BASELINE_PROF = compute_baseline()

# ─────────────────────────────────────────────
# 2.3  PYOMO MILP MODEL
# ─────────────────────────────────────────────

def build_model(
    min_rev_frac=0.95,
    min_margin=0.20,
    max_avg_inc=0.10,
    E_matrix=None,
    cand_dict=None,
    follow_rho_override=None,
):
    """
    Build the profit-maximizing discrete-price MILP with linearized volume V_i(y)
    and exact McCormick linearization of p_i * V_i = sum_{k,j,l} coeff * y_{i,k} * y_{j,l}.
    """
    if E_matrix is None:
        E_matrix = E
    if cand_dict is None:
        cand_dict = candidates

    num_k = len(cand_dict[PEPSICO[0]])
    m = pyo.ConcreteModel("PepsiCo_Profit_McCormick_MILP")

    # ── Sets ──
    m.I_P = pyo.Set(initialize=PEPSICO)
    m.I_C = pyo.Set(initialize=COMPETITOR)
    m.I   = pyo.Set(initialize=ALL_PRODUCTS)
    m.K   = pyo.Set(initialize=range(num_k))

    # ── Parameters ──
    m.p0 = pyo.Param(m.I, initialize={i: float(products.loc[i, "current_price_mxn"]) for i in ALL_PRODUCTS})
    m.V0 = pyo.Param(m.I, initialize={i: float(products.loc[i, "baseline_volume_units_wk"]) for i in ALL_PRODUCTS})
    m.s  = pyo.Param(m.I, initialize={i: float(products.loc[i, "size_g"]) for i in ALL_PRODUCTS})
    m.c  = pyo.Param(m.I, initialize={i: float(products.loc[i, "unit_cost_mxn"]) for i in ALL_PRODUCTS}, mutable=True)

    delta_cand, price_cand = {}, {}
    for i in PEPSICO:
        p0_i = float(products.loc[i, "current_price_mxn"])
        for k_idx, pk in enumerate(cand_dict[i]):
            delta_cand[i, k_idx] = (pk - p0_i) / p0_i
            price_cand[i, k_idx] = float(pk)

    m.delta_cand = pyo.Param(m.I_P, m.K, initialize=delta_cand)
    m.price_cand = pyo.Param(m.I_P, m.K, initialize=price_cand)

    follow_hero = {c_id: int(products.loc[c_id, "follows_product"]) for c_id in COMPETITOR}
    follow_rho  = {
        c_id: (float(products.loc[c_id, "follow_ratio"]) if follow_rho_override is None else float(follow_rho_override))
        for c_id in COMPETITOR
    }

    # ── Decision Variables ──
    m.y = pyo.Var(m.I_P, m.K, within=pyo.Binary)

    # McCormick bilinear variables w_{i,k,j,l} = y_{i,k} * y_{j,l} for i < j
    pairs = [(i, k, j, l) for i in PEPSICO for k in range(num_k)
             for j in PEPSICO if j > i for l in range(num_k)]
    m.W_IDX = pyo.Set(initialize=pairs, dimen=4)
    m.w = pyo.Var(m.W_IDX, bounds=(0.0, 1.0))

    m.mc1 = pyo.Constraint(m.W_IDX, rule=lambda m, i, k, j, l: m.w[i, k, j, l] <= m.y[i, k])
    m.mc2 = pyo.Constraint(m.W_IDX, rule=lambda m, i, k, j, l: m.w[i, k, j, l] <= m.y[j, l])
    m.mc3 = pyo.Constraint(m.W_IDX, rule=lambda m, i, k, j, l: m.w[i, k, j, l] >= m.y[i, k] + m.y[j, l] - 1.0)

    def y_prod(i, k, j, l):
        if i == j:
            return m.y[i, k] if k == l else 0.0
        elif i < j:
            return m.w[i, k, j, l]
        else:
            return m.w[j, l, i, k]

    # ── Derived Expressions ──
    m.delta_pep = pyo.Expression(m.I_P, rule=lambda m, i: sum(m.delta_cand[i, k] * m.y[i, k] for k in m.K))
    m.delta     = pyo.Expression(m.I, rule=lambda m, i: m.delta_pep[i] if i in PEPSICO else follow_rho[i] * m.delta_pep[follow_hero[i]])
    m.pnew      = pyo.Expression(m.I_P, rule=lambda m, i: sum(m.price_cand[i, k] * m.y[i, k] for k in m.K))
    m.vol       = pyo.Expression(m.I, rule=lambda m, i: m.V0[i] * (1.0 + sum(E_matrix[i - 1, j - 1] * m.delta[j] for j in ALL_PRODUCTS)))

    # Effective elasticity matrix on PepsiCo decisions (incorporating competitor follow-through)
    E_eff = np.zeros((len(PEPSICO), len(PEPSICO)))
    for idx_i, i in enumerate(PEPSICO):
        for idx_j, j in enumerate(PEPSICO):
            val = E_matrix[i - 1, j - 1]
            for c_id in COMPETITOR:
                if follow_hero[c_id] == j:
                    val += E_matrix[i - 1, c_id - 1] * follow_rho[c_id]
            E_eff[idx_i, idx_j] = val

    # Exact Revenue Rev_i = p_i * V_i via McCormick products
    def rev_exact_prod_rule(m, i):
        idx_i = PEPSICO.index(i)
        expr = 0.0
        for k in m.K:
            pk = m.price_cand[i, k]
            expr += pk * m.V0[i] * m.y[i, k]
            for idx_j, j in enumerate(PEPSICO):
                for l in m.K:
                    expr += pk * m.V0[i] * E_eff[idx_i, idx_j] * m.delta_cand[j, l] * y_prod(i, k, j, l)
        return expr

    m.rev  = pyo.Expression(m.I_P, rule=rev_exact_prod_rule)
    m.prof = pyo.Expression(m.I_P, rule=lambda m, i: m.rev[i] - m.c[i] * m.vol[i])

    m.total_rev  = pyo.Expression(expr=sum(m.rev[i] for i in PEPSICO))
    m.total_prof = pyo.Expression(expr=sum(m.prof[i] for i in PEPSICO))

    # ── Objective: Maximize Total Weekly PepsiCo Gross Profit (MXN/wk) ──
    m.obj = pyo.Objective(expr=m.total_prof, sense=pyo.maximize)

    # ── Constraints ──
    m.c_selection = pyo.Constraint(m.I_P, rule=lambda m, i: sum(m.y[i, k] for k in m.K) == 1)
    m.c_rev_floor = pyo.Constraint(expr=m.total_rev >= min_rev_frac * BASELINE_REV)
    m.c_margin    = pyo.Constraint(expr=m.total_prof >= min_margin * m.total_rev)
    m.c_ladder    = pyo.Constraint(
        range(len(LADDER_PAIRS)),
        rule=lambda m, idx: m.pnew[LADDER_PAIRS[idx][0]] / m.s[LADDER_PAIRS[idx][0]]
                            <= m.pnew[LADDER_PAIRS[idx][1]] / m.s[LADDER_PAIRS[idx][1]]
    )
    m.c_vol_pos   = pyo.Constraint(m.I_P, rule=lambda m, i: m.vol[i] >= 0.0)

    total_p0 = sum(float(products.loc[i, "current_price_mxn"]) for i in PEPSICO)
    m.c_avg_increase = pyo.Constraint(
        expr=sum(m.p0[i] * m.delta_pep[i] for i in PEPSICO) <= max_avg_inc * total_p0
    )

    return m

# ─────────────────────────────────────────────
# 2.4  SOLVER & EXTRACTION HELPERS
# ─────────────────────────────────────────────

def solve_model(m, solver_name="glpk"):
    """Solve the Pyomo model with the given solver."""
    solver = pyo.SolverFactory(solver_name)
    return solver.solve(m, tee=False)

def get_chosen_prices(m):
    """Return {product_id: chosen_price} for all PepsiCo products."""
    num_k = len(m.K)
    return {
        i: int(round(pyo.value(m.price_cand[i, [k for k in range(num_k) if pyo.value(m.y[i, k]) > 0.5][0]])))
        for i in PEPSICO
    }

def extract_results(m):
    """Extract a results DataFrame from a solved model."""
    num_k = len(m.K)
    rows = []
    for i in PEPSICO:
        chosen_k = [k for k in range(num_k) if pyo.value(m.y[i, k]) > 0.5][0]
        new_p = int(round(pyo.value(m.price_cand[i, chosen_k])))
        old_p = int(products.loc[i, "current_price_mxn"])
        delta_pct = (new_p - old_p) / old_p * 100
        vol_new = pyo.value(m.vol[i])
        vol_old = products.loc[i, "baseline_volume_units_wk"]
        rev_new = pyo.value(m.rev[i])
        prof_new = pyo.value(m.prof[i])
        rows.append({
            "Product": products.loc[i, "name"],
            "Old Price": old_p,
            "New Price": new_p,
            "Price Δ%": f"{delta_pct:+.1f}%",
            "Old Vol": vol_old,
            "New Vol": f"{vol_new:.1f}",
            "Vol Δ%": f"{(vol_new - vol_old)/vol_old*100:+.1f}%",
            "Rev (MXN)": f"{rev_new:,.2f}",
            "Profit (MXN)": f"{prof_new:,.2f}",
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# 2.5  TEST: Build and solve, print formulation summary
# ─────────────────────────────────────────────

if __name__ == "__main__":

    model = build_model()

    print(f"\n  Decision variables : {len(list(model.y))} binary (y_{{i,k}}), {len(list(model.w))} McCormick (w)")
    print(f"  Products optimized : {len(PEPSICO)}")
    print(f"  Candidates/product : {NUM_CANDIDATES}")
    print(f"  Constraints        : selection({len(PEPSICO)}) + "
          f"rev_floor(1) + margin(1) + ladders({len(LADDER_PAIRS)}) + "
          f"vol_pos({len(PEPSICO)}) + avg_inc(1) + McCormick({3 * len(model.W_IDX)})")
    print(f"  Baseline profit    : {BASELINE_PROF:,.2f} MXN/week")
    print(f"  Baseline revenue   : {BASELINE_REV:,.2f} MXN/week")
    print(f"  Baseline margin    : {BASELINE_PROF/BASELINE_REV:.2%}")

    print(f"\n  Solving with GLPK...")
    result = solve_model(model)
    print(f"  Solver status: {result.solver.status}")

    opt_rev = pyo.value(model.total_rev)
    opt_prof = pyo.value(model.total_prof)
    print(f"\n  Optimal profit  : {opt_prof:,.2f} MXN/week ({(opt_prof/BASELINE_PROF - 1)*100:+.2f}%)")
    print(f"  Optimal revenue : {opt_rev:,.2f} MXN/week ({(opt_rev/BASELINE_REV - 1)*100:+.2f}%)")
    print(f"  Optimal margin  : {opt_prof/opt_rev:.2%}")

    print("\n  Results table:")
    df = extract_results(model)
    print(df.to_string(index=False))