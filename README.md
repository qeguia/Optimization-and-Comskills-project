# Optimization-and-Comskills-project
Optimization I and Communication Skills

---

## Overview

This project applies Mixed-Integer Linear Programming (MILP) to solve base pricing optimization for PepsiCo snack products across Mexican convenience stores (OXXO retail channel). 

The project adapts the non-linear demand model presented in Llenas et al. (2026) (*"PepsiCo Deploys AI-Driven Pricing and Promotion Optimization at Scale"*, INFORMS Journal on Applied Analytics) by linearizing the volume response function using a first-order Taylor series expansion.

The project integrates:
- Data Engineering & Setup: Processing synthetic convenience store data for 7 SKUs (5 PepsiCo + 2 Competitors), discrete Peso price points, and a 7×7 cross-price elasticity matrix.
- MILP Pyomo Formulation: Discrete binary decision choices ($y_{i,k} \in \{0,1\}$), volume linearization, price-ladder rules, margin protection bounds, and revenue floors.
- Solver Execution & Performance Analysis: Solving the base case to provable global optimality.
- Sensitivity & What-If Parameter Analysis: Testing cost inflation squeezes, competitor price wars, and margin floor rigidity.

---

# Installation

### 1. Clone the repository

```bash
git clone https://github.com/qeguia/Optimization-and-Comskills-project.git
cd Optimization-and-Comskills-project
```

### 2. Create the environment using the .yml file

```bash
conda env create -f environment.yml
```

### 3. Activate the environment

```bash
conda activate optimization-project
```

---

## Usage

This project uses a modular `src/`-based structure. To ensure imports work correctly, all commands must be executed from inside the `src/` directory:

```bash
cd src
```

### Running the End-to-End Pipeline
Execute the pipeline step-by-step:

#### 1. Motivation & Data Inspection (Step 1)
Loads product catalogs, candidate price options, and computes baseline financial KPIs:
```bash
python 01_motivation_and_data.py
```

#### 2. Model Formulation & Baseline Build (Step 2)
Defines the Pyomo MILP model, mathematical decision variables, and constraint structures:
```bash
python 02_formulation.py
```

#### 3. Model Validation & Linearization Checks (Step 3)
Evaluates Taylor approximation accuracy against the exact exponential model and executes 5 sanity checks:
```bash
python 03_small_use_case.py
```

#### 4. Realistic Use Case & Pyomo Solver Execution (Step 4)
Solves the convenience store portfolio model and compares MILP optimal strategy against status quo and naive rules:
```bash
python 04_realistic_use_case.py
```

#### 5. Managerial What-If Sensitivity Analysis (Step 5)
Runs parameter experiments (cost inflation squeeze, competitor price war, margin floor variations) and generates sensitivity plots:
```bash
python 05_sensitivity_analysis.py
```

#### 6. Managerial Synthesis & Insights Report (Step 6)
Aggregates financial KPIs, generates an executive strategy comparison table, and prints managerial takeaways:
```bash
python 06_managerial_analysis.py
```

---

## Mathematical Model Formulation

The optimization model determines discrete integer shelf prices $p_i$ for each PepsiCo product $i \in I_P$ to maximize total portfolio revenue while respecting operational constraints.

### 1. Volume Linearization (First-Order Taylor Expansion)
The original paper utilizes an exponential volume response equation:

$$
V_i = V_i^0 \cdot \exp\left( \sum_{j \in I} E_{i,j} \ln\frac{p_j}{p_j^0} \right)
$$

For small price changes ($\pm 10\%$ to $15\%$), we approximate this via a first-order Taylor series expansion:

$$
V_i \approx V_i^0 \left( 1 + \sum_{j \in I} E_{i,j} \delta_j \right)
$$

where the fractional price change $\delta_j$ of Stock Keeping Unit(SKU) $j$ is defined as:

$$
\delta_j = \frac{p_j - p_j^0}{p_j^0}
$$

and $E_{i,j}$ represents the cross-price elasticity between product $i$ and product $j$.

### 2. Decision Variables & Price Choice
Since Mexican shelf prices are restricted to integer Pesos, we define binary variables $y_{i,k} \in \{0, 1\}$ for each product $i$ and candidate price index $k$:

$$
\delta_i = \sum_k \left( \frac{p_{i,k} - p_i^0}{p_i^0} \right) y_{i,k}
$$

### 3. Optimization Model Structure

$$
\max_{y} \quad \sum_{i \in I_P} \text{Rev}_i \approx \sum_{i \in I_P} \left( \frac{V_i^0 p_i^0}{s_i} \right) \left( 1 + \delta_i + \sum_{j \in I} E_{i,j} \delta_j \right)
$$

$$\text{such that}$$

| Constraint | Mathematical Expression | Operational Interpretation |
| :--- | :--- | :--- |
| Single Price Choice | $\sum_k y_{i,k} = 1 \quad \forall i \in I_P$ | Exactly one shelf price option selected per SKU. |
| Price Ladder | $\frac{p_{\text{large}}}{s_{\text{large}}} \le \frac{p_{\text{small}}}{s_{\text{small}}}$ | 170g family packs must maintain lower price-per-gram than 62g single-serve packs. |
| Gross Margin Floor | &sum;<sub>i</sub> Profit<sub>i</sub> &ge; &gamma;<sub>&Pi;</sub> &middot; &sum;<sub>i</sub> Rev<sub>i</sub> | Total portfolio gross profit margin must stay $\ge 20\%$. |
| Revenue Floor | &sum;<sub>i</sub> Rev<sub>i</sub> &ge; &gamma;<sub>R</sub> &middot; Rev<sub>baseline</sub> | Total weekly revenue must stay $\ge 95\%$ of status quo baseline. |
| Avg Price Increase | $\frac{\sum_i p_i^0 \delta_i}{\sum_i p_i^0} \le \theta_{\max}$ | Weighted portfolio price increases capped at $+10\%$. |

---

## Project Structure

```
Optimization-and-Comskills-project/
├── data/
│   ├── candidate_prices.csv        # 5 discrete integer Peso options per SKU
│   ├── elasticity_matrix.csv       # 7x7 cross-price elasticity matrix
│   └── products.csv                # SKU catalog, base prices, costs, baseline volumes
│
├── src/
│   ├── 01_motivation_and_data.py   # Step 1: Data setup & baseline KPIs
│   ├── 02_formulation.py            # Step 2: Pyomo MILP builder & solver wrapper
│   ├── 03_small_use_case.py        # Step 3: Validation, Taylor error & sanity checks
│   ├── 04_realistic_use_case.py    # Step 4: Pyomo solver & benchmark comparison
│   ├── 05_sensitivity_analysis.py # Step 5: Cost, retaliation & margin What-Ifs
│   └── 06_managerial_analysis.py   # Step 6: Executive synthesis & trade-off matrix
│
├── pepsico_pricing_poster.html     # Single-page A4 Landscape Academic Poster
├── environment.yml                 # Conda dependencies specification
└── README.md                       # Project documentation
```

---

## Technologies Used

* Python
* Pyomo
* HiGHS / GLPK / CBC
* Pandas
* NumPy
* Matplotlib & Seaborn

---

## License, Academic Context & Reference

* This project is intended for academic use within: Optimization I & Communication Skills
* Team: Matisse Aaftink-De Luca · Fernando Cordero · Qin Eguía · Tamara Shopova
* Reference Article: Llenas et al. (2026) *"PepsiCo Deploys AI-Driven Pricing and Promotion Optimization at Scale"*, *INFORMS Journal on Applied Analytics*.

No formal license has been defined.
