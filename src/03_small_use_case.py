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

# Import model builder from Step 2
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