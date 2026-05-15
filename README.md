# EE 608 — Optimal Shipping Mode Selection in E-Commerce

**Team:** Pranil Pawar · Daniel Landau · Sabbir Ujjal  
**Course:** EE 608 Applied Modeling and Optimization — Stevens Institute of Technology

---

## Problem

Assign N e-commerce orders to shipping modes (Ship, Flight, Road) to minimize total cost while keeping average on-time delivery rate above R_min.

Formulated as an Integer Linear Program (ILP). Because cost and on-time parameters are mode-level averages, the full N×3 binary ILP reduces to 3 integer variables — solvable in milliseconds.

---

## Key Results

| | Value |
|---|---|
| R_min (selected) | 0.4156 |
| ILP optimal cost | $1,621,849.79 |
| LP relaxation cost | $1,621,849.56 |
| Integrality gap | $0.23 (0.000014%) |
| n_Road | 6,679 (86.8%) |
| n_Flight | 1,020 (13.2%) |
| n_Ship | 0 |

Note: Flight has the lowest average cost in this specific dataset because the cost column reflects product value, not carrier fees. This is a dataset characteristic, not a general rule.

---

## Dataset

Kaggle: [prachi13/customer-analytics](https://www.kaggle.com/datasets/prachi13/customer-analytics)

The `Reached_on_time_Y_N` column is inverted — 1 means late. Corrected with `on_time = 1 - Reached_on_time_Y_N`.

---

## Files

```
├── ILP.ipynb          ← main notebook with full analysis (Sabbir Ujjal)
├── main.py            ← standalone script to reproduce key results
├── requirements.txt
└── data/              ← put the Kaggle CSV here (not committed)
```

---

## How to Run

```bash
pip install -r requirements.txt
```

Put the dataset CSV in the `data/` folder, then:

```bash
python main.py
```

Should reproduce: αFlight=210.17, αRoad=210.73, αShip=210.72, rRoad=0.4176 (best), n_Road=6679, n_Flight=1020, gap=0.000014%

---

## Model

**Decision variable:** x_im ∈ {0,1} — 1 if order i goes to mode m

**Objective:** min Σ_i Σ_m α_m · x_im

**Constraints:**
- Σ_m x_im = 1 for all i (each order gets one mode)
- (1/N) Σ_i Σ_m r_m · x_im ≥ R_min (service level)
- x_im ∈ {0,1}

**Reduced form** (since α_m and r_m are mode-level constants):

```
min  Σ_m α_m · n_m
s.t. Σ_m n_m = N
     Σ_m r_m · n_m ≥ R_min · N
     n_m ∈ Z≥0
```

Feasibility: max_m(r_m) ≥ R_min → r_Road = 0.4176 ≥ 0.4156 ✓
