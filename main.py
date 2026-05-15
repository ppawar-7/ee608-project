import pandas as pd
import matplotlib.pyplot as plt
import pulp
from sklearn.model_selection import train_test_split

CSV_PATH = "data/ecommerce_shipments.csv"   # change this if the filename is different
R_MIN = 0.4156
SEED = 42

df = pd.read_csv(CSV_PATH)

# 1 means late in this dataset, so flip it
df["on_time"] = 1 - df["Reached.on.Time_Y.N"]
df = df[["Mode of Shipment", "Cost of the Product", "on_time"]].copy()
df.columns = ["mode", "cost", "on_time"]

train, temp = train_test_split(df, test_size=0.30, random_state=SEED)
val, test = train_test_split(temp, test_size=0.50, random_state=SEED)

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)

print("Train:", len(train), "Val:", len(val), "Test:", len(test))

stats = train.groupby("mode").agg(
    alpha=("cost", "mean"),
    r=("on_time", "mean"),
    n=("mode", "size")
)

print("\nTrain mode stats:")
print(stats)

modes = ["Flight", "Ship", "Road"]
stats = stats.reindex(modes)

N = len(train)

def solve_model(stats, N, r_min, integer=True):
    prob = pulp.LpProblem("shipping", pulp.LpMinimize)
    cat = pulp.LpInteger if integer else pulp.LpContinuous
    n = {}

    for m in modes:
        n[m] = pulp.LpVariable(f"n_{m}", lowBound=0, cat=cat)

    prob += pulp.lpSum(stats.loc[m, "alpha"] * n[m] for m in modes)
    prob += pulp.lpSum(n[m] for m in modes) == N
    prob += pulp.lpSum(stats.loc[m, "r"] * n[m] for m in modes) >= r_min * N

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    counts = {m: pulp.value(n[m]) for m in modes}
    cost = pulp.value(prob.objective)
    avg_on_time = sum(stats.loc[m, "r"] * counts[m] for m in modes) / N

    return counts, cost, avg_on_time

lp_sol = solve_model(stats, N, R_MIN, integer=False)
ilp_sol = solve_model(stats, N, R_MIN, integer=True)

if lp_sol is None or ilp_sol is None:
    print("\nProblem is infeasible at R_MIN =", R_MIN)
    raise SystemExit

lp_counts, lp_cost, lp_on_time = lp_sol
ilp_counts, ilp_cost, ilp_on_time = ilp_sol

gap = ilp_cost - lp_cost
gap_pct = gap / lp_cost * 100

print("\nLP cost:", f"${lp_cost:,.2f}")
print("ILP cost:", f"${ilp_cost:,.2f}")
print("Gap:", f"${gap:.2f}", f"({gap_pct:.6f}%)")
print("Counts:", ilp_counts)
print("Avg on-time:", f"{ilp_on_time:.4f}")

print("\nValidation on val/test:")

for name, split_df in [("Val", val), ("Test", test)]:
    split_stats = split_df.groupby("mode").agg(
        alpha=("cost", "mean"),
        r=("on_time", "mean")
    ).reindex(modes)

    split_N = len(split_df)
    total = sum(ilp_counts.values())

    scaled = {m: round(ilp_counts[m] / total * split_N) for m in modes}
    diff = split_N - sum(scaled.values())
    if diff != 0:
        scaled["Road"] += diff

    on_time = sum(split_stats.loc[m, "r"] * scaled[m] for m in modes) / split_N
    print(name, "on-time with train allocation:", f"{on_time:.4f}")

print("\nSensitivity sweep:")
sweep_thresholds = [0.35, 0.38, 0.40, 0.405, 0.410, 0.4156, 0.4176, 0.4226]
rows = []

for r in sweep_thresholds:
    sol = solve_model(stats, N, r, integer=True)
    if sol is None:
        rows.append({"r_min": r, "status": "Infeasible", "cost": None})
        print("  R_min =", r, "INFEASIBLE")
    else:
        counts, cost, avg_on_time = sol
        rows.append({
            "r_min": r,
            "status": "Optimal",
            "cost": cost,
            "n_Flight": round(counts.get("Flight", 0)),
            "n_Ship": round(counts.get("Ship", 0)),
            "n_Road": round(counts.get("Road", 0)),
            "avg_on_time": avg_on_time
        })
        print("  R_min =", r, "cost =", f"${cost:,.0f}", "on_time =", f"{avg_on_time:.4f}")

sweep_df = pd.DataFrame(rows)
sweep_df.to_csv("rmin_sweep.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(modes, stats["alpha"])
axes[0].set_title("Average Shipping Cost by Mode")
axes[0].set_ylabel("Avg cost ($)")
for i, v in enumerate(stats["alpha"]):
    axes[0].text(i, v + 0.01, f"${v:.2f}", ha="center", fontsize=9)

axes[1].bar(modes, stats["r"])
axes[1].set_title("On-Time Delivery Rate by Mode")
axes[1].set_ylabel("On-time rate")
for i, v in enumerate(stats["r"]):
    axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("mode_stats.png", dpi=150)
plt.show()

feasible = sweep_df[sweep_df["status"] == "Optimal"].dropna(subset=["cost"])
plt.figure(figsize=(8, 4))
plt.plot(feasible["r_min"], feasible["cost"], marker="o")
plt.axvline(R_MIN, color="orange", linestyle="--", label=f"Selected R_min={R_MIN}")
plt.axvline(0.4176, color="red", linestyle=":", label="Infeasible above 0.4176")
plt.xlabel("R_min")
plt.ylabel("Total cost ($)")
plt.title("Optimal Cost vs Service Threshold")
plt.legend()
plt.tight_layout()
plt.savefig("cost_vs_rmin.png", dpi=150)
plt.show()
