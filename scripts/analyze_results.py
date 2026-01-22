# scripts/analyze_results.py
"""
Analyze experiments CSV and produce summary CSV + figures.

Usage (from project root):
    python scripts/analyze_results.py

Outputs:
    - results/best_defenses_summary.csv
    - results/figures/rmse_vs_frac_<attack>.png
    - results/figures/avg_rmse_per_defense.png
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # use non-interactive backend that can save figures without a display/Tk

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8,5)

# Resolve project root relative to this script (robust)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Candidate CSV locations (prefer project/results)
candidates = [
    os.path.join(RESULTS_DIR, "experiments.csv"),
    os.path.join(PROJECT_ROOT, "experiments.csv"),
    os.path.join(PROJECT_ROOT, "..", "results", "experiments.csv")
]

experiments_csv = None
for c in candidates:
    if os.path.exists(c):
        experiments_csv = c
        break

if experiments_csv is None:
    raise FileNotFoundError(f"Could not find experiments.csv in expected locations: {candidates}")

print("Loading experiments CSV from:", experiments_csv)
df = pd.read_csv(experiments_csv)

# Ensure numeric columns
df['rmse'] = pd.to_numeric(df['rmse'], errors='coerce')
df['r2'] = pd.to_numeric(df['r2'], errors='coerce')

print("\n=== Basic info ===")
print(f"Rows: {len(df)}")
print("Columns:", list(df.columns))
print("Unique attacks:", df['attack'].unique().tolist())
print("Unique defenses:", df['defense'].unique().tolist())
print("Poison fractions:", sorted(df['poison_fraction'].unique().tolist()))

print("\n=== Head (first 10 rows) ===")
print(df.head(10).to_string(index=False))

# Pivot to get mean RMSE per attack/frac/defense
pivot = df.pivot_table(index=['attack','poison_fraction','defense'], values='rmse', aggfunc='mean').reset_index()

# Determine best defense per (attack, poison_fraction)
best = pivot.loc[pivot.groupby(['attack','poison_fraction'])['rmse'].idxmin()].reset_index(drop=True)
best_csv = os.path.join(RESULTS_DIR, "best_defenses_summary.csv")
best.to_csv(best_csv, index=False)
print(f"\nSaved best defenses summary to: {best_csv}")

# Average RMSE per defense (ranking)
avg_by_def = df.groupby('defense')['rmse'].mean().reset_index().sort_values('rmse')
print("\n=== Average RMSE per defense (lower is better) ===")
print(avg_by_def.to_string(index=False))

# Save avg_by_def CSV
avg_csv = os.path.join(RESULTS_DIR, "avg_rmse_per_defense.csv")
avg_by_def.to_csv(avg_csv, index=False)
print(f"Saved average RMSE per defense to: {avg_csv}")

# Plot: RMSE vs poison_fraction per defense for each attack
for attack in df['attack'].unique():
    subset = pivot[pivot['attack'] == attack]
    plt.figure()
    sns.lineplot(data=subset, x='poison_fraction', y='rmse', hue='defense', marker='o')
    plt.title(f'RMSE vs Poison Fraction — Attack: {attack}')
    plt.xlabel('Poison fraction')
    plt.ylabel('RMSE on clean test set')
    plt.ylim(bottom=0)  # RMSE can't be negative
    plt.legend(title='Defense', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    fname = os.path.join(FIG_DIR, f"rmse_vs_frac_{attack}.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print("Saved plot:", fname)

# Plot: average RMSE per defense (bar)
plt.figure(figsize=(8,5))
sns.barplot(data=avg_by_def, x='defense', y='rmse')
plt.title('Average RMSE per Defense (lower is better)')
plt.ylabel('Average RMSE')
plt.xlabel('Defense')
plt.xticks(rotation=15)
plt.tight_layout()
avg_plot = os.path.join(FIG_DIR, "avg_rmse_per_defense.png")
plt.savefig(avg_plot, dpi=200)
plt.close()
print("Saved plot:", avg_plot)

# Additional helpful table: best defense counts (how often each defense was best)
best_counts = best['defense'].value_counts().reset_index()
best_counts.columns = ['defense', 'best_count']
best_counts_csv = os.path.join(RESULTS_DIR, "best_defense_counts.csv")
best_counts.to_csv(best_counts_csv, index=False)
print("Saved best defense counts to:", best_counts_csv)

print("\nAll analysis complete. Check the 'results' folder for CSVs and the 'results/figures' folder for PNGs.")