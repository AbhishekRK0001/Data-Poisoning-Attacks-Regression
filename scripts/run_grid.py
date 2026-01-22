# scripts/run_grid.py
import os
import sys
import csv
import numpy as np
import pandas as pd

# Allow importing src from project root
# Allow importing src from project root (robust regardless of current working directory)
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # scripts/ directory
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # project root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Imports
from src.data_utils import load_housing, get_split
from src.attacks import label_flip, outlier_injection
from src.defenses import zscore_filter, iqr_filter, isolation_forest_filter, ransac_regression
from src.models import train_model, evaluate

# Settings
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def apply_defense(defense_name, Xp, yp):
    """Returns (Xd, yd) or None for RANSAC special case."""
    if defense_name == "none":
        return Xp, yp, None

    elif defense_name == "zscore":
        Xd, yd = zscore_filter(Xp, yp, threshold=3.0)
        return Xd, yd, None

    elif defense_name == "iqr":
        Xd, yd = iqr_filter(Xp, yp)
        return Xd, yd, None

    elif defense_name == "isolation_forest":
        Xd, yd = isolation_forest_filter(Xp, yp, contamination=0.05)
        return Xd, yd, None

    elif defense_name == "ransac":
        # RANSAC returns a trained model instead of cleaned data
        model = ransac_regression(Xp, yp)
        return None, None, model

    else:
        raise ValueError("Unknown defense")


def run_experiment():
    print("\n=== Running Medium Grid Experiments ===\n")

    # Load clean data
    X, y = load_housing()
    X_train, X_test, y_train, y_test = get_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Attack types
    attacks = {
        "label_flip": label_flip,
        "outlier_injection": outlier_injection
    }

    # Poison fractions
    poison_fracs = [0.01, 0.05, 0.1, 0.2]

    # Defenses
    defenses = ["none", "zscore", "iqr", "isolation_forest", "ransac"]

    # Ensure results folder exists
    os.makedirs("../results", exist_ok=True)

    # Create CSV file
    outfile = "../results/experiments.csv"
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "attack", "poison_fraction", "defense",
            "n_before", "n_after", "rmse", "r2"
        ])

        # Loop through attack types
        for attack_name, attack_fn in attacks.items():

            for frac in poison_fracs:
                print(f"\n--- Attack: {attack_name}, Poison fraction: {frac} ---")

                # Apply attack
                if attack_name == "label_flip":
                    Xp, yp = attack_fn(X_train, y_train, fraction=frac, bias=10.0)

                elif attack_name == "outlier_injection":
                    k = int(len(X_train) * frac)
                    Xp, yp = attack_fn(
                        X_train, y_train, k=k,
                        feature_scale=10, target_scale=30
                    )

                n_before = len(Xp)

                # Apply defenses
                for defense in defenses:
                    print(f"  → Running defense: {defense}")

                    Xd, yd, ransac_model = apply_defense(defense, Xp, yp)

                    # Special case — RANSAC
                    if defense == "ransac":
                        metrics = evaluate(ransac_model, X_test, y_test)
                        writer.writerow([
                            attack_name, frac, defense,
                            n_before, n_before,
                            metrics["rmse"], metrics["r2"]
                        ])
                        continue

                    # If defense removed everything
                    if len(Xd) == 0:
                        writer.writerow([
                            attack_name, frac, defense,
                            n_before, 0, None, None
                        ])
                        continue

                    n_after = len(Xd)

                    # Train normal linear model
                    model = train_model("linear", Xd, yd)
                    metrics = evaluate(model, X_test, y_test)

                    writer.writerow([
                        attack_name, frac, defense,
                        n_before, n_after,
                        metrics["rmse"], metrics["r2"]
                    ])

    print("\n🎉 Grid complete!")
    print("Saved to: ../results/experiments.csv\n")


if __name__ == "__main__":
    run_experiment()