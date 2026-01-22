"""
experiment.py
--------------
Runs full pipeline: clean → poison → defend → train → evaluate.
"""

import joblib
import pandas as pd

from src.data_utils import load_housing, get_split
from src.models import train_model, evaluate
from src.attacks import label_flip, outlier_injection
from src.defenses import zscore_filter, iqr_filter, ransac_regression


def run_experiment(
    model_name="linear",
    attack="label_flip",
    defense="zscore",
    poison_fraction=0.1
):
    # Load and split dataset
    X, y = load_housing()
    X_train, X_test, y_train, y_test = get_split(X, y)

    # Apply attack
    if attack == "label_flip":
        X_train, y_train = label_flip(X_train, y_train, fraction=poison_fraction)
    elif attack == "outlier":
        X_train, y_train = outlier_injection(X_train, y_train, k=30)

    # Apply defense
    if defense == "zscore":
        X_train, y_train = zscore_filter(X_train, y_train)
    elif defense == "iqr":
        X_train, y_train = iqr_filter(X_train, y_train)
    elif defense == "ransac":
        model = ransac_regression(X_train, y_train)
        return evaluate(model, X_test, y_test)

    # Train model
    model = train_model(model_name, X_train, y_train)

    # Evaluate
    metrics = evaluate(model, X_test, y_test)
    return metrics


if __name__ == "__main__":
    result = run_experiment(
        model_name="linear",
        attack="outlier",
        defense="zscore",
        poison_fraction=0.1
    )
    print("Experiment Result:", result)
