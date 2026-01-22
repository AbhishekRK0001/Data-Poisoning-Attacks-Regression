"""
attacks.py
----------
Simulates data poisoning attacks on regression datasets.
"""

import numpy as np
import pandas as pd


def label_flip(X, y, fraction=0.05, bias=5.0, random_state=42):
    """Adds bias to a fraction of target labels."""
    np.random.seed(random_state)

    y_new = y.copy().astype(float)
    n = len(y)
    idx = np.random.choice(n, int(n * fraction), replace=False)

    y_new.iloc[idx] += bias  # flip labels by adding bias
    return X.copy(), y_new


def outlier_injection(X, y, k=20, feature_scale=10, target_scale=20, random_state=42):
    """Injects outlier points into dataset."""
    np.random.seed(random_state)

    X_out = X.sample(n=k, replace=True).copy().reset_index(drop=True)
    X_out += feature_scale * np.random.randn(*X_out.shape)

    y_out = pd.Series(
        y.sample(n=k, replace=True).values + target_scale * np.random.randn(k)
    )

    X_poisoned = pd.concat([X.reset_index(drop=True), X_out], ignore_index=True)
    y_poisoned = pd.concat([y.reset_index(drop=True), y_out], ignore_index=True)

    return X_poisoned, y_poisoned
