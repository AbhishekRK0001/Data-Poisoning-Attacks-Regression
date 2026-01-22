"""
data_utils.py
-------------
Handles dataset loading, splitting, and optional synthetic data generation.
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_housing():
    """Loads California Housing dataset (regression)."""
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


def get_split(X, y, test_size=0.2, random_state=42):
    """Returns train-test split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def generate_synthetic_regression(n_samples=500, noise=5.0, random_state=42):
    """Generates synthetic linear regression dataset."""
    import numpy as np
    np.random.seed(random_state)

    X = np.random.rand(n_samples, 1) * 10
    y = 3 * X[:, 0] + 7 + np.random.randn(n_samples) * noise

    return pd.DataFrame(X, columns=["feature"]), pd.Series(y, name="target")
