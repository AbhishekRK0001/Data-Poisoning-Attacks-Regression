# src/defenses.py
"""
Robust defenses for regression poisoning experiments.
Includes:
- zscore_filter
- iqr_filter
- isolation_forest_filter
- ransac_regression (compatible across sklearn versions)
"""

import numpy as np
import inspect

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

# attempt to import RANSACRegressor from sklearn; name may be in different module versions
try:
    # modern sklearn
    from sklearn.linear_model import RANSACRegressor as _RANSAC
except Exception:
    # fallback: import from experimental or other locations if needed
    try:
        from sklearn.linear_model._ransac import RANSACRegressor as _RANSAC  # unlikely but safe
    except Exception:
        _RANSAC = None


def zscore_filter(X, y, threshold=3.0):
    """
    Removes points with high Z-score in the target variable.
    Returns (X_filtered, y_filtered) as the same types (pandas DataFrame / Series).
    """
    # Support pandas Series / DataFrame or numpy arrays
    if hasattr(y, "values"):
        y_vals = y.values
    else:
        y_vals = np.asarray(y)

    z = np.abs((y_vals - np.mean(y_vals)) / (np.std(y_vals) + 1e-12))
    mask = z < threshold

    # apply mask preserving types
    if hasattr(y, "iloc"):
        return X.iloc[mask].copy(), y.iloc[mask].copy()
    else:
        return X[mask].copy(), y[mask].copy()


def iqr_filter(X, y, multiplier=1.5):
    """
    Removes outliers using the IQR rule on the target variable.
    Returns filtered (X, y).
    """
    if hasattr(y, "quantile"):
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        mask = (y >= lower) & (y <= upper)
        return X[mask].copy(), y[mask].copy()
    else:
        # numpy fallback
        y_vals = np.asarray(y)
        Q1 = np.percentile(y_vals, 25)
        Q3 = np.percentile(y_vals, 75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        mask = (y_vals >= lower) & (y_vals <= upper)
        return X[mask].copy(), y[mask].copy()


def isolation_forest_filter(X, y, contamination=0.05, random_state=42):
    """
    Uses IsolationForest on features X to detect outliers and remove them.
    Returns filtered (X, y).
    """
    # If X is a pandas DataFrame, use its values for the model but preserve indexing
    X_vals = X.values if hasattr(X, "values") else np.asarray(X)
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    preds = clf.fit_predict(X_vals)  # 1 for inliers, -1 for outliers
    mask = preds == 1

    if hasattr(X, "iloc"):
        return X.iloc[mask].copy(), y.iloc[mask].copy()
    else:
        return X[mask].copy(), y[mask].copy()


def ransac_regression(X_train, y_train, random_state=42):
    """
    Fits a RANSAC regressor robustly across sklearn versions.
    Returns the fitted model (supports .predict).
    """
    if _RANSAC is None:
        # If RANSAC is not available in sklearn import path, fallback to LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    base = LinearRegression()

    # Inspect constructor to determine whether it accepts 'estimator' or 'base_estimator'
    try:
        sig = inspect.signature(_RANSAC.__init__)
        params = sig.parameters
    except Exception:
        params = {}

    try:
        if 'estimator' in params:
            ransac = _RANSAC(estimator=base, random_state=random_state)
        elif 'base_estimator' in params:
            ransac = _RANSAC(base_estimator=base, random_state=random_state)
        else:
            # try positional or default
            try:
                ransac = _RANSAC(base, random_state=random_state)
            except Exception:
                ransac = _RANSAC(random_state=random_state)
    except TypeError:
        # last resort: try without passing base estimator
        ransac = _RANSAC(random_state=random_state)

    ransac.fit(X_train, y_train)
    return ransac
