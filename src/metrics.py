"""
metrics.py
----------
Custom evaluation metrics if needed later.
"""

from sklearn.metrics import mean_squared_error,r2_score
import math

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))



def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)
