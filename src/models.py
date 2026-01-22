"""
models.py
---------
Defines regression model training functions.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import math

def train_model(model_name, X_train, y_train):
    """Initializes and trains a regression model."""
    if model_name == "linear":
        model = LinearRegression()
    elif model_name == "ridge":
        model = Ridge(alpha=1.0)
    elif model_name == "lasso":
        model = Lasso(alpha=0.1)
    else:
        raise ValueError("Invalid model name")

    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    """Returns evaluation metrics for a model."""
    preds = model.predict(X_test)

    
    mse = mean_squared_error(y_test, preds)   # mean squared error (always returns MSE)
    rmse = math.sqrt(mse)                     # root mean squared error
    r2 = r2_score(y_test, preds)

    return {
        "rmse": rmse,
        "r2": r2,
        "predictions": preds
    }
