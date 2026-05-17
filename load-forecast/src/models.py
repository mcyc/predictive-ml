"""
models.py
---------
Model training utilities for short-term electrical load forecasting.

Provides thin wrappers around scikit-learn and LightGBM with sensible
default hyperparameters established during initial exploration. Parameters
can be overridden via kwargs for experimentation.

Main functions:
    train_linear(X_train, y_train)
        Linear regression baseline.

    train_rf(X_train, y_train, **kwargs)
        Random Forest with default hyperparameters.

    train_lgbm(X_train, y_train, X_val, y_val, **kwargs)
        LightGBM with early stopping.

    train_all(X_train, y_train, X_val, y_val)
        Convenience wrapper — trains all three models and returns a dict.

Hyperparameter notes:
    Defaults reflect the best configuration found during exploration on
    the north grid dataset (2017–2022). They are reasonable starting points
    for new datasets but should be re-validated, especially:
        - RF:   n_estimators, max_depth, min_samples_leaf
        - LGBM: n_estimators, num_leaves, learning_rate
"""

import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# ── Default hyperparameters ───────────────────────────────────────────────────

RF_DEFAULTS = dict(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
)

LGBM_DEFAULTS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=64,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)

LGBM_EARLY_STOPPING_ROUNDS = 50


# ── Public API ────────────────────────────────────────────────────────────────

def train_linear(X_train, y_train):
    """
    Train a Linear Regression baseline model.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
    y_train : pd.Series or np.ndarray

    Returns
    -------
    sklearn.linear_model.LinearRegression (fitted)
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Trained: Linear Regression")
    return model


def train_rf(X_train, y_train, **kwargs):
    """
    Train a Random Forest regressor.

    Default hyperparameters can be overridden via kwargs, e.g.:
        train_rf(X_train, y_train, n_estimators=500, max_depth=15)

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
    y_train : pd.Series or np.ndarray
    **kwargs
        Overrides for RF_DEFAULTS.

    Returns
    -------
    sklearn.ensemble.RandomForestRegressor (fitted)
    """
    params = {**RF_DEFAULTS, **kwargs}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    print(f"Trained: Random Forest  "
          f"(n_estimators={params['n_estimators']}, "
          f"max_depth={params['max_depth']}, "
          f"min_samples_leaf={params['min_samples_leaf']})")
    return model


def train_lgbm(X_train, y_train, X_val, y_val,
               early_stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS,
               objective="regression",
               **kwargs):
    """
    Train a LightGBM regressor with early stopping on a validation set.

    Default hyperparameters can be overridden via kwargs, e.g.:
        train_lgbm(X_train, y_train, X_val, y_val, num_leaves=128)

    For quantile regression (used in peak probability experiments):
        train_lgbm(..., objective='quantile', alpha=0.90)

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
    y_train : pd.Series or np.ndarray
    X_val   : pd.DataFrame or np.ndarray
        Validation features for early stopping (typically X_test).
    y_val   : pd.Series or np.ndarray
        Validation target for early stopping (typically y_test).
    early_stopping_rounds : int
        Stop if validation loss doesn't improve for this many rounds.
    objective : str
        LightGBM objective. 'regression' for point forecast,
        'quantile' for quantile regression. Default: 'regression'.
    **kwargs
        Overrides for LGBM_DEFAULTS.

    Returns
    -------
    lgb.LGBMRegressor (fitted)
    """
    params = {**LGBM_DEFAULTS, "objective": objective, **kwargs}
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(100),
        ],
    )
    best = model.best_iteration_
    print(f"Trained: LightGBM  "
          f"(best_iteration={best}, "
          f"objective={objective}, "
          f"num_leaves={params['num_leaves']}, "
          f"learning_rate={params['learning_rate']})")
    return model


def train_all(X_train, y_train, X_val, y_val):
    """
    Train the full model ladder: Linear Regression, Random Forest, LightGBM.

    Convenience wrapper for notebooks that want all three models
    without separate calls. Returns a dict keyed by short model name.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val     : validation / test data (used for LGBM early stopping)

    Returns
    -------
    dict
        {
          'lr':   LinearRegression (fitted),
          'rf':   RandomForestRegressor (fitted),
          'lgbm': LGBMRegressor (fitted),
        }
    """
    print("── Training model ladder ──")
    models = {}
    models["lr"]   = train_linear(X_train, y_train)
    models["rf"]   = train_rf(X_train, y_train)
    models["lgbm"] = train_lgbm(X_train, y_train, X_val, y_val)
    print("── Done ──")
    return models


def get_feature_importance(model, features, model_type="rf", top_n=None):
    """
    Extract and return a sorted feature importance DataFrame.

    Parameters
    ----------
    model : fitted model
        RandomForestRegressor or LGBMRegressor.
    features : list of str
        Feature names corresponding to model input columns.
    model_type : str
        'rf' for sklearn importance, 'lgbm' for gain-based importance.
    top_n : int or None
        If set, return only the top N features. Default: all.

    Returns
    -------
    pd.DataFrame
        Columns: ['feature', 'importance'], sorted descending.
    """
    if model_type == "lgbm":
        importances = model.booster_.feature_importance(importance_type="gain")
    else:
        importances = model.feature_importances_

    df = (
        pd.DataFrame({"feature": features, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    if top_n is not None:
        df = df.head(top_n)

    return df
