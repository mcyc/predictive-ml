"""
evaluate.py
-----------
Evaluation metrics and reporting utilities for load forecasting models.

Main functions:
    evaluate(model_name, y_true, y_pred)
        Global MAE, RMSE, MAPE — appends to a results list.

    peak_weighted_evaluate(model_name, y_true, y_pred, threshold)
        Same metrics computed only on peak hours (top X% of actual load).

    compare_models(results)
        Pretty-prints a comparison table from a list of evaluate() outputs.

    plot_residuals(results_df, models)
        Residuals over time + predicted vs actual for each model.

    plot_residuals_by_time(results_df, model_col, resid_col)
        Mean residuals by hour, weekday, and month for a single model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker


# ── Core metrics ──────────────────────────────────────────────────────────────

def _mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def _rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def _mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate(model_name, y_true, y_pred, results=None):
    """
    Compute and print global MAE, RMSE, and MAPE for a model.

    Parameters
    ----------
    model_name : str
        Label for the model (used in print output and results list).
    y_true : array-like
        Actual load values.
    y_pred : array-like
        Predicted load values.
    results : list or None
        If provided, appends the result dict to this list in place.
        Allows accumulating results across models for compare_models().

    Returns
    -------
    dict
        {'model': model_name, 'MAE': ..., 'RMSE': ..., 'MAPE': ...}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    result = {
        "model": model_name,
        "MAE":   _mae(y_true, y_pred),
        "RMSE":  _rmse(y_true, y_pred),
        "MAPE":  _mape(y_true, y_pred),
    }

    print(f"\n── {model_name} ──")
    print(f"  MAE:  {result['MAE']:,.1f} MW")
    print(f"  RMSE: {result['RMSE']:,.1f} MW")
    print(f"  MAPE: {result['MAPE']:.2f}%")

    if results is not None:
        results.append(result)

    return result


def peak_weighted_evaluate(model_name, y_true, y_pred, threshold=0.90, results=None):
    """
    Compute and print MAE, RMSE, and MAPE restricted to peak hours.

    Peak hours are defined as hours where actual load exceeds the
    `threshold` quantile of y_true (e.g., top 10% by default).

    This metric is more relevant than global MAPE for BESS peak shaving,
    where accuracy during high-load hours matters most.

    Parameters
    ----------
    model_name : str
    y_true : array-like
    y_pred : array-like
    threshold : float
        Quantile above which hours are considered peaks. Default: 0.90.
    results : list or None
        If provided, appends the result dict to this list in place.

    Returns
    -------
    dict
        {'model': model_name, 'MAE_peak': ..., 'RMSE_peak': ...,
         'MAPE_peak': ..., 'n_peak_hours': ...}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    peak_mask = y_true >= np.quantile(y_true, threshold)
    y_true_peak = y_true[peak_mask]
    y_pred_peak = y_pred[peak_mask]

    result = {
        "model":        model_name,
        "MAE_peak":     _mae(y_true_peak, y_pred_peak),
        "RMSE_peak":    _rmse(y_true_peak, y_pred_peak),
        "MAPE_peak":    _mape(y_true_peak, y_pred_peak),
        "n_peak_hours": int(peak_mask.sum()),
        "threshold_pct": int(threshold * 100),
    }

    print(f"\n── {model_name} (top {result['threshold_pct']}% peak hours) ──")
    print(f"  Peak hours:  {result['n_peak_hours']}")
    print(f"  MAE_peak:    {result['MAE_peak']:,.1f} MW")
    print(f"  RMSE_peak:   {result['RMSE_peak']:,.1f} MW")
    print(f"  MAPE_peak:   {result['MAPE_peak']:.2f}%")

    if results is not None:
        results.append(result)

    return result


def compare_models(results, peak=False):
    """
    Print a formatted comparison table from a list of result dicts.

    Parameters
    ----------
    results : list of dict
        Output of evaluate() or peak_weighted_evaluate() calls.
    peak : bool
        If True, display peak metric columns instead of global ones.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame(results)
    if peak:
        cols = ["model", "MAE_peak", "RMSE_peak", "MAPE_peak", "n_peak_hours"]
    else:
        cols = ["model", "MAE", "RMSE", "MAPE"]

    df = df[[c for c in cols if c in df.columns]]
    print("\n── Model Comparison ──")
    print(df.to_string(index=False))
    return df


# ── Plotting utilities ────────────────────────────────────────────────────────

def plot_residuals(results_df, models, figsize=(16, 4)):
    """
    Plot predicted vs actual and residuals over time for each model.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have DatetimeIndex and columns:
        'actual', and for each model a 'pred_{key}' and 'resid_{key}' column.
    models : list of (name, pred_col, resid_col, color) tuples
        e.g. [('Random Forest', 'pred_rf', 'resid_rf', 'seagreen'), ...]
    figsize : tuple
        Figure size per row. Total height = figsize[1] * n_models.
    """
    n = len(models)
    fig, axes = plt.subplots(n, 2, figsize=(figsize[0], figsize[1] * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    date_min = results_df.index.min()
    date_max = results_df.index.max()
    title = (f"Residual Analysis — Test Set "
             f"({date_min.date()} → {date_max.date()})")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for i, (name, pred_col, resid_col, color) in enumerate(models):
        # Predicted vs Actual
        ax1 = axes[i, 0]
        ax1.scatter(results_df["actual"], results_df[pred_col],
                    alpha=0.15, s=5, color=color)
        lims = [results_df["actual"].min(), results_df["actual"].max()]
        ax1.plot(lims, lims, "k--", linewidth=1, label="Perfect fit")
        ax1.set_xlabel("Actual (MW)")
        ax1.set_ylabel("Predicted (MW)")
        ax1.set_title(f"{name} — Predicted vs Actual")
        ax1.legend(fontsize=8)

        # Residuals over time
        ax2 = axes[i, 1]
        ax2.plot(results_df.index, results_df[resid_col],
                 color=color, linewidth=0.4, alpha=0.7)
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Residual (MW)")
        ax2.set_title(f"{name} — Residuals Over Time")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    return fig


def plot_residuals_by_time(results_df, model_name, resid_col, figsize=(16, 4)):
    """
    Plot mean residuals by hour of day, weekday, and month for one model.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have a DatetimeIndex and the residual column specified.
    model_name : str
        Used in the plot title.
    resid_col : str
        Column name of the residuals to plot.
    figsize : tuple
    """
    df = results_df.copy()
    df["hour"]    = df.index.hour
    df["weekday"] = df.index.day_name()
    df["month"]   = df.index.month_name()

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
    month_order   = ["January", "February", "March", "April", "May", "June",
                     "July", "August", "September", "October",
                     "November", "December"]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f"{model_name} — Residuals by Time Component",
                 fontsize=13, fontweight="bold")

    color = "steelblue"

    for ax, col, order, title in [
        (axes[0], "hour",    None,          "By Hour of Day"),
        (axes[1], "weekday", weekday_order, "By Weekday"),
        (axes[2], "month",   month_order,   "By Month"),
    ]:
        grouped = df.groupby(col)[resid_col].mean()
        if order:
            grouped = grouped.reindex(order)
        grouped.plot(kind="bar", ax=ax, color=color, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_ylabel("Mean Residual (MW)")
        ax.set_xlabel("")
        if order:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    return fig


def build_results_df(y_test, predictions):
    """
    Assemble a results DataFrame with actuals, predictions, and residuals.

    Parameters
    ----------
    y_test : pd.Series
        Actual load values with DatetimeIndex.
    predictions : dict
        {key: y_pred array} e.g. {'lr': y_pred_lr, 'rf': y_pred_rf}

    Returns
    -------
    pd.DataFrame
        Columns: actual, pred_{key}, resid_{key} for each key.
    """
    df = pd.DataFrame({"actual": y_test})
    for key, y_pred in predictions.items():
        df[f"pred_{key}"]  = y_pred
        df[f"resid_{key}"] = df["actual"] - y_pred
    return df
