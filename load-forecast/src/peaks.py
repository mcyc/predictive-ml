"""
peaks.py
--------
Peak labelling, characterisation, and analysis utilities for monthly
maximum load forecasting and BESS peak shaving.

All functions operate on a feature-engineered DataFrame produced by
build_features() with a DatetimeIndex at hourly frequency.

Main functions:
    label_monthly_peaks(df, threshold, target)
        Binary top-N% peak label within each calendar month.

    label_monthly_rank(df, target)
        Fractional rank of each hour within its calendar month (1.0 = max).

    get_monthly_maxima(df, target)
        Series of monthly maximum values and their timestamps.

    peak_distance_stats(df, quantiles, target)
        Distance from each quantile level to the monthly maximum.

    peak_duration_analysis(df, target, threshold)
        Characterise whether monthly maxima are isolated spikes or
        sustained high-load periods.

    peak_temporal_profile(df, target, threshold)
        Hour-of-day and month-of-year distributions of peak hours.

    empirical_alpha(df, target)
        Minimum quantile level that would have bounded every monthly
        maximum in the dataset — informs alpha selection for quantile
        regression.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

__version__  = "0.1.0"
__notebook__ = "04"


# ── Public API ────────────────────────────────────────────────────────────────

def label_monthly_peaks(df, threshold=0.90, target="north_clean"):
    """
    Assign a binary peak label: 1 if the hour is in the top (1-threshold)
    fraction of load within its calendar month, 0 otherwise.

    Uses year + month grouping so each month's threshold is computed
    independently, avoiding summer peaks dominating the label.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame with DatetimeIndex.
    threshold : float
        Quantile above which an hour is labelled as peak. Default: 0.90.
    target : str
        Load column name. Default: 'north_clean'.

    Returns
    -------
    pd.Series
        Binary Series (0/1) with same index as df, named 'peak'.
    """
    peak = (
        df.groupby([df.index.year, df.index.month])[target]
          .transform(lambda x: (x >= x.quantile(threshold)).astype(int))
    )
    peak.name = "peak"
    return peak


def label_monthly_rank(df, target="north_clean"):
    """
    Compute the fractional rank of each hour within its calendar month.

    Rank 1.0 = monthly maximum. Rank 0.0 = monthly minimum.
    Ties are averaged (method='average').

    This label is more informative than binary peak/non-peak for training
    sample-weighted models that should prioritise hours near the maximum.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame with DatetimeIndex.
    target : str
        Load column name. Default: 'north_clean'.

    Returns
    -------
    pd.Series
        Float Series in [0, 1] with same index as df, named 'monthly_rank'.
    """
    rank = (
        df.groupby([df.index.year, df.index.month])[target]
          .transform(lambda x: x.rank(pct=True))
    )
    rank.name = "monthly_rank"
    return rank


def get_monthly_maxima(df, target="north_clean"):
    """
    Return the monthly maximum load value and its timestamp for each month.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame with DatetimeIndex.
    target : str
        Load column name. Default: 'north_clean'.

    Returns
    -------
    pd.DataFrame
        Indexed by (year, month) with columns:
            max_load  : float, monthly maximum load (MW)
            timestamp : pd.Timestamp, time of the monthly maximum
            hour      : int, hour of day of the maximum
            month     : int, calendar month
            year      : int, calendar year
    """
    records = []
    for (year, month), grp in df.groupby([df.index.year, df.index.month]):
        idx_max   = grp[target].idxmax()
        max_load  = grp[target].max()
        records.append({
            "year"     : year,
            "month"    : month,
            "max_load" : max_load,
            "timestamp": idx_max,
            "hour"     : idx_max.hour,
            "weekday"  : idx_max.day_name(),
        })

    result = pd.DataFrame(records).set_index(["year", "month"])
    print(f"Monthly maxima computed: {len(result)} months")
    print(f"  Overall max: {result['max_load'].max():,.0f} MW "
          f"({result['max_load'].idxmax()})")
    print(f"  Overall min: {result['max_load'].min():,.0f} MW "
          f"({result['max_load'].idxmin()})")
    print(f"  Mean monthly max: {result['max_load'].mean():,.0f} MW")
    return result


def peak_distance_stats(df, quantiles=(0.90, 0.95, 0.98, 0.99),
                        target="north_clean"):
    """
    Compute the distance from each quantile level to the monthly maximum.

    For each month, computes:
        gap_MW   = monthly_max - quantile_value
        gap_pct  = gap_MW / monthly_max * 100

    Summarises across all months to show how far above each quantile
    threshold the monthly maximum typically sits. This informs alpha
    selection for quantile regression.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame with DatetimeIndex.
    quantiles : tuple of float
        Quantile levels to evaluate. Default: (0.90, 0.95, 0.98, 0.99).
    target : str
        Load column name. Default: 'north_clean'.

    Returns
    -------
    pd.DataFrame
        Rows = quantile levels, columns = summary statistics of the gap.
    """
    records = []
    for q in quantiles:
        gaps_mw  = []
        gaps_pct = []
        for (year, month), grp in df.groupby([df.index.year, df.index.month]):
            monthly_max = grp[target].max()
            q_val       = grp[target].quantile(q)
            gap_mw      = monthly_max - q_val
            gaps_mw.append(gap_mw)
            gaps_pct.append(gap_mw / monthly_max * 100)

        records.append({
            "quantile"       : q,
            "gap_mw_mean"    : np.mean(gaps_mw),
            "gap_mw_median"  : np.median(gaps_mw),
            "gap_mw_max"     : np.max(gaps_mw),
            "gap_pct_mean"   : np.mean(gaps_pct),
            "gap_pct_median" : np.median(gaps_pct),
            "gap_pct_max"    : np.max(gaps_pct),
        })

    result = pd.DataFrame(records).set_index("quantile")
    print("\n── Distance from quantile to monthly maximum ──")
    print(result.round(2).to_string())
    return result


def peak_duration_analysis(df, target="north_clean", threshold=0.95,
                           window_hours=6):
    """
    Characterise whether monthly maxima are isolated spikes or part of
    sustained high-load periods.

    For each month, counts how many hours within `window_hours` of the
    monthly maximum also exceed the monthly `threshold` quantile. A count
    of 1 (only the maximum itself) indicates an isolated spike; a higher
    count indicates a sustained peak period.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame with DatetimeIndex.
    target : str
        Load column name. Default: 'north_clean'.
    threshold : float
        Quantile level defining 'high load'. Default: 0.95.
    window_hours : int
        Hours before and after the monthly maximum to inspect. Default: 6.

    Returns
    -------
    pd.DataFrame
        One row per month with columns:
            max_load, max_timestamp, n_high_hours_in_window,
            is_isolated (bool: True if n_high_hours_in_window == 1)
    """
    records = []
    for (year, month), grp in df.groupby([df.index.year, df.index.month]):
        monthly_max   = grp[target].max()
        monthly_q     = grp[target].quantile(threshold)
        idx_max       = grp[target].idxmax()

        # Window around maximum
        window_start = idx_max - pd.Timedelta(hours=window_hours)
        window_end   = idx_max + pd.Timedelta(hours=window_hours)
        window_data  = grp.loc[window_start:window_end, target]

        n_high = (window_data >= monthly_q).sum()

        records.append({
            "year"                   : year,
            "month"                  : month,
            "max_load"               : monthly_max,
            "max_timestamp"          : idx_max,
            "n_high_hours_in_window" : int(n_high),
            "is_isolated"            : (n_high == 1),
        })

    result = pd.DataFrame(records).set_index(["year", "month"])

    isolated_pct = result["is_isolated"].mean() * 100
    mean_window  = result["n_high_hours_in_window"].mean()
    print(f"\n── Peak Duration Analysis (threshold={threshold}, window=±{window_hours}h) ──")
    print(f"  Isolated spikes (only max hour above threshold): {isolated_pct:.1f}% of months")
    print(f"  Mean high-load hours in window: {mean_window:.1f}")
    print(f"  Max high-load hours in window:  {result['n_high_hours_in_window'].max()}")
    return result


def peak_temporal_profile(df, target="north_clean", threshold=0.90):
    """
    Analyse the hour-of-day, weekday, and month-of-year distributions
    of peak hours and monthly maxima.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame with DatetimeIndex.
    target : str
        Load column name. Default: 'north_clean'.
    threshold : float
        Quantile threshold defining peak hours. Default: 0.90.

    Returns
    -------
    dict with keys:
        'hour_dist'    : pd.Series — count of peak hours by hour of day
        'weekday_dist' : pd.Series — count of peak hours by weekday
        'month_dist'   : pd.Series — count of peak hours by calendar month
        'max_hour_dist': pd.Series — count of monthly maxima by hour of day
        'max_month_dist': pd.Series — count of monthly maxima by calendar month
    """
    peak_mask = label_monthly_peaks(df, threshold=threshold, target=target)
    peak_df   = df[peak_mask == 1].copy()
    maxima    = get_monthly_maxima(df, target=target)

    hour_dist     = peak_df.index.hour
    weekday_dist  = peak_df.index.day_name()
    month_dist    = peak_df.index.month

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
    month_names   = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    result = {
        "hour_dist"     : pd.Series(hour_dist).value_counts().sort_index(),
        "weekday_dist"  : (pd.Series(weekday_dist).value_counts()
                             .reindex(weekday_order)),
        "month_dist"    : (pd.Series(month_dist).value_counts()
                             .sort_index()
                             .rename(index=dict(enumerate(month_names, 1)))),
        "max_hour_dist": maxima["hour"].value_counts().sort_index(),
        "max_month_dist": maxima.reset_index()["month"].value_counts().sort_index()
        .rename(index=dict(enumerate(month_names, 1))),
    }
    return result


def empirical_alpha(df, target="north_clean",
                    quantiles=None):
    """
    Compute the minimum quantile level that would have bounded every monthly
    maximum in the dataset.

    For each month, finds the fractional rank of the monthly maximum within
    that month's load distribution. The maximum across all months is the
    minimum alpha needed to guarantee 100% coverage.

    Also reports coverage at standard alpha levels (0.90, 0.95, 0.98, 0.99).

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame with DatetimeIndex.
    target : str
        Load column name. Default: 'north_clean'.
    quantiles : list of float or None
        Alpha levels to evaluate coverage for. Default: [0.90, 0.95, 0.98, 0.99].

    Returns
    -------
    pd.DataFrame
        Coverage at each alpha level, plus the empirical minimum alpha.
    """
    if quantiles is None:
        quantiles = [0.90, 0.95, 0.98, 0.99]

    max_ranks   = []
    month_ranks = []

    for (year, month), grp in df.groupby([df.index.year, df.index.month]):
        monthly_max  = grp[target].max()
        # Rank of the maximum within the month (fractional, 0–1)
        rank_of_max  = (grp[target] < monthly_max).sum() / len(grp)
        max_ranks.append(rank_of_max)
        month_ranks.append({
            "year": year, "month": month,
            "rank_of_max": rank_of_max,
            "max_load": monthly_max,
        })

    rank_df    = pd.DataFrame(month_ranks).set_index(["year", "month"])
    min_alpha  = np.max(max_ranks)

    records = []
    for q in quantiles:
        n_bounded  = (rank_df["rank_of_max"] <= q).sum()
        coverage   = n_bounded / len(rank_df)
        records.append({
            "alpha"           : q,
            "months_bounded"  : int(n_bounded),
            "total_months"    : len(rank_df),
            "coverage"        : round(coverage, 4),
            "months_missed"   : int(len(rank_df) - n_bounded),
        })

    result = pd.DataFrame(records).set_index("alpha")

    print("\n── Empirical Alpha Analysis ──")
    print(f"  Minimum alpha for 100% monthly max coverage: {min_alpha:.4f}")
    print(f"  (i.e. the monthly max sits above the {min_alpha*100:.2f}th percentile "
          f"in its worst month)")
    print("\n  Coverage at standard alpha levels:")
    print(result.to_string())

    return result, rank_df


# ── Plotting utilities ────────────────────────────────────────────────────────

def plot_monthly_maxima(monthly_maxima, figsize=(13, 4)):
    """
    Plot monthly maximum load over time as a bar chart with a trend line.

    Parameters
    ----------
    monthly_maxima : pd.DataFrame
        Output of get_monthly_maxima().
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)

    x     = range(len(monthly_maxima))
    bars  = ax.bar(x, monthly_maxima["max_load"],
                   color="#2196F3", alpha=0.7, width=0.8)

    # Trend line
    z    = np.polyfit(list(x), monthly_maxima["max_load"].values, 1)
    p    = np.poly1d(z)
    ax.plot(x, p(list(x)), color="#FF5722", linewidth=2,
            linestyle="--", label=f"Trend ({z[0]:+.0f} MW/month)")

    # X-axis labels: Jan-YY every 6 months
    tick_positions = list(range(0, len(monthly_maxima), 6))
    tick_labels    = [
        f"{row['timestamp'].strftime('%b-%y')}"
        for _, row in monthly_maxima.reset_index().iterrows()
    ][::6]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_ylabel("Monthly Maximum Load (MW)")
    ax.set_title("Monthly Maximum Load — North Grid")
    ax.legend()
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    return fig, ax


def plot_temporal_profile(profile, figsize=(15, 9)):
    """
    Plot hour-of-day, weekday, and month distributions for peak hours
    and monthly maxima side by side.

    Parameters
    ----------
    profile : dict
        Output of peak_temporal_profile().
    figsize : tuple

    Returns
    -------
    fig
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Peak Hour & Monthly Maximum — Temporal Distributions",
                 fontsize=13, y=1.01)

    plot_specs = [
        # (row, col, data_key, title, xlabel, color)
        (0, 0, "hour_dist",      "Peak Hours by Hour of Day",  "Hour",    "#2196F3"),
        (0, 1, "weekday_dist",   "Peak Hours by Weekday",      "Weekday", "#2196F3"),
        (0, 2, "month_dist",     "Peak Hours by Month",        "Month",   "#2196F3"),
        (1, 0, "max_hour_dist",  "Monthly Max by Hour of Day", "Hour",    "#FF5722"),
        (1, 1, None,             "",                           "",        None),
        (1, 2, "max_month_dist", "Monthly Max by Month",       "Month",   "#FF5722"),
    ]

    for row, col, key, title, xlabel, color in plot_specs:
        ax = axes[row, col]
        if key is None:
            ax.set_visible(False)
            continue
        data = profile[key].fillna(0)
        ax.bar(range(len(data)), data.values, color=color, alpha=0.8,
               edgecolor="white")
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data.index, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.35)

    plt.tight_layout()
    return fig


def plot_peak_distance(distance_stats, figsize=(10, 5)):
    """
    Plot the mean and max gap (MW) between each quantile level and
    the monthly maximum, to visualise how much headroom each alpha provides.

    Parameters
    ----------
    distance_stats : pd.DataFrame
        Output of peak_distance_stats().
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Gap Between Quantile Level and Monthly Maximum", fontsize=12)

    quantile_labels = [f"Q{int(q*100)}" for q in distance_stats.index]
    x = range(len(quantile_labels))

    for ax, col_mean, col_max, ylabel, title in [
        (axes[0], "gap_mw_mean",  "gap_mw_max",
         "Gap (MW)",  "Gap in MW"),
        (axes[1], "gap_pct_mean", "gap_pct_max",
         "Gap (%)",   "Gap as % of Monthly Max"),
    ]:
        ax.bar([i - 0.2 for i in x], distance_stats[col_mean],
               width=0.35, label="Mean gap", color="#2196F3", alpha=0.8)
        ax.bar([i + 0.2 for i in x], distance_stats[col_max],
               width=0.35, label="Max gap",  color="#FF5722", alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(quantile_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.35)

    plt.tight_layout()
    return fig, axes


def plot_duration_distribution(duration_df, figsize=(10, 4)):
    """
    Plot the distribution of high-load hours in the window around
    each month's maximum, to characterise spike vs sustained behaviour.

    Parameters
    ----------
    duration_df : pd.DataFrame
        Output of peak_duration_analysis().
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Peak Duration — Are Monthly Maxima Isolated Spikes?",
                 fontsize=12)

    counts = duration_df["n_high_hours_in_window"].value_counts().sort_index()
    axes[0].bar(counts.index, counts.values, color="#9C27B0", alpha=0.8,
                edgecolor="white")
    axes[0].set_xlabel("High-load hours in ±6h window around monthly max")
    axes[0].set_ylabel("Number of months")
    axes[0].set_title("Distribution of Peak Window Width")
    axes[0].grid(axis="y", alpha=0.35)

    # Isolated vs sustained pie
    isolated = duration_df["is_isolated"].sum()
    sustained = len(duration_df) - isolated
    axes[1].pie([isolated, sustained],
                labels=[f"Isolated spike\n({isolated} months)",
                        f"Sustained peak\n({sustained} months)"],
                colors=["#9C27B0", "#4CAF50"],
                autopct="%1.0f%%", startangle=90)
    axes[1].set_title("Isolated vs Sustained Monthly Maxima")

    plt.tight_layout()
    return fig, axes
