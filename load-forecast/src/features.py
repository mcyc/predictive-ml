"""
features.py
-----------
Feature engineering pipeline for short-term electrical load forecasting.

Main entry point:
    build_features(df) -> pd.DataFrame

Assumes input df has:
    - 'date_time' column (datetime64[ns]) or a DatetimeIndex
    - 'north_clean' column: power load at native sampling frequency
    - 'Tx' column: temperature, interpolated onto the same sampling frequency

Outputs a clean hourly DataFrame ready for modelling, with columns:
    north_clean, Tx,
    hour_sin, hour_cos, weekday_sin, weekday_cos, month_sin, month_cos,
    lag_1h, lag_24h, lag_168h,
    rolling_mean_24h, rolling_mean_168h, rolling_mean_720h,
    Tx_rolling_mean_720h
"""

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum number of 10-min samples required per hour to produce a valid mean.
# 6 samples/hour * 0.67 threshold = 4 samples minimum.
MIN_SAMPLES_PER_HOUR = 4

# Maximum gap length (hours) to interpolate across. Larger gaps are left as NaN.
INTERPOLATE_LIMIT_HOURS = 3

# Minimum valid samples required for rolling means (50% of window).
# This avoids NaNs at gap boundaries while still requiring a meaningful average.
ROLLING_MIN_PERIODS = {
    "24h":  12,
    "168h": 84,
    "720h": 360,
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _mean_with_threshold(x, min_samples=MIN_SAMPLES_PER_HOUR):
    """Hourly mean only if enough sub-hourly samples are present."""
    return x.mean() if x.count() >= min_samples else np.nan


def _resample_to_hourly(df):
    """
    Resample a sub-hourly DataFrame to hourly frequency.

    - north_clean: mean with minimum sample threshold
    - Tx: mean (recovers original hourly value if linearly interpolated)
    - Tx is masked wherever north_clean is NaN to prevent spurious
      temperature values over load data gaps.
    """
    df_hourly = df.resample("h").agg(
        north_clean=("north_clean", _mean_with_threshold),
        Tx=("Tx", "mean"),
    )
    # Mask Tx wherever load is missing — gap handling driven by load data
    df_hourly.loc[df_hourly["north_clean"].isna(), "Tx"] = np.nan
    return df_hourly


def _reindex_to_continuous(df_hourly):
    """
    Reindex to a guaranteed continuous hourly DatetimeIndex.
    Any missing hours are made explicit as NaN rows.
    """
    full_range = pd.date_range(
        start=df_hourly.index.min(),
        end=df_hourly.index.max(),
        freq="h"
    )
    df_hourly = df_hourly.reindex(full_range)
    df_hourly.index.name = "date_time"
    return df_hourly


def _interpolate_small_gaps(df_hourly, limit=INTERPOLATE_LIMIT_HOURS):
    """
    Interpolate gaps up to `limit` hours using time-based linear interpolation.
    Larger gaps are left as NaN. Tx is re-masked after interpolation to stay
    consistent with north_clean.
    """
    df_hourly["north_clean"] = (
        df_hourly["north_clean"]
        .interpolate(method="time", limit=limit)
    )
    df_hourly["Tx"] = (
        df_hourly["Tx"]
        .interpolate(method="time", limit=limit)
    )
    # Re-mask Tx wherever north_clean is still NaN after interpolation
    df_hourly.loc[df_hourly["north_clean"].isna(), "Tx"] = np.nan
    return df_hourly


def _add_time_features(df_hourly):
    """
    Compute cyclical sin/cos encodings for hour, weekday, and month
    directly from the DatetimeIndex. Raw integer components are dropped.

    Encoding conventions:
        hour:    period 24, maps midnight to (sin=0, cos=1)
        weekday: period 7,  maps Monday to (sin=0, cos=1)
        month:   period 12, maps January to (sin=0, cos=1)
    """
    idx = df_hourly.index

    hour    = idx.hour
    weekday = idx.dayofweek   # 0=Monday, 6=Sunday
    month   = idx.month       # 1=January, 12=December

    df_hourly["hour_sin"]    = np.sin(2 * np.pi * hour    / 24)
    df_hourly["hour_cos"]    = np.cos(2 * np.pi * hour    / 24)
    df_hourly["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    df_hourly["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    df_hourly["month_sin"]   = np.sin(2 * np.pi * (month - 1) / 12)
    df_hourly["month_cos"]   = np.cos(2 * np.pi * (month - 1) / 12)

    return df_hourly


def _add_lag_features(df_hourly):
    """
    Add autoregressive lag features and rolling means for north_clean and Tx.

    All features are computed from shifted series (shift(1)) to ensure
    no data leakage — the current hour's load is never used as its own feature.

    Lags:
        lag_1h   : load 1 hour ago  (t-1)
        lag_24h  : load 24 hours ago (t-24, same hour yesterday)
        lag_168h : load 168 hours ago (t-168, same hour last week)

    Rolling means (computed from shift(1) to exclude current hour):
        rolling_mean_24h  : 24h mean of recent load (smoothed trend)
        rolling_mean_168h : 7-day mean of recent load (weekly baseline)
        rolling_mean_720h : 30-day mean of recent load (monthly baseline,
                            captures behavioural drift and climate anomalies)

    Climate feature:
        Tx_rolling_mean_720h : 30-day rolling mean of temperature
                               (captures monthly climate anomalies)
    """
    load = df_hourly["north_clean"]
    tx   = df_hourly["Tx"]

    df_hourly["lag_1h"]   = load.shift(1)
    df_hourly["lag_24h"]  = load.shift(24)
    df_hourly["lag_168h"] = load.shift(168)

    df_hourly["rolling_mean_24h"]  = (
        load.shift(1)
        .rolling(window=24,  min_periods=ROLLING_MIN_PERIODS["24h"]).mean()
    )
    df_hourly["rolling_mean_168h"] = (
        load.shift(1)
        .rolling(window=168, min_periods=ROLLING_MIN_PERIODS["168h"]).mean()
    )
    df_hourly["rolling_mean_720h"] = (
        load.shift(1)
        .rolling(window=720, min_periods=ROLLING_MIN_PERIODS["720h"]).mean()
    )
    df_hourly["Tx_rolling_mean_720h"] = (
        tx.shift(1)
        .rolling(window=720, min_periods=ROLLING_MIN_PERIODS["720h"]).mean()
    )

    return df_hourly


# ── Public API ────────────────────────────────────────────────────────────────

# Default feature columns produced by build_features()
FEATURES = [
    "Tx",
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_24h", "rolling_mean_168h", "rolling_mean_720h",
    "Tx_rolling_mean_720h",
]

TARGET = "north_clean"


def build_features(
    df,
    datetime_col="date_time",
    interpolate_limit=INTERPOLATE_LIMIT_HOURS,
    min_samples_per_hour=MIN_SAMPLES_PER_HOUR,
    drop_na=True,
):
    """
    Full feature engineering pipeline for hourly load forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe with columns 'date_time', 'north_clean', 'Tx'.
        'date_time' can be a column or the index.
    datetime_col : str
        Name of the datetime column if not already the index.
    interpolate_limit : int
        Maximum gap size (hours) to interpolate. Default: 3.
    min_samples_per_hour : int
        Minimum sub-hourly samples required per hour for a valid mean. Default: 4.
    drop_na : bool
        If True, drop all rows with any NaN after feature construction.
        Set to False to inspect NaN distribution before dropping. Default: True.

    Returns
    -------
    pd.DataFrame
        Hourly DataFrame with TARGET + FEATURES columns, DatetimeIndex,
        sorted chronologically. NaN rows dropped if drop_na=True.
    """
    # ── Ensure DatetimeIndex ──────────────────────────────────────────────────
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(datetime_col)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # ── Resample to hourly ────────────────────────────────────────────────────
    df_hourly = _resample_to_hourly(df)

    # ── Reindex to continuous hourly range ────────────────────────────────────
    df_hourly = _reindex_to_continuous(df_hourly)

    # ── Interpolate small gaps ────────────────────────────────────────────────
    df_hourly = _interpolate_small_gaps(df_hourly, limit=interpolate_limit)

    # ── Time features ─────────────────────────────────────────────────────────
    df_hourly = _add_time_features(df_hourly)

    # ── Lag and rolling features ──────────────────────────────────────────────
    df_hourly = _add_lag_features(df_hourly)

    # ── Drop NaN rows ─────────────────────────────────────────────────────────
    if drop_na:
        df_hourly = df_hourly.dropna().copy()

    return df_hourly


def add_tx_leads(df_hourly, leads=(1, 2, 3)):
    """
    Add future temperature values as features to mock a weather forecast.

    These are Tx lead features (t+1, t+2, t+3) — valid only for experiments
    where forecast temperature is assumed available. Do NOT use in a real
    deployment unless actual forecast Tx is provided.

    Parameters
    ----------
    df_hourly : pd.DataFrame
        Output of build_features() with drop_na=False, so leads can be computed
        before NaN rows are dropped.
    leads : tuple of int
        Lead hours to add. Default: (1, 2, 3).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional 'Tx_lead_{n}h' columns.
    """
    for n in leads:
        df_hourly[f"Tx_lead_{n}h"] = df_hourly["Tx"].shift(-n)
    return df_hourly


def get_gap_distribution(series):
    """
    Return a summary of consecutive NaN gap lengths in a Series.
    Useful for inspecting missing data before choosing interpolation limits.

    Parameters
    ----------
    series : pd.Series

    Returns
    -------
    pd.Series
        Value counts of gap lengths (index = gap length in hours,
        values = number of gaps of that length).
    """
    gap_lengths = []
    count = 0
    for val in series.isna():
        if val:
            count += 1
        elif count > 0:
            gap_lengths.append(count)
            count = 0
    if count > 0:
        gap_lengths.append(count)

    s = pd.Series(gap_lengths)
    if s.empty:
        print("No gaps found.")
        return s

    summary = s.value_counts().sort_index().rename("n_gaps").to_frame()
    print(summary)
    print(f"\nTotal gaps:  {len(s)}")
    print(f"Max gap:     {s.max():.0f}h")
    print(f"Median gap:  {s.median():.0f}h")
    print(f"Gaps <= 3h:  {(s <= 3).sum()} ({(s <= 3).mean()*100:.1f}%)")
    print(f"Gaps >  3h:  {(s > 3).sum()} ({(s > 3).mean()*100:.1f}%)")
    return s
