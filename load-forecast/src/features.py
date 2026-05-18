"""
features.py
-----------
Feature engineering pipeline for short-term electrical load forecasting.

Main entry point:
    build_features(df, freq=None) -> pd.DataFrame

Supports arbitrary resampling frequencies via the `freq` argument, which
accepts any offset string understood by pd.resample() and pd.date_range()
(e.g. '10min', 'h', '30min', '15min').

When freq=None (default), the native sampling frequency is detected
automatically from the index via detect_freq(). Pass freq explicitly
to override detection or to handle irregular inputs.

All lag periods and rolling windows are defined in calendar time (hours)
and converted to step counts automatically based on `freq`. Feature column
names are always human-readable time labels (e.g. 'lag_1h', 'rolling_mean_24h')
regardless of the underlying sampling frequency.

Assumes input df has:
    - 'date_time' column (datetime64[ns]) or a DatetimeIndex
    - 'north_clean' column: power load at native sampling frequency
    - 'Tx' column: temperature at the same sampling frequency

Default output columns (at any freq):
    north_clean, Tx,
    hour_sin, hour_cos, weekday_sin, weekday_cos, month_sin, month_cos,
    lag_1h, lag_24h, lag_168h,
    rolling_mean_24h, rolling_mean_168h, rolling_mean_720h,
    Tx_rolling_mean_720h
"""

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

# Fallback sampling frequency used only when auto-detection fails and
# freq is not explicitly provided. Override by passing freq= to build_features.
FALLBACK_FREQ = "10min"

# Minimum fraction of expected samples per resampled period required to
# produce a valid aggregate. Applied to load (north_clean) only.
# e.g. at 10min→1h: 6 expected, 0.67 threshold = 4 samples minimum.
MIN_SAMPLE_FRACTION = 0.67

# Maximum gap duration (hours) to interpolate across. Larger gaps are left
# as NaN. Converted to steps at runtime based on freq.
INTERPOLATE_LIMIT_HOURS = 3

# Minimum fraction of window required for rolling means (50%).
# Avoids NaNs at series boundaries while still requiring a meaningful average.
ROLLING_MIN_PERIOD_FRACTION = 0.50

# Lag periods defined in hours — converted to steps at runtime.
LAG_HOURS = [1, 24, 168]

# Rolling window sizes defined in hours — converted to steps at runtime.
ROLLING_HOURS = [24, 168, 720]


# ── Frequency utilities ───────────────────────────────────────────────────────

def _steps_per_hour(freq):
    """
    Return the number of resampled periods per hour for a given freq string.

    Parameters
    ----------
    freq : str
        Pandas offset string, e.g. '10min', 'h', '30min', '15min'.

    Returns
    -------
    float
        Number of periods per hour. Will be 1.0 for 'h', 6.0 for '10min', etc.

    Examples
    --------
    >>> _steps_per_hour('10min')
    6.0
    >>> _steps_per_hour('h')
    1.0
    >>> _steps_per_hour('30min')
    2.0
    """
    one_hour = pd.Timedelta(hours=1)
    period   = pd.tseries.frequencies.to_offset(freq).nanos
    return one_hour.value / period


def _hours_to_steps(hours, freq):
    """
    Convert a duration in hours to an integer number of steps at `freq`.

    Parameters
    ----------
    hours : int or float
    freq  : str

    Returns
    -------
    int
    """
    return int(round(hours * _steps_per_hour(freq)))


def _min_samples_per_period(freq, target_freq="h"):
    """
    Compute the minimum number of input samples required per resampled period.

    Uses MIN_SAMPLE_FRACTION of the expected samples per period.

    Parameters
    ----------
    freq        : str  Input (native) frequency, e.g. '10min'.
    target_freq : str  Target (resampled) frequency, e.g. 'h'.

    Returns
    -------
    int  Minimum samples required (at least 1).
    """
    native_period   = pd.tseries.frequencies.to_offset(freq).nanos
    target_period   = pd.tseries.frequencies.to_offset(target_freq).nanos
    expected        = target_period / native_period
    return max(1, int(np.floor(expected * MIN_SAMPLE_FRACTION)))


def detect_freq(df, return_freq=False, verbose=True):
    """
    Detect the native sampling frequency of a DataFrame's DatetimeIndex.

    Strategy:
        1. Compute the median gap between consecutive timestamps. Using the
           median rather than the mode or minimum makes the estimate robust
           to isolated duplicates or missing rows.
        2. Round to the nearest standard pandas offset from a candidate list.
        3. Validate regularity: check what fraction of actual gaps match the
           detected period (within a small tolerance). Warn if below threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex. Must have at least 2 rows.
    return_freq : bool
        If True, return the detected freq string in addition to printing it.
        Default: False (print only — no return value change to build_features
        callers that don't expect a return value from detect_freq directly).
    verbose : bool
        If True, print detection results and regularity statistics.
        Default: True.

    Returns
    -------
    str or None
        Detected freq string if return_freq=True, else None.
        Returns None if detection fails (irregular or insufficient data).

    Raises
    ------
    ValueError
        If df has fewer than 2 rows (cannot compute gaps).

    Examples
    --------
    >>> detect_freq(df_raw)
    Detected frequency : 10min
    Regularity         : 99.8% of gaps match detected period
    Grid coverage      : 279,991 / 280,800 expected periods present

    >>> freq = detect_freq(df_raw, return_freq=True, verbose=False)
    >>> freq
    '10min'
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex. "
                        "Call df.set_index('date_time') first.")
    if len(df) < 2:
        raise ValueError("df must have at least 2 rows to detect frequency.")

    # ── Candidate standard offsets to snap to ────────────────────────────────
    # Ordered from finest to coarsest. Detection snaps the median gap to the
    # closest candidate. Add entries here for non-standard frequencies.
    candidates = [
        ("1min",  pd.Timedelta(minutes=1)),
        ("5min",  pd.Timedelta(minutes=5)),
        ("10min", pd.Timedelta(minutes=10)),
        ("15min", pd.Timedelta(minutes=15)),
        ("30min", pd.Timedelta(minutes=30)),
        ("h",     pd.Timedelta(hours=1)),
        ("2h",    pd.Timedelta(hours=2)),
        ("3h",    pd.Timedelta(hours=3)),
        ("6h",    pd.Timedelta(hours=6)),
        ("12h",   pd.Timedelta(hours=12)),
        ("D",     pd.Timedelta(days=1)),
    ]

    # ── Compute median gap ────────────────────────────────────────────────────
    gaps        = pd.Series(df.index).diff().dropna()
    median_gap  = gaps.median()

    if pd.isna(median_gap) or median_gap <= pd.Timedelta(0):
        if verbose:
            print("WARNING: Could not detect frequency — "
                  "median gap is zero or NaN (duplicate timestamps?).")
        return None if return_freq else None

    # ── Snap to nearest candidate ─────────────────────────────────────────────
    diffs        = [(name, abs(median_gap - td)) for name, td in candidates]
    best_name, _ = min(diffs, key=lambda x: x[1])
    best_td      = dict(candidates)[best_name]

    # ── Regularity check ──────────────────────────────────────────────────────
    tolerance    = best_td * 0.01   # 1% of period
    n_regular    = (abs(gaps - best_td) <= tolerance).sum()
    regularity   = n_regular / len(gaps)

    # Expected number of periods in the full date range
    full_range   = pd.date_range(df.index.min(), df.index.max(), freq=best_name)
    n_expected   = len(full_range)
    n_actual     = len(df)
    coverage_pct = n_actual / n_expected * 100

    if verbose:
        print(f"Detected frequency : {best_name}")
        print(f"Median gap         : {median_gap}")
        print(f"Regularity         : {regularity*100:.1f}% of gaps match "
              f"detected period (tolerance ±{tolerance})")
        print(f"Grid coverage      : {n_actual:,} / {n_expected:,} expected "
              f"periods present ({coverage_pct:.1f}%)")
        if regularity < 0.90:
            print(f"WARNING: Low regularity ({regularity*100:.1f}%). "
                  "The index may be irregular. Consider passing freq= "
                  "explicitly to build_features().")

    return best_name if return_freq else None




def _make_threshold_agg(min_samples):
    """
    Return a named aggregation function that computes the mean only if
    at least `min_samples` non-NaN values are present in the group.
    """
    def _agg(x):
        return x.mean() if x.count() >= min_samples else np.nan
    _agg.__name__ = "mean_with_threshold"
    return _agg


def _resample(df, freq, target_freq):
    """
    Resample df from its native `freq` to `target_freq`.

    - north_clean: mean with minimum sample threshold (load quality gate)
    - Tx: simple mean (recovers original value if already interpolated)
    - Tx is masked wherever north_clean is NaN (gap handling driven by load)

    If freq == target_freq, resampling is skipped and the original df is
    returned as-is (pass-through for native-resolution workflows).
    """
    if pd.tseries.frequencies.to_offset(freq) == \
       pd.tseries.frequencies.to_offset(target_freq):
        # Already at target resolution — just ensure column presence
        df_out = df[["north_clean", "Tx"]].copy()
        df_out.loc[df_out["north_clean"].isna(), "Tx"] = np.nan
        return df_out

    min_samples = _min_samples_per_period(freq, target_freq)
    agg_fn      = _make_threshold_agg(min_samples)

    df_out = df.resample(target_freq).agg(
        north_clean=("north_clean", agg_fn),
        Tx=("Tx", "mean"),
    )
    df_out.loc[df_out["north_clean"].isna(), "Tx"] = np.nan
    return df_out


def _reindex_to_continuous(df, freq):
    """
    Reindex df to a guaranteed continuous DatetimeIndex at `freq`.
    Any missing periods are made explicit as NaN rows.
    """
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq,
    )
    df = df.reindex(full_range)
    df.index.name = "date_time"
    return df


def _interpolate_small_gaps(df, limit_steps):
    """
    Interpolate gaps up to `limit_steps` periods using time-based linear
    interpolation. Larger gaps are left as NaN.

    Parameters
    ----------
    df          : pd.DataFrame  Must contain 'north_clean' and 'Tx'.
    limit_steps : int           Maximum number of consecutive NaN steps to fill.
    """
    df["north_clean"] = df["north_clean"].interpolate(
        method="time", limit=limit_steps
    )
    df["Tx"] = df["Tx"].interpolate(
        method="time", limit=limit_steps
    )
    # Re-mask Tx wherever north_clean is still NaN
    df.loc[df["north_clean"].isna(), "Tx"] = np.nan
    return df


def _add_time_features(df):
    """
    Compute cyclical sin/cos encodings for hour, minute-within-hour,
    weekday, and month from the DatetimeIndex.

    For sub-hourly frequencies, `minute_sin` / `minute_cos` encode the
    position within the hour (useful for 10-min and 15-min resolution).
    For hourly and coarser frequencies, minute encodings are omitted
    (all zeros — no information added).

    Encoding conventions:
        hour:    period 24
        minute:  period 60 (sub-hourly only)
        weekday: period 7
        month:   period 12
    """
    idx = df.index

    hour    = idx.hour
    minute  = idx.minute
    weekday = idx.dayofweek
    month   = idx.month

    df["hour_sin"]    = np.sin(2 * np.pi * hour    / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * hour    / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    df["month_sin"]   = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"]   = np.cos(2 * np.pi * (month - 1) / 12)

    # Sub-hourly: add minute-within-hour encoding
    if minute.max() > 0:
        df["minute_sin"] = np.sin(2 * np.pi * minute / 60)
        df["minute_cos"] = np.cos(2 * np.pi * minute / 60)

    return df


def _add_lag_features(df, freq):
    """
    Add autoregressive lag features and rolling means for north_clean and Tx.

    All lags and windows are defined in hours and converted to step counts
    based on `freq`. Feature names always use hour labels for readability
    and cross-frequency consistency.

    No data leakage: all features are computed from shift(1) so the current
    period's value is never used as its own feature.

    Lags (hours → steps at freq):
        lag_1h   : 1h  (6 steps at 10min, 1 step at 1h)
        lag_24h  : 24h
        lag_168h : 168h (1 week)

    Rolling means (hours → steps at freq):
        rolling_mean_24h  : 24h  smoothed recent load
        rolling_mean_168h : 168h weekly baseline
        rolling_mean_720h : 720h monthly baseline

    Climate feature:
        Tx_rolling_mean_720h : 30-day rolling mean of temperature
    """
    sph  = _steps_per_hour(freq)   # steps per hour
    load = df["north_clean"]
    tx   = df["Tx"]

    # ── Lags ──────────────────────────────────────────────────────────────────
    for lag_h in LAG_HOURS:
        steps           = _hours_to_steps(lag_h, freq)
        df[f"lag_{lag_h}h"] = load.shift(steps)

    # ── Rolling means ─────────────────────────────────────────────────────────
    for window_h in ROLLING_HOURS:
        steps       = _hours_to_steps(window_h, freq)
        min_periods = max(1, int(steps * ROLLING_MIN_PERIOD_FRACTION))
        df[f"rolling_mean_{window_h}h"] = (
            load.shift(1)
                .rolling(window=steps, min_periods=min_periods)
                .mean()
        )

    # ── Temperature rolling mean ───────────────────────────────────────────────
    tx_window_h   = 720
    tx_steps      = _hours_to_steps(tx_window_h, freq)
    tx_min_periods = max(1, int(tx_steps * ROLLING_MIN_PERIOD_FRACTION))
    df["Tx_rolling_mean_720h"] = (
        tx.shift(1)
          .rolling(window=tx_steps, min_periods=tx_min_periods)
          .mean()
    )

    return df


# ── Public API ────────────────────────────────────────────────────────────────

# Default feature columns produced by build_features().
# These names are freq-agnostic — step counts are internal implementation details.
FEATURES = [
    "Tx",
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_24h", "rolling_mean_168h", "rolling_mean_720h",
    "Tx_rolling_mean_720h",
]

# Sub-hourly feature set — adds minute encoding for 10min/15min/30min resolutions
FEATURES_SUBHOURLY = FEATURES + ["minute_sin", "minute_cos"]

TARGET = "north_clean"


def get_features(freq=FALLBACK_FREQ):
    """
    Return the appropriate feature list for a given sampling frequency.

    Uses FEATURES_SUBHOURLY (includes minute_sin/cos) for frequencies
    finer than hourly, FEATURES otherwise.

    Parameters
    ----------
    freq : str  Sampling frequency, e.g. '10min', 'h'.

    Returns
    -------
    list of str
    """
    sph = _steps_per_hour(freq)
    return FEATURES_SUBHOURLY if sph > 1 else FEATURES


def build_features(
    df,
    freq=None,
    target_freq=None,
    datetime_col="date_time",
    interpolate_limit_hours=INTERPOLATE_LIMIT_HOURS,
    drop_na=True,
):
    """
    Full feature engineering pipeline for load forecasting at any resolution.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame with columns 'date_time' (or DatetimeIndex),
        'north_clean' (load), and 'Tx' (temperature).
    freq : str or None
        Native sampling frequency of the input data. When None (default),
        the frequency is detected automatically from the index via
        detect_freq(). Pass explicitly to override detection or to handle
        irregular inputs. Examples: '10min', 'h', '30min', '15min'.
    target_freq : str or None
        If provided, resample from `freq` to `target_freq` before feature
        engineering. Use this to downsample (e.g. '10min' → 'h') while
        retaining generalised step-count logic.
        If None, features are built at the native (or detected) resolution.
    datetime_col : str
        Name of the datetime column if not already the index. Default: 'date_time'.
    interpolate_limit_hours : int or float
        Maximum gap size in hours to interpolate. Converted to steps internally.
        Default: 3 (hours).
    drop_na : bool
        If True, drop all rows with any NaN after feature construction.
        Set to False to inspect NaN distribution or to add lead features
        (add_tx_leads) before dropping. Default: True.

    Returns
    -------
    pd.DataFrame
        DataFrame at native (or target_freq) resolution with TARGET +
        get_features(freq) columns, DatetimeIndex, sorted chronologically.
        NaN rows dropped if drop_na=True.

    Examples
    --------
    # Auto-detect frequency (recommended default)
    df = build_features(df_raw)

    # Explicit native 10-minute resolution
    df = build_features(df_raw, freq='10min')

    # Downsample to hourly for compatibility with notebook 01–03 experiments
    df = build_features(df_raw, target_freq='h')

    # Build from already-hourly input
    df = build_features(df_raw, freq='h')
    """
    # ── Ensure DatetimeIndex ──────────────────────────────────────────────────
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(datetime_col)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # ── Detect or validate freq ───────────────────────────────────────────────
    if freq is None:
        freq = detect_freq(df, return_freq=True, verbose=True)
        if freq is None:
            freq = FALLBACK_FREQ
            print(f"WARNING: Frequency detection failed. "
                  f"Falling back to '{FALLBACK_FREQ}'. "
                  f"Pass freq= explicitly to override.")

    # ── Determine working frequency ───────────────────────────────────────────
    working_freq = target_freq if target_freq is not None else freq

    # ── Resample if target_freq requested ─────────────────────────────────────
    if target_freq is not None:
        df = _resample(df, freq, target_freq)
    else:
        df = df[["north_clean", "Tx"]].copy()
        df.loc[df["north_clean"].isna(), "Tx"] = np.nan

    # ── Reindex to continuous grid ────────────────────────────────────────────
    df = _reindex_to_continuous(df, working_freq)

    # ── Interpolate small gaps ────────────────────────────────────────────────
    limit_steps = _hours_to_steps(interpolate_limit_hours, working_freq)
    df = _interpolate_small_gaps(df, limit_steps)

    # ── Time features ─────────────────────────────────────────────────────────
    df = _add_time_features(df)

    # ── Lag and rolling features ──────────────────────────────────────────────
    df = _add_lag_features(df, working_freq)

    # ── Drop NaN rows ─────────────────────────────────────────────────────────
    if drop_na:
        df = df.dropna().copy()

    return df


def add_tx_leads(df, leads, freq=None):
    """
    Add future temperature values as oracle features to mock a weather forecast.

    Lead periods are specified in hours and converted to steps based on `freq`.
    Column names always use hour labels for cross-frequency consistency.

    Only valid for experiments where future temperature is assumed known.
    Do NOT use in production unless actual NWP forecast Tx is supplied.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_features() with drop_na=False.
    leads : iterable of int or float
        Lead durations in hours. e.g. [1, 2, 3] adds Tx_lead_1h, Tx_lead_2h,
        Tx_lead_3h regardless of the underlying sampling frequency.
    freq : str or None
        Sampling frequency of df. When None, detected automatically.
        Default: None.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional 'Tx_lead_{n}h' columns.
    """
    if freq is None:
        freq = detect_freq(df, return_freq=True, verbose=False) or FALLBACK_FREQ
    for lead_h in leads:
        steps = _hours_to_steps(lead_h, freq)
        df[f"Tx_lead_{lead_h}h"] = df["Tx"].shift(-steps)
    return df


def get_gap_distribution(series, freq=None):
    """
    Return a summary of consecutive NaN gap lengths in a Series.

    Gap lengths are reported in both steps and hours for readability.
    Useful for inspecting missing data before choosing interpolation limits.

    Parameters
    ----------
    series : pd.Series with DatetimeIndex
    freq   : str or None
        Sampling frequency. When None, detected automatically from series index.

    Returns
    -------
    pd.Series  Value counts of gap lengths in steps.
    """
    if freq is None:
        _tmp = series.to_frame() if isinstance(series, pd.Series) else series
        freq = detect_freq(_tmp, return_freq=True, verbose=False) or FALLBACK_FREQ
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

    sph     = _steps_per_hour(freq)
    summary = s.value_counts().sort_index().rename("n_gaps").to_frame()
    summary["hours"] = (summary.index / sph).round(1)
    print(summary.to_string())
    print(f"\nTotal gaps:      {len(s)}")
    print(f"Max gap:         {s.max()} steps ({s.max()/sph:.1f}h)")
    print(f"Median gap:      {s.median():.0f} steps ({s.median()/sph:.1f}h)")
    limit_steps = _hours_to_steps(INTERPOLATE_LIMIT_HOURS, freq)
    print(f"Gaps <= {INTERPOLATE_LIMIT_HOURS}h:     "
          f"{(s <= limit_steps).sum()} ({(s <= limit_steps).mean()*100:.1f}%)")
    print(f"Gaps >  {INTERPOLATE_LIMIT_HOURS}h:     "
          f"{(s > limit_steps).sum()} ({(s > limit_steps).mean()*100:.1f}%)")
    return s