"""
data.py
-------
Data loading and train/test splitting utilities for load forecasting.

Main functions:
    load_data(path, datetime_col, load_col, temp_col)
        Load raw data from CSV or parquet, detect native sampling frequency,
        and return a clean DataFrame ready for build_features().

    resample_monthly(df, target, agg)
        Resample a feature-engineered DataFrame to monthly frequency.
        Useful for the monthly maximum forecaster in notebook 05+.

    split_chronological(df, features, target, test_days)
        Chronological train/test split by reserving the last N calendar days
        as the test set. Freq-agnostic — works at any sampling resolution.

    split_summary(train, test)
        Print a resolution-aware summary of a train/test split.
"""

import pandas as pd


# ── Public API ────────────────────────────────────────────────────────────────

def load_data(
    path,
    datetime_col="date_time",
    load_col="north_clean",
    temp_col="Tx",
    dtype=None,
):
    """
    Load raw load and temperature data from a CSV or parquet file.

    Reads the file, parses the datetime index, renames columns to the
    standard names expected by build_features() if needed, and returns
    a DataFrame sorted chronologically. Calls detect_freq() from
    features.py to identify and report the native sampling frequency.

    Parameters
    ----------
    path : str
        Path to the data file. Supports .parquet and .csv.
    datetime_col : str
        Name of the datetime column in the file. Default: 'date_time'.
    load_col : str
        Name of the load column. Default: 'north_clean'.
    temp_col : str
        Name of the temperature column. Default: 'Tx'.
    dtype : dict or None
        Optional dtype overrides passed to pd.read_csv(). Ignored for parquet.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and columns ['north_clean', 'Tx'],
        sorted chronologically.
    """
    # ── Load file ─────────────────────────────────────────────────────────────
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=[datetime_col], dtype=dtype)

    # ── Rename to standard column names if needed ─────────────────────────────
    rename_map = {}
    if load_col != "north_clean":
        rename_map[load_col] = "north_clean"
    if temp_col != "Tx":
        rename_map[temp_col] = "Tx"
    if datetime_col != "date_time":
        rename_map[datetime_col] = "date_time"
    if rename_map:
        df = df.rename(columns=rename_map)

    # ── Set datetime index ────────────────────────────────────────────────────
    if "date_time" in df.columns:
        df = df.set_index("date_time")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # ── Keep only required columns ────────────────────────────────────────────
    df = df[["north_clean", "Tx"]].copy()

    # ── Detect native frequency ───────────────────────────────────────────────
    # Import here to avoid circular imports — features.py is a sibling module.
    try:
        from features import detect_freq
        native_freq = detect_freq(df, return_freq=True, verbose=False)
    except ImportError:
        native_freq = None

    # ── Print summary ─────────────────────────────────────────────────────────
    n_load_nan = df["north_clean"].isna().sum()
    n_tx_nan   = df["Tx"].isna().sum()
    date_range_days = (df.index.max() - df.index.min()).days

    print(f"Loaded: {path}")
    print(f"  Shape:           {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Date range:      {df.index.min()} → {df.index.max()} "
          f"({date_range_days:,} days)")
    if native_freq is not None:
        print(f"  Native freq:     {native_freq}")
    print(f"  NaNs:            north_clean={n_load_nan:,}  Tx={n_tx_nan:,}")

    return df


def resample_monthly(df, target="north_clean", agg=None):
    """
    Resample a feature-engineered DataFrame to monthly frequency.

    Computes a configurable set of monthly aggregates useful for the
    monthly maximum forecaster in notebooks 05–06.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame with DatetimeIndex at any resolution.
    target : str
        Load column name. Default: 'north_clean'.
    agg : dict or None
        Aggregation spec passed to df.resample('MS').agg(). If None, uses
        the default set: max, mean, std, and count of the target column.

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame indexed by month-start timestamps with columns
        defined by `agg`. Default columns:
            max_load  : monthly maximum load (the billing-relevant quantity)
            mean_load : monthly mean load
            std_load  : monthly standard deviation of load
            n_periods : number of valid (non-NaN) periods in the month
    """
    if agg is None:
        # Aggregate target column only, then rename for clarity.
        # Use explicit lambdas with skipna=True to ensure max and mean are
        # computed consistently — avoiding NaN-driven false failures where
        # a vanilla "max" returns NaN for an all-NaN month while "mean"
        # does not (or vice versa), breaking the max >= mean invariant.
        monthly = df[target].resample("MS").agg(
            max_load=lambda s: s.max(skipna=True),
            mean_load=lambda s: s.mean(skipna=True),
            std_load=lambda s: s.std(skipna=True),
            n_periods=lambda s: s.count(),
        )
    else:
        monthly = df.resample("MS").agg(agg)
        # Flatten MultiIndex columns if produced by a custom agg spec
        if isinstance(monthly.columns, pd.MultiIndex):
            monthly.columns = ["_".join(filter(None, col)) for col in monthly.columns]

    # Warn about fully-empty months but keep them so the MS freq is preserved.
    # Dropping rows would create a gap that pandas 2.x refuses to re-stamp as MS.
    # Callers should filter with monthly[monthly["n_periods"] > 0] before use.
    n_empty = (monthly["n_periods"] == 0).sum() if "n_periods" in monthly.columns else 0
    if n_empty:
        empty_dates = monthly.index[monthly["n_periods"] == 0].tolist()
        print(f"  Warning: {n_empty} month(s) with no valid readings "
              f"(kept for index continuity): {[d.date() for d in empty_dates]}")

    print(f"Resampled to monthly: {len(monthly)} months "
          f"({monthly.index.min().date()} → {monthly.index.max().date()})")
    return monthly


def split_chronological(df, features, target, test_days=365):
    """
    Split a feature-engineered DataFrame into train and test sets
    using a chronological cutoff.

    The test set is the last `test_days` calendar days of the series.
    The split is calendar-based (pd.DateOffset) and therefore
    freq-agnostic — it works correctly at 10-minute, hourly, or any
    other sampling resolution without adjustment.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_features() with DatetimeIndex.
    features : list of str
        Feature column names. Use get_features(freq) from features.py.
    target : str
        Target column name. Use TARGET from features.py.
    test_days : int
        Number of calendar days to reserve for the test set. Default: 365.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
    """
    split_date = df.index.max() - pd.DateOffset(days=test_days)

    train = df[df.index <= split_date]
    test  = df[df.index >  split_date]

    X_train = train[features]
    y_train = train[target]
    X_test  = test[features]
    y_test  = test[target]

    split_summary(train, test)

    return X_train, X_test, y_train, y_test


def split_summary(train, test):
    """
    Print a resolution-aware summary of a train/test split.

    Reports row counts alongside approximate calendar equivalents
    (days and hours) so the numbers are interpretable at any sampling
    frequency without mental arithmetic.

    Parameters
    ----------
    train : pd.DataFrame  Training split with DatetimeIndex.
    test  : pd.DataFrame  Test split with DatetimeIndex.
    """
    total = len(train) + len(test)

    # Infer approximate period duration from the index for calendar conversion
    def _approx_period_hours(df):
        """Estimate hours per row from the actual index gaps."""
        if len(df) < 2:
            return None
        median_gap = pd.Series(df.index).diff().dropna().median()
        return median_gap.total_seconds() / 3600

    period_h = _approx_period_hours(train)

    def _calendar_str(n_rows, period_h):
        if period_h is None:
            return ""
        total_h   = n_rows * period_h
        total_d   = total_h / 24
        if total_d >= 2:
            return f" ≈ {total_d:,.0f} days"
        return f" ≈ {total_h:,.0f} hours"

    train_cal = _calendar_str(len(train), period_h)
    test_cal  = _calendar_str(len(test),  period_h)

    print("\n── Train/Test Split ──")
    print(f"  Split date : {train.index.max().date()}")
    if period_h is not None:
        freq_label = f"{period_h*60:.0f}min" if period_h < 1 else f"{period_h:.0f}h"
        print(f"  Resolution : ~{freq_label} per row")
    print(f"  Train      : {len(train):,} rows{train_cal}  "
          f"({train.index.min().date()} → {train.index.max().date()})  "
          f"[{len(train)/total*100:.1f}%]")
    print(f"  Test       : {len(test):,} rows{test_cal}  "
          f"({test.index.min().date()} → {test.index.max().date()})  "
          f"[{len(test)/total*100:.1f}%]")