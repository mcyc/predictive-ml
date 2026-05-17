"""
data.py
-------
Data loading and train/test splitting utilities for load forecasting.

Main functions:
    load_data(path, datetime_col, load_col, temp_col)
        Load raw CSV data and return a clean DataFrame ready for build_features().

    split_chronological(df, test_days, features, target)
        Chronological train/test split by reserving the last N days as test.

    split_summary(train, test)
        Print a summary of the split for quick sanity checking.
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
    Load raw load and temperature data from a CSV file.

    Reads the file, parses the datetime column, renames columns to the
    standard names expected by build_features() if needed, and returns
    a DataFrame sorted by datetime with datetime as the index.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    datetime_col : str
        Name of the datetime column in the CSV. Default: 'date_time'.
    load_col : str
        Name of the load column in the CSV. Default: 'north_clean'.
    temp_col : str
        Name of the temperature column in the CSV. Default: 'Tx'.
    dtype : dict or None
        Optional dtype overrides passed to pd.read_csv().

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and columns 'north_clean', 'Tx',
        sorted chronologically.
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=[datetime_col], dtype=dtype)

    # Rename to standard column names if needed
    rename_map = {}
    if load_col != "north_clean":
        rename_map[load_col] = "north_clean"
    if temp_col != "Tx":
        rename_map[temp_col] = "Tx"
    if datetime_col != "date_time":
        rename_map[datetime_col] = "date_time"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Set datetime index
    if "date_time" in df.columns:
        df = df.set_index("date_time")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Keep only required columns
    df = df[["north_clean", "Tx"]].copy()

    print(f"Loaded: {path}")
    print(f"  Shape:      {df.shape}")
    print(f"  Date range: {df.index.min()} → {df.index.max()}")
    print(f"  NaNs:       north_clean={df['north_clean'].isna().sum()}, "
          f"Tx={df['Tx'].isna().sum()}")

    return df


def split_chronological(df, features, target, test_days=365):
    """
    Split a feature-engineered DataFrame into train and test sets
    using a chronological cutoff.

    The test set is the last `test_days` days of the series.
    The split is inclusive on the train side and exclusive on the test side
    at the cutoff date, so there is no overlap.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_features() with DatetimeIndex.
    features : list of str
        Feature column names. Use FEATURES from features.py.
    target : str
        Target column name. Use TARGET from features.py.
    test_days : int
        Number of days to reserve for the test set. Default: 365.

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
    Print a concise summary of a train/test split.

    Parameters
    ----------
    train : pd.DataFrame
    test  : pd.DataFrame
    """
    total = len(train) + len(test)
    print("\n── Train/Test Split ──")
    print(f"  Split date: {train.index.max().date()}")
    print(f"  Train:      {len(train):,} rows  "
          f"({train.index.min().date()} → {train.index.max().date()})  "
          f"[{len(train)/total*100:.1f}%]")
    print(f"  Test:       {len(test):,} rows  "
          f"({test.index.min().date()} → {test.index.max().date()})  "
          f"[{len(test)/total*100:.1f}%]")