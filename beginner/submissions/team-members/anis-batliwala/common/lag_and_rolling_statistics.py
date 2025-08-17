def engineer_lag_and_rolling(dataset, zone, lags=[1], rolling_windows=[3], 
                             rolling_stats=['mean', 'std', 'median'], 
                             fill_value=None, fill_method='ffill'):
    """
    Engineer lag features and rolling statistics for a single zone.

    Parameters:
    - dataset: pd.DataFrame with datetime index
    - zone: str, the target zone column to create features for
    - lags: list of ints, lag periods to create
    - rolling_windows: list of ints, rolling window sizes
    - rolling_stats: list of stats to compute ('mean', 'std', 'median', etc.)
    - fill_value: scalar to fill NaNs
    - fill_method: method to fill NaNs if fill_value is None ('ffill' or None)

    Returns:
    - pd.DataFrame with new lag and rolling features
    """
    valid_stats = {'mean', 'std', 'median', 'min', 'max', 'sum', 'var'}
    invalid_stats = set(rolling_stats) - valid_stats
    if invalid_stats:
        raise ValueError(f"Invalid rolling_stat(s): {invalid_stats}. Choose from {valid_stats}.")

    df = dataset.copy()
    zone_data = df[zone]

    # Lag features
    for lag in lags:
        df[f'{zone}_lag{lag}'] = zone_data.shift(lag)

    # Rolling features
    for window in rolling_windows:
        roll_obj = zone_data.rolling(window=window)
        for stat in rolling_stats:
            df[f'{zone}_roll_{stat}{window}'] = getattr(roll_obj, stat)()

    # Fill NaNs
    if fill_value is not None:
        df = df.fillna(fill_value)
    elif fill_method == 'ffill':
        df = df.ffill()
    elif fill_method == 'bfill':
        df = df.bfill()

    return df
