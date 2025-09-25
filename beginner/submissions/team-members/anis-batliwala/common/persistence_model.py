import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def persistence_baseline_performance(test_dataset, target_col, lag=1):
    """
    Evaluate a naïve persistence (last known value) model for time series.
    
    Parameters:
    - test_dataset: pd.DataFrame with datetime index.
    - target_col: str, target variable name.
    - lag: int, how many steps back to use as prediction.
    
    Returns:
    - metrics_df: DataFrame with RMSE, MAE, and R².
    
    Raises:
    - ValueError: If input checks fail.
    """

    # Input validation
    if len(test_dataset) == 0:
        raise ValueError("Input DataFrame is empty.")
    if target_col not in test_dataset.columns:
        raise ValueError(f"Column '{target_col}' not found in the dataset.")
    if not isinstance(test_dataset.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex.")
    if lag <= 0 or lag >= len(test_dataset):
        raise ValueError("Lag must be between 1 and len(dataset)-1.")
    
    # Shift the target to create naive predictions
    y_true = test_dataset[target_col].copy()
    y_pred = test_dataset[target_col].shift(lag)

    # Drop the first `lag` rows where prediction is NaN
    valid_idx = y_pred.dropna().index
    y_true = y_true.loc[valid_idx]
    y_pred = y_pred.loc[valid_idx]

    # Compute metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R2'],
        'Value': [rmse, mae, r2]
    })

    return metrics_df
