def add_time_based_features(dataset):
  """
  Adds time-based features to the dataset based on the datetime index column.
  Optimized using vectorized operations and attribute access.

  Parameters:
  - dataset: pd.DataFrame with a datetime index column.

  Returns:
  - pd.DataFrame with additional time-based features.
  """
  import pandas as pd

  # Convert index to datetime if needed (using errors='coerce' for safety)
  if not pd.api.types.is_datetime64_any_dtype(dataset.index):
      dataset.index = pd.to_datetime(dataset.index, errors='coerce')
      # Drop rows with invalid datetime indices
      dataset = dataset[dataset.index.notna()].copy()

  # Get index as DatetimeIndex for faster access
  dt_index = dataset.index

  # Create all features in one operation using assign (more efficient)
  dataset = dataset.assign(
      hour=dt_index.hour,
      weekday=dt_index.weekday,
      is_weekend=dt_index.weekday.isin([5, 6]).astype(int),
      month=dt_index.month
  )

  return dataset