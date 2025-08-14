def chronological_train_test_split(df, split_ratio=0.8):
  """
  Splits a DataFrame into train and test sets while preserving chronological order.

  This method is suitable for time-series or sequential data where shuffling
  would cause data leakage. The split is made by selecting the first `split_ratio`
  fraction of rows as training data and the remaining rows as test data.

  Parameters
  ----------
  df : pd.DataFrame
      The dataset to split. It should be sorted in chronological order before calling.
  split_ratio : float, optional, default=0.8
      The proportion of data to use for training. Must be between 0 and 1.

  Returns
  -------
  train : pd.DataFrame
      The training set (first `split_ratio` portion of the data).
  test : pd.DataFrame
      The test set (remaining portion of the data).
  """
  if not 0 < split_ratio < 1:
    raise ValueError("split_ratio must be between 0 and 1")

  if 'DateTime' in df.columns:
    df = df.sort_values('DateTime')

  split_idx = int(len(df) * split_ratio)
  # Ensure the test set is a copy to avoid SettingWithCopyWarning
  train = df.iloc[:split_idx].copy()
  # Ensure the test set is a copy to avoid SettingWithCopyWarning
  test = df.iloc[split_idx:].copy()
  
  return train, test
