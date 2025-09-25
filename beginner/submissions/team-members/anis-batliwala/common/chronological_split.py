def chronological_train_test_split(df, train_ratio=0.8, val_ratio=0.15):
  """
  Splits a DataFrame into train, validation, and test sets while preserving chronological order.
  Supports both train/test and train/val/test splits.

  Parameters
  ----------
  df : pd.DataFrame
      The dataset to split. It should be sorted in chronological order before calling.
  train_ratio : float, default=0.8
      Proportion of data to use for training.
  val_ratio : float or None, default=0.15
      Proportion of data to use for validation. If None, only train/test split is performed.

  Returns
  -------
  If val_ratio is None:
      train, test
  Else:
      train, val, test
  """
  if 'DateTime' in df.columns:
      df = df.sort_values('DateTime')

  n_samples = len(df)

  if val_ratio is None:
    # Train/Test split only
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    split_idx = int(n_samples * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test
  else:
    # Train/Val/Test split
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test
