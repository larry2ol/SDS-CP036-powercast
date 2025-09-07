import numpy as np

def create_lookback_windows(df, target_cols, lookback=24, predict_next=1, seq_type='seq2one'):
  """
  Create lookback windows for time series modeling (sequence-to-one or sequence-to-sequence).

  Parameters:
  df (pd.DataFrame): Preprocessed DataFrame (scaled features + cyclical features).
  target_cols (list): List of target column names.
  lookback (int): Number of past time steps to include in each input window.
  predict_next (int): Number of future steps to predict (for seq2seq).
  seq_type (str): 'seq2one' or 'seq2seq'.

  Returns:
  X (np.array): Input windows, shape (num_samples, lookback, num_features)
  y (np.array): Targets, shape depends on seq_type
  """
  
  data = df.values
  target_indices = [df.columns.get_loc(c) for c in target_cols]
  
  X, y = [], []
  
  if seq_type == 'seq2one':
      for i in range(lookback, len(df)):
          X.append(data[i-lookback:i])
          y.append(data[i, target_indices])
      X = np.array(X)
      y = np.array(y)
      if y.shape[1] == 1:
          y = y.flatten()  # shape (num_samples,) for single target
  elif seq_type == 'seq2seq':
      for i in range(lookback, len(df) - predict_next + 1):
          X.append(data[i-lookback:i])
          y.append(data[i:i+predict_next, target_indices])
      X = np.array(X)
      y = np.array(y)
  else:
      raise ValueError("seq_type must be 'seq2one' or 'seq2seq'")
  
  return X, y
