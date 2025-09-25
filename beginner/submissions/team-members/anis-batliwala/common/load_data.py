def load_data(file_path, index_col=None, fill_method='ffill'):
  """
  Load and preprocess time series data from a CSV file with optimized operations.
  
  Parameters:
  - file_path: Path to the CSV file
  - index_col: Column to use as datetime index (None for auto-detection)
  - fill_method: Method for filling missing values ('ffill', 'bfill', or None)
  
  Returns:
  - Preprocessed DataFrame with datetime index
  """
  import pandas as pd

  # Load data with optimized parameters
  dataset = pd.read_csv(
      file_path,
      low_memory=False,
      parse_dates=False,  # Manual parsing for better control
  )

  # Auto-detect datetime column if not specified
  if index_col is None:
      # Vectorized datetime detection (faster than looping)
      datetime_candidates = [
          col for col in dataset.columns
          if pd.to_datetime(dataset[col], errors='coerce').notna().mean() > 0.9
      ]
      
      if not datetime_candidates:
          raise ValueError("No suitable datetime column found. Please specify index_col.")
      
      index_col = datetime_candidates[0]
      print(f"[INFO] Using '{index_col}' as datetime index column")

  # Convert to datetime in one operation
  datetime_series = pd.to_datetime(dataset[index_col], errors='coerce')
  
  dataset['DateTime_Copy'] = datetime_series  # Temporary column to avoid SettingWithCopyWarning

  # Filter valid rows before sorting (more efficient)
  valid_mask = datetime_series.notna()
  dataset = dataset[valid_mask].copy()  # Copy to avoid SettingWithCopyWarning
  
  # Set index in one operation
  dataset.index = pd.DatetimeIndex(datetime_series[valid_mask])
  
  # Sort index (faster than sort_values)
  dataset.sort_index(inplace=True)

  # Handle missing values - FIXED VERSION (assigns the result)
  if fill_method == 'ffill':
      dataset = dataset.ffill()  # Assign the filled DataFrame
  elif fill_method == 'bfill':
      dataset = dataset.bfill()  # Assign the filled DataFrame
  elif fill_method is not None:
      print(f"[WARNING] Unknown fill_method '{fill_method}'. No filling applied.")

  dataset.drop(columns=['DateTime'], inplace=True, errors='ignore')

  return dataset