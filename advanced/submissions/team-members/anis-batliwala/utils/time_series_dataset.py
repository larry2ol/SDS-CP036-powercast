import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
  """
    Custom PyTorch Dataset for time-series data.

    This wraps pre-processed lookback windows into a format that 
    can be fed into a DataLoader for batching, shuffling, etc.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (num_samples, lookback, num_features).
        Example: 52272 samples, each using a 144-step window with 14 features.
    y : np.ndarray
        Targets. 
        - Shape (num_samples,) for sequence-to-one (predicting a single step).
        - Shape (num_samples, predict_next, num_targets) for sequence-to-sequence.
    device : str, optional
        'cpu' or 'cuda'. Tensors will be moved to this device.
    """
  def __init__(self, X, y, device="cpu"):
    self.X = torch.tensor(X, dtype=torch.float32, device=device)
    self.y = torch.tensor(y, dtype=torch.float32, device=device)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]
  