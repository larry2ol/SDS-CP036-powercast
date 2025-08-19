"""
Week 2: Feature Engineering & Deep Learning Prep
Tetuan City Power Consumption Dataset

This script handles:
1. Creating lookback windows for sequence modeling
2. Normalizing continuous variables and engineering cyclical time features
3. Converting data into tensors and formatting train/val/test splits
4. Preparing PyTorch DataLoader objects
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class PowerConsumptionDataset(Dataset):
    """Custom PyTorch Dataset for power consumption sequence data"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def load_and_prepare_data():
    """Load and prepare the Tetuan City power consumption dataset"""
    print("Loading Tetuan City power consumption dataset...")
    
    # Load data
    df = pd.read_csv('data/Tetuan City power consumption.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%m/%d/%Y %H:%M')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
    print(f"Features: {df.columns.tolist()}")
    
    return df

def engineer_cyclical_features(df):
    """Engineer cyclical time features using sine/cosine transforms"""
    print("\nEngineering cyclical time features...")
    
    # Extract time components
    df = df.copy()
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['month'] = df['DateTime'].dt.month
    df['day_of_year'] = df['DateTime'].dt.dayofyear
    
    # Create cyclical features using sine/cosine transforms
    # Hour of day (24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week (7-day cycle)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Month (12-month cycle)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Day of year (365-day cycle)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    cyclical_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
                        'month_sin', 'month_cos', 'doy_sin', 'doy_cos']
    
    print(f"Created {len(cyclical_features)} cyclical features:")
    for feat in cyclical_features:
        print(f"  - {feat}: range [{df[feat].min():.3f}, {df[feat].max():.3f}]")
    
    return df, cyclical_features

def determine_optimal_lookback_window(df, target_cols):
    """Determine optimal lookback window size based on autocorrelation analysis"""
    print("\nDetermining optimal lookback window size...")
    
    # Test different window sizes based on previous lag analysis findings
    # From Week 1 analysis: 3-6 hour lags were most relevant
    # 10-minute intervals: 6 hours = 36 intervals, 3 hours = 18 intervals
    
    candidate_windows = [12, 18, 24, 36, 48, 72]  # 2h, 3h, 4h, 6h, 8h, 12h
    window_names = ['2h', '3h', '4h', '6h', '8h', '12h']
    
    # Calculate autocorrelation for total power at different lags
    # Create total power by summing the three zones
    total_power = (df['Zone 1 Power Consumption'] + 
                  df['Zone 2  Power Consumption'] + 
                  df['Zone 3  Power Consumption']).values
    autocorr_scores = []
    
    for window in candidate_windows:
        # Calculate autocorrelation at this lag
        if len(total_power) > window:
            corr = np.corrcoef(total_power[:-window], total_power[window:])[0, 1]
            autocorr_scores.append(corr)
        else:
            autocorr_scores.append(0)
    
    # Find optimal window based on strong but not too high autocorrelation
    # Too high = very short-term dependency, too low = weak relationship
    optimal_idx = np.argmax([abs(score) for score in autocorr_scores if abs(score) < 0.95])
    optimal_window = candidate_windows[optimal_idx]
    optimal_name = window_names[optimal_idx]
    
    print(f"Autocorrelation analysis for different lookback windows:")
    for i, (window, name, score) in enumerate(zip(candidate_windows, window_names, autocorr_scores)):
        marker = " *" if i == optimal_idx else ""
        print(f"  {name:>3} ({window:2d} steps): {score:.4f}{marker}")
    
    print(f"\nOptimal lookback window: {optimal_name} ({optimal_window} time steps)")
    return optimal_window

def create_sequences(data, feature_cols, target_cols, window_size, forecast_horizon=1):
    """Create input sequences and target values for sequence modeling"""
    print(f"\nCreating sequences with window_size={window_size}, forecast_horizon={forecast_horizon}")
    
    sequences = []
    targets = []
    
    # Create sequences
    for i in range(len(data) - window_size - forecast_horizon + 1):
        # Input sequence (features over lookback window)
        seq = data[feature_cols].iloc[i:i+window_size].values
        sequences.append(seq)
        
        # Target (power consumption at forecast horizon)
        if forecast_horizon == 1:
            # Sequence-to-one: predict next single time step
            target = data[target_cols].iloc[i+window_size].values
        else:
            # Sequence-to-sequence: predict next forecast_horizon steps
            target = data[target_cols].iloc[i+window_size:i+window_size+forecast_horizon].values
        
        targets.append(target)
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    print(f"Created {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Target shape: {targets.shape}")
    
    return sequences, targets

def normalize_features(train_data, val_data, test_data, feature_cols):
    """Normalize features using training data statistics to prevent data leakage"""
    print("\nNormalizing features...")
    
    # Initialize scalers
    feature_scaler = MinMaxScaler()
    
    # Fit on training data only
    train_features = train_data[feature_cols].values
    feature_scaler.fit(train_features)
    
    # Transform all datasets
    train_data_norm = train_data.copy()
    val_data_norm = val_data.copy()
    test_data_norm = test_data.copy()
    
    train_data_norm[feature_cols] = feature_scaler.transform(train_data[feature_cols])
    val_data_norm[feature_cols] = feature_scaler.transform(val_data[feature_cols])
    test_data_norm[feature_cols] = feature_scaler.transform(test_data[feature_cols])
    
    print(f"Features normalized using MinMaxScaler")
    print(f"Feature ranges after normalization (train):")
    for col in feature_cols[:5]:  # Show first 5 features
        values = train_data_norm[col]
        print(f"  {col}: [{values.min():.3f}, {values.max():.3f}]")
    
    return train_data_norm, val_data_norm, test_data_norm, feature_scaler

def temporal_train_val_test_split(df, train_ratio=0.7, val_ratio=0.15):
    """Split data temporally to maintain time series integrity"""
    print(f"\nSplitting data temporally:")
    print(f"  Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {1-train_ratio-val_ratio:.1%}")
    
    n_samples = len(df)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_data = df.iloc[:train_end].copy()
    val_data = df.iloc[train_end:val_end].copy()
    test_data = df.iloc[val_end:].copy()
    
    print(f"Train: {len(train_data)} samples ({train_data['DateTime'].min()} to {train_data['DateTime'].max()})")
    print(f"Val:   {len(val_data)} samples ({val_data['DateTime'].min()} to {val_data['DateTime'].max()})")
    print(f"Test:  {len(test_data)} samples ({test_data['DateTime'].min()} to {test_data['DateTime'].max()})")
    
    return train_data, val_data, test_data

def create_data_loaders(train_sequences, train_targets, val_sequences, val_targets, 
                       test_sequences, test_targets, batch_size=32):
    """Create PyTorch DataLoader objects"""
    print(f"\nCreating PyTorch DataLoaders with batch_size={batch_size}")
    
    # Create datasets
    train_dataset = PowerConsumptionDataset(train_sequences, train_targets)
    val_dataset = PowerConsumptionDataset(val_sequences, val_targets)
    test_dataset = PowerConsumptionDataset(test_sequences, test_targets)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader:   {len(val_loader)} batches")
    print(f"Test loader:  {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

def visualize_sample_sequences(sequences, targets, feature_names, n_samples=3):
    """Visualize sample sequences to verify data preparation"""
    print(f"\nVisualizing {n_samples} sample sequences...")
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(15, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Plot input sequence features (first 4 features)
        ax1 = axes[i, 0]
        seq = sequences[i]
        for j in range(min(4, len(feature_names))):
            ax1.plot(seq[:, j], label=feature_names[j], alpha=0.7)
        ax1.set_title(f'Sample {i+1}: Input Sequence Features')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Normalized Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot target values
        ax2 = axes[i, 1]
        target = targets[i]
        if target.ndim == 1:
            ax2.bar(range(len(target)), target)
            ax2.set_title(f'Sample {i+1}: Target Values')
            ax2.set_xlabel('Zone')
            ax2.set_ylabel('Power Consumption')
        else:
            for j in range(target.shape[1]):
                ax2.plot(target[:, j], label=f'Zone {j+1}', marker='o')
            ax2.set_title(f'Sample {i+1}: Target Sequence')
            ax2.set_xlabel('Forecast Horizon')
            ax2.set_ylabel('Power Consumption')
            ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sequence_samples_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved sequence visualization as 'sequence_samples_visualization.png'")

def main():
    """Main function to execute the complete feature engineering pipeline"""
    print("="*80)
    print("WEEK 2: FEATURE ENGINEERING & DEEP LEARNING PREP")
    print("="*80)
    
    # Step 1: Load data
    df = load_and_prepare_data()
    
    # Step 2: Engineer cyclical time features
    df, cyclical_features = engineer_cyclical_features(df)
    
    # Define feature columns and target columns
    env_features = ['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']
    target_cols = ['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']
    feature_cols = env_features + cyclical_features
    
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Target columns ({len(target_cols)}): {target_cols}")
    
    # Step 3: Determine optimal lookback window
    optimal_window = determine_optimal_lookback_window(df, target_cols)
    
    # Step 4: Temporal data splitting
    train_data, val_data, test_data = temporal_train_val_test_split(df)
    
    # Step 5: Normalize features (fit only on training data)
    train_data_norm, val_data_norm, test_data_norm, feature_scaler = normalize_features(
        train_data, val_data, test_data, feature_cols)
    
    # Step 6: Create sequences for sequence-to-one modeling
    print("\n" + "="*50)
    print("CREATING SEQUENCES FOR SEQUENCE-TO-ONE MODELING")
    print("="*50)
    
    train_sequences, train_targets = create_sequences(
        train_data_norm, feature_cols, target_cols, optimal_window, forecast_horizon=1)
    val_sequences, val_targets = create_sequences(
        val_data_norm, feature_cols, target_cols, optimal_window, forecast_horizon=1)
    test_sequences, test_targets = create_sequences(
        test_data_norm, feature_cols, target_cols, optimal_window, forecast_horizon=1)
    
    # Step 7: Create PyTorch DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_sequences, train_targets, val_sequences, val_targets, 
        test_sequences, test_targets, batch_size=64)
    
    # Step 8: Visualize sample sequences
    visualize_sample_sequences(train_sequences, train_targets, feature_cols, n_samples=3)
    
    # Step 9: Data integrity verification
    print("\n" + "="*50)
    print("DATA INTEGRITY VERIFICATION")
    print("="*50)
    
    # Test data loader
    print("Testing data loader functionality...")
    for batch_idx, (sequences, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Sequences shape: {sequences.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Sequences dtype: {sequences.dtype}")
        print(f"  Targets dtype: {targets.dtype}")
        
        # Check for NaN values
        if torch.isnan(sequences).any():
            print("  WARNING: NaN values detected in sequences!")
        if torch.isnan(targets).any():
            print("  WARNING: NaN values detected in targets!")
        
        # Only check first batch
        break
    
    # Summary statistics
    print(f"\nFinal dataset summary:")
    print(f"  Original dataset: {len(df)} samples")
    print(f"  Training sequences: {len(train_sequences)}")
    print(f"  Validation sequences: {len(val_sequences)}")
    print(f"  Test sequences: {len(test_sequences)}")
    print(f"  Feature dimensionality: {len(feature_cols)}")
    print(f"  Target dimensionality: {len(target_cols)}")
    print(f"  Lookback window: {optimal_window} time steps")
    
    # Save processed data artifacts
    print(f"\nSaving processed data artifacts...")
    
    # Save sequences as numpy arrays for later use
    np.save('train_sequences.npy', train_sequences)
    np.save('train_targets.npy', train_targets)
    np.save('val_sequences.npy', val_sequences)
    np.save('val_targets.npy', val_targets)
    np.save('test_sequences.npy', test_sequences)
    np.save('test_targets.npy', test_targets)
    
    # Save feature names and metadata
    metadata = {
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'optimal_window': optimal_window,
        'train_size': len(train_sequences),
        'val_size': len(val_sequences),
        'test_size': len(test_sequences),
        'cyclical_features': cyclical_features
    }
    
    import json
    with open('dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("SUCCESS: Feature engineering pipeline completed successfully!")
    print("SUCCESS: Data is ready for deep learning model training!")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'feature_scaler': feature_scaler,
        'metadata': metadata
    }

if __name__ == "__main__":
    results = main()