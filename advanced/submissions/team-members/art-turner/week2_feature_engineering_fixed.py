"""
Week 2: Feature Engineering & Deep Learning Prep - FIXED VERSION
Fixed target normalization for RNN models
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import json
import pickle

class PowerConsumptionDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def create_sequences_efficient(data, feature_cols, target_cols, window_size):
    """Efficiently create sequences using vectorized operations"""
    n_samples = len(data) - window_size
    n_features = len(feature_cols)
    n_targets = len(target_cols)
    
    # Pre-allocate arrays
    sequences = np.zeros((n_samples, window_size, n_features))
    targets = np.zeros((n_samples, n_targets))
    
    # Get feature and target data as numpy arrays
    feature_data = data[feature_cols].values
    target_data = data[target_cols].values
    
    # Create sequences efficiently
    for i in range(n_samples):
        sequences[i] = feature_data[i:i+window_size]
        targets[i] = target_data[i+window_size]
    
    return sequences, targets

def main():
    print("WEEK 2: FEATURE ENGINEERING & DEEP LEARNING PREP - FIXED")
    print("="*60)
    
    # Load data
    print("1. Loading and preparing dataset...")
    df = pd.read_csv('data/Tetuan City power consumption.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%m/%d/%Y %H:%M')
    print(f"   Dataset: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Engineer cyclical features
    print("2. Engineering cyclical time features...")
    df['hour_sin'] = np.sin(2 * np.pi * df['DateTime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['DateTime'].dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['DateTime'].dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['DateTime'].dt.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['DateTime'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['DateTime'].dt.month / 12)
    
    # Define features and targets
    env_features = ['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']
    cyclical_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    # Include past power usage as autoregressive inputs by adding target columns to features
    target_cols = ['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']
    base_feature_cols = env_features + cyclical_features
    feature_cols = base_feature_cols + target_cols
    
    print(f"   Features: {len(feature_cols)} columns (includes autoregressive past power)")
    print(f"   Targets: {len(target_cols)} columns (next step)")
    
    # Temporal splitting
    print("3. Temporal data splitting...")
    n_samples = len(df)
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)
    
    train_data = df.iloc[:train_end].copy()
    val_data = df.iloc[train_end:val_end].copy()
    test_data = df.iloc[val_end:].copy()
    
    print(f"   Train: {len(train_data)} samples")
    print(f"   Val: {len(val_data)} samples") 
    print(f"   Test: {len(test_data)} samples")
    
    # CRITICAL FIX: Normalize both features AND targets
    print("4. Feature AND target normalization...")
    
    # Feature normalization (env + cyclical only, not targets)
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train_data[base_feature_cols])
    
    train_data[base_feature_cols] = feature_scaler.transform(train_data[base_feature_cols])
    val_data[base_feature_cols] = feature_scaler.transform(val_data[base_feature_cols])
    test_data[base_feature_cols] = feature_scaler.transform(test_data[base_feature_cols])
    
    # NEW: Target normalization (used for both learning target and as normalized inputs)
    target_scaler = MinMaxScaler()
    target_scaler.fit(train_data[target_cols])
    
    print("   Target statistics before normalization:")
    print(f"     Range: [{train_data[target_cols].min().min():.1f}, {train_data[target_cols].max().max():.1f}]")
    print(f"     Mean: {train_data[target_cols].mean().mean():.1f}")
    
    train_data[target_cols] = target_scaler.transform(train_data[target_cols])
    val_data[target_cols] = target_scaler.transform(val_data[target_cols])
    test_data[target_cols] = target_scaler.transform(test_data[target_cols])
    
    print("   Target statistics after normalization:")
    print(f"     Range: [{train_data[target_cols].min().min():.3f}, {train_data[target_cols].max().max():.3f}]")
    print(f"     Mean: {train_data[target_cols].mean().mean():.3f}")
    
    # Lookback window (6 hours = 36 steps based on Week 1 analysis)
    lookback_window = 36
    print(f"5. Using lookback window: {lookback_window} steps (6 hours)")
    
    # Create sequences
    print("6. Creating sequences...")
    print("   Processing training data...")
    train_sequences, train_targets = create_sequences_efficient(
        train_data, feature_cols, target_cols, lookback_window)
    
    print("   Processing validation data...")
    val_sequences, val_targets = create_sequences_efficient(
        val_data, feature_cols, target_cols, lookback_window)
    
    print("   Processing test data...")
    test_sequences, test_targets = create_sequences_efficient(
        test_data, feature_cols, target_cols, lookback_window)
    
    print(f"   Train sequences: {train_sequences.shape}")
    print(f"   Val sequences: {val_sequences.shape}")
    print(f"   Test sequences: {test_sequences.shape}")
    
    # Create DataLoaders
    print("7. Creating PyTorch DataLoaders...")
    batch_size = 64
    
    train_dataset = PowerConsumptionDataset(train_sequences, train_targets)
    val_dataset = PowerConsumptionDataset(val_sequences, val_targets)
    test_dataset = PowerConsumptionDataset(test_sequences, test_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"   Train loader: {len(train_loader)} batches")
    print(f"   Val loader: {len(val_loader)} batches")
    print(f"   Test loader: {len(test_loader)} batches")
    
    # Test functionality
    print("8. Testing DataLoader functionality...")
    for batch_seq, batch_target in train_loader:
        print(f"   Batch sequence shape: {batch_seq.shape}")
        print(f"   Batch target shape: {batch_target.shape}")
        print(f"   Data types: {batch_seq.dtype}, {batch_target.dtype}")
        print(f"   Target range: [{batch_target.min():.3f}, {batch_target.max():.3f}]")
        
        if torch.isnan(batch_seq).any() or torch.isnan(batch_target).any():
            print("   ERROR: NaN values detected!")
        else:
            print("   Data integrity: PASSED")
        break
    
    # Save results
    print("9. Saving processed data...")
    
    # Save arrays with fixed suffix
    np.save('train_sequences_fixed.npy', train_sequences)
    np.save('train_targets_fixed.npy', train_targets)
    np.save('val_sequences_fixed.npy', val_sequences)
    np.save('val_targets_fixed.npy', val_targets)
    np.save('test_sequences_fixed.npy', test_sequences)
    np.save('test_targets_fixed.npy', test_targets)
    
    # Save scalers for denormalization during evaluation
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(feature_scaler, f)
    
    with open('target_scaler.pkl', 'wb') as f:
        pickle.dump(target_scaler, f)
    
    # Save metadata
    metadata = {
        'feature_cols': feature_cols,
        'base_feature_cols': base_feature_cols,
        'target_cols': target_cols,
        'env_features': env_features,
        'cyclical_features': cyclical_features,
        'uses_autoregressive_features': True,
        'lookback_window': lookback_window,
        'batch_size': batch_size,
        'train_size': len(train_sequences),
        'val_size': len(val_sequences),
        'test_size': len(test_sequences),
        'original_dataset_size': len(df),
        'target_normalized': True,  # NEW FLAG
        'train_date_range': [str(train_data['DateTime'].min()), str(train_data['DateTime'].max())],
        'val_date_range': [str(val_data['DateTime'].min()), str(val_data['DateTime'].max())],
        'test_date_range': [str(test_data['DateTime'].min()), str(test_data['DateTime'].max())]
    }
    
    with open('dataset_metadata_fixed.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("   Saved fixed sequence arrays, scalers, and metadata")
    
    # Final summary
    print("\n" + "="*60)
    print("FIXED FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Dataset Summary:")
    print(f"  Original data: {len(df):,} samples")
    print(f"  Train sequences: {len(train_sequences):,}")
    print(f"  Val sequences: {len(val_sequences):,}")
    print(f"  Test sequences: {len(test_sequences):,}")
    print(f"  Input features: {len(feature_cols)}")
    print(f"  Target variables: {len(target_cols)}")
    print(f"  Lookback window: {lookback_window} time steps")
    print(f"  Batch size: {batch_size}")
    print()
    print("Key Features Engineered:")
    print("  - Environmental variables (normalized)")
    print("  - Cyclical time features (hour, day, month)")
    print("  - FIXED: Target variables (normalized 0-1)")
    print("  - Sequence-to-one prediction setup")
    print("  - Temporal train/val/test splits")
    print("  - PyTorch-ready DataLoaders")
    print()
    print("Files Created:")
    print("  - *_fixed.npy (sequences and targets)")
    print("  - feature_scaler.pkl, target_scaler.pkl")
    print("  - dataset_metadata_fixed.json")
    print()
    print("CRITICAL FIX: Targets are now normalized 0-1 like inputs!")
    print("This should dramatically improve RNN model performance!")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'metadata': metadata
    }

if __name__ == "__main__":
    results = main()
