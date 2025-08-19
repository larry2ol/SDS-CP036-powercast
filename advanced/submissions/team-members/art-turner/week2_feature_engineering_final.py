"""
Week 2: Feature Engineering & Deep Learning Prep - Final Optimized Version
Tetuan City Power Consumption Dataset
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import json

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
    print("WEEK 2: FEATURE ENGINEERING & DEEP LEARNING PREP - FINAL")
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
    feature_cols = env_features + cyclical_features
    target_cols = ['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']
    
    print(f"   Features: {len(feature_cols)} columns")
    print(f"   Targets: {len(target_cols)} columns")
    
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
    
    # Normalization
    print("4. Feature normalization...")
    scaler = MinMaxScaler()
    scaler.fit(train_data[feature_cols])
    
    # Apply normalization
    train_data[feature_cols] = scaler.transform(train_data[feature_cols])
    val_data[feature_cols] = scaler.transform(val_data[feature_cols])
    test_data[feature_cols] = scaler.transform(test_data[feature_cols])
    
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
        
        if torch.isnan(batch_seq).any() or torch.isnan(batch_target).any():
            print("   ERROR: NaN values detected!")
        else:
            print("   Data integrity: PASSED")
        break
    
    # Save results
    print("9. Saving processed data...")
    
    # Save arrays
    np.save('train_sequences.npy', train_sequences)
    np.save('train_targets.npy', train_targets)
    np.save('val_sequences.npy', val_sequences)
    np.save('val_targets.npy', val_targets)
    np.save('test_sequences.npy', test_sequences)
    np.save('test_targets.npy', test_targets)
    
    # Save metadata
    metadata = {
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'env_features': env_features,
        'cyclical_features': cyclical_features,
        'lookback_window': lookback_window,
        'batch_size': batch_size,
        'train_size': len(train_sequences),
        'val_size': len(val_sequences),
        'test_size': len(test_sequences),
        'original_dataset_size': len(df),
        'train_date_range': [str(train_data['DateTime'].min()), str(train_data['DateTime'].max())],
        'val_date_range': [str(val_data['DateTime'].min()), str(val_data['DateTime'].max())],
        'test_date_range': [str(test_data['DateTime'].min()), str(test_data['DateTime'].max())]
    }
    
    with open('dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("   Saved sequence arrays and metadata")
    
    # Final summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
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
    print("  - Sequence-to-one prediction setup")
    print("  - Temporal train/val/test splits")
    print("  - PyTorch-ready DataLoaders")
    print()
    print("Files Created:")
    print("  - train/val/test_sequences.npy")
    print("  - train/val/test_targets.npy") 
    print("  - dataset_metadata.json")
    print()
    print("The dataset is ready for deep learning model training!")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': scaler,
        'metadata': metadata
    }

if __name__ == "__main__":
    results = main()