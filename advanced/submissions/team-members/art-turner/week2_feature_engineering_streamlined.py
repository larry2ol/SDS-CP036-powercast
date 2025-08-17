"""
Week 2: Feature Engineering & Deep Learning Prep - Streamlined Version
Tetuan City Power Consumption Dataset
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import json

class PowerConsumptionDataset(Dataset):
    """Custom PyTorch Dataset for power consumption sequence data"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def main():
    print("WEEK 2: FEATURE ENGINEERING & DEEP LEARNING PREP - STREAMLINED")
    print("="*70)
    
    # Step 1: Load data
    print("1. Loading dataset...")
    df = pd.read_csv('data/Tetuan City power consumption.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%m/%d/%Y %H:%M')
    print(f"   Dataset shape: {df.shape}")
    
    # Step 2: Engineer cyclical time features
    print("2. Engineering cyclical time features...")
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['month'] = df['DateTime'].dt.month
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    cyclical_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    print(f"   Created {len(cyclical_features)} cyclical features")
    
    # Step 3: Define features and targets
    env_features = ['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']
    feature_cols = env_features + cyclical_features
    target_cols = ['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']
    
    print(f"   Features: {len(feature_cols)} columns")
    print(f"   Targets: {len(target_cols)} columns")
    
    # Step 4: Temporal data splitting (70/15/15)
    print("3. Splitting data temporally...")
    n_samples = len(df)
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)
    
    train_data = df.iloc[:train_end].copy()
    val_data = df.iloc[train_end:val_end].copy()
    test_data = df.iloc[val_end:].copy()
    
    print(f"   Train: {len(train_data)} samples")
    print(f"   Val:   {len(val_data)} samples")
    print(f"   Test:  {len(test_data)} samples")
    
    # Step 5: Normalize features (fit only on training data)
    print("4. Normalizing features...")
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train_data[feature_cols])
    
    train_data_norm = train_data.copy()
    val_data_norm = val_data.copy()
    test_data_norm = test_data.copy()
    
    train_data_norm[feature_cols] = feature_scaler.transform(train_data[feature_cols])
    val_data_norm[feature_cols] = feature_scaler.transform(val_data[feature_cols])
    test_data_norm[feature_cols] = feature_scaler.transform(test_data[feature_cols])
    
    print("   Features normalized using MinMaxScaler")
    
    # Step 6: Determine lookback window (based on Week 1 analysis: 6 hours optimal)
    # 6 hours = 36 time steps (10-minute intervals)
    lookback_window = 36
    print(f"5. Using lookback window: {lookback_window} steps (6 hours)")
    
    # Step 7: Create sequences
    print("6. Creating sequences...")
    
    def create_sequences(data, feature_cols, target_cols, window_size):
        sequences = []
        targets = []
        
        for i in range(len(data) - window_size):
            # Input sequence
            seq = data[feature_cols].iloc[i:i+window_size].values
            sequences.append(seq)
            
            # Target (next time step)
            target = data[target_cols].iloc[i+window_size].values
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    train_sequences, train_targets = create_sequences(train_data_norm, feature_cols, target_cols, lookback_window)
    val_sequences, val_targets = create_sequences(val_data_norm, feature_cols, target_cols, lookback_window)
    test_sequences, test_targets = create_sequences(test_data_norm, feature_cols, target_cols, lookback_window)
    
    print(f"   Train sequences: {train_sequences.shape}")
    print(f"   Val sequences: {val_sequences.shape}")
    print(f"   Test sequences: {test_sequences.shape}")
    
    # Step 8: Create PyTorch DataLoaders
    print("7. Creating PyTorch DataLoaders...")
    
    train_dataset = PowerConsumptionDataset(train_sequences, train_targets)
    val_dataset = PowerConsumptionDataset(val_sequences, val_targets)
    test_dataset = PowerConsumptionDataset(test_sequences, test_targets)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print(f"   Train loader: {len(train_loader)} batches")
    print(f"   Val loader: {len(val_loader)} batches")
    print(f"   Test loader: {len(test_loader)} batches")
    
    # Step 9: Test data loader functionality
    print("8. Testing data loader functionality...")
    for batch_sequences, batch_targets in train_loader:
        print(f"   Batch sequences shape: {batch_sequences.shape}")
        print(f"   Batch targets shape: {batch_targets.shape}")
        print(f"   Data types: {batch_sequences.dtype}, {batch_targets.dtype}")
        
        # Check for NaN values
        if torch.isnan(batch_sequences).any() or torch.isnan(batch_targets).any():
            print("   WARNING: NaN values detected!")
        else:
            print("   Data integrity check: PASSED")
        break
    
    # Step 10: Save processed data
    print("9. Saving processed data...")
    
    # Save sequences
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
        'lookback_window': lookback_window,
        'train_size': len(train_sequences),
        'val_size': len(val_sequences),
        'test_size': len(test_sequences),
        'cyclical_features': cyclical_features,
        'batch_size': batch_size,
        'train_date_range': [str(train_data['DateTime'].min()), str(train_data['DateTime'].max())],
        'val_date_range': [str(val_data['DateTime'].min()), str(val_data['DateTime'].max())],
        'test_date_range': [str(test_data['DateTime'].min()), str(test_data['DateTime'].max())]
    }
    
    with open('dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("   Saved: train/val/test sequences (.npy files)")
    print("   Saved: dataset_metadata.json")
    
    # Summary
    print("\n" + "="*70)
    print("FEATURE ENGINEERING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Dataset Summary:")
    print(f"  Original samples: {len(df)}")
    print(f"  Training sequences: {len(train_sequences)}")
    print(f"  Validation sequences: {len(val_sequences)}")
    print(f"  Test sequences: {len(test_sequences)}")
    print(f"  Feature dimensions: {len(feature_cols)}")
    print(f"  Target dimensions: {len(target_cols)}")
    print(f"  Lookback window: {lookback_window} time steps (6 hours)")
    print(f"  Batch size: {batch_size}")
    print()
    print("The dataset is now ready for deep learning model training!")
    print("Use the DataLoaders for training PyTorch models.")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'feature_scaler': feature_scaler,
        'metadata': metadata
    }

if __name__ == "__main__":
    results = main()