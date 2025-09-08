"""
Quick Data Analysis to Debug Neural Network Issues
Investigates why neural networks are getting negative R² scores
"""

import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
import seaborn as sns

def load_data():
    """Load all data and metadata"""
    print("Loading data and metadata...")
    
    # Load metadata
    with open('dataset_metadata_fixed.json', 'r') as f:
        metadata = json.load(f)
    
    # Load scalers
    with open('feature_scaler.pkl', 'rb') as f:
        feature_scaler = pickle.load(f)
    with open('target_scaler.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    
    # Load sequences and targets
    data = {}
    for split in ['train', 'val', 'test']:
        data[f'{split}_sequences'] = np.load(f'{split}_sequences_fixed.npy')
        data[f'{split}_targets'] = np.load(f'{split}_targets_fixed.npy')
    
    return data, metadata, feature_scaler, target_scaler

def analyze_data_quality(data, metadata, feature_scaler, target_scaler):
    """Analyze data quality and preprocessing"""
    print("\n" + "="*60)
    print("DATA QUALITY ANALYSIS")
    print("="*60)
    
    # Check data shapes
    print(f"Metadata: {metadata}")
    print(f"\nData shapes:")
    for key, array in data.items():
        print(f"  {key}: {array.shape}")
    
    # Check for NaN/inf values
    print(f"\nChecking for NaN/inf values:")
    for key, array in data.items():
        nan_count = np.isnan(array).sum()
        inf_count = np.isinf(array).sum()
        print(f"  {key}: {nan_count} NaN, {inf_count} inf")
    
    # Check data ranges (should be normalized)
    print(f"\nData ranges (should be roughly [-3, 3] for normalized):")
    for key, array in data.items():
        print(f"  {key}: [{array.min():.3f}, {array.max():.3f}]")
    
    # Check target distribution
    print(f"\nTarget statistics:")
    for split in ['train', 'val', 'test']:
        targets = data[f'{split}_targets']
        print(f"  {split} targets:")
        print(f"    Shape: {targets.shape}")
        print(f"    Range: [{targets.min():.3f}, {targets.max():.3f}]")
        print(f"    Mean: {targets.mean(axis=0)}")
        print(f"    Std: {targets.std(axis=0)}")

def check_temporal_alignment(data):
    """Check if sequences and targets are properly aligned"""
    print(f"\n" + "="*60)
    print("TEMPORAL ALIGNMENT CHECK")
    print("="*60)
    
    # Take first few sequences and targets to check alignment
    train_seq = data['train_sequences'][:5]  # First 5 samples
    train_targ = data['train_targets'][:5]
    
    print(f"Sequence shape: {train_seq.shape}")
    print(f"Target shape: {train_targ.shape}")
    
    # The last time step of sequence should relate to the target
    print(f"\nFirst sample analysis:")
    print(f"  Last timestep of sequence: {train_seq[0, -1, :3]}")  # Last timestep, first 3 features
    print(f"  Target for next timestep: {train_targ[0]}")
    
    # Check if there's obvious temporal relationship
    correlation = np.corrcoef(train_seq[0, -1, :3], train_targ[0])[0, 1] if len(train_seq[0, -1, :3]) == len(train_targ[0]) else "N/A"
    print(f"  Correlation between last input and target: {correlation}")

def test_baseline_models(data, target_scaler):
    """Test simple ML models as baseline"""
    print(f"\n" + "="*60)
    print("BASELINE MODEL COMPARISON")
    print("="*60)
    
    # Prepare data - flatten sequences for traditional ML
    def prepare_ml_data(sequences, targets):
        # Take last timestep of each sequence as features
        X = sequences[:, -1, :]  # Shape: (samples, features)
        y = targets
        return X, y
    
    X_train, y_train = prepare_ml_data(data['train_sequences'], data['train_targets'])
    X_test, y_test = prepare_ml_data(data['test_sequences'], data['test_targets'])
    
    print(f"ML Data shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics (normalized space)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"  R² (normalized): {r2:.4f}")
        print(f"  RMSE (normalized): {rmse:.4f}")
        print(f"  MAE (normalized): {mae:.4f}")
        
        # Try to denormalize for real-world metrics
        try:
            y_test_real = target_scaler.inverse_transform(y_test)
            y_pred_real = target_scaler.inverse_transform(y_pred)
            
            r2_real = r2_score(y_test_real, y_pred_real)
            rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
            mae_real = mean_absolute_error(y_test_real, y_pred_real)
            
            print(f"  R² (real): {r2_real:.4f}")
            print(f"  RMSE (real): {rmse_real:.1f}")
            print(f"  MAE (real): {mae_real:.1f}")
            
            results[name] = {
                'r2_norm': r2, 'rmse_norm': rmse, 'mae_norm': mae,
                'r2_real': r2_real, 'rmse_real': rmse_real, 'mae_real': mae_real
            }
        except Exception as e:
            print(f"  Error denormalizing: {e}")
            results[name] = {'r2_norm': r2, 'rmse_norm': rmse, 'mae_norm': mae}
    
    return results

def check_scaling_consistency(data, feature_scaler, target_scaler):
    """Check if scaling is consistent and working correctly"""
    print(f"\n" + "="*60)
    print("SCALING CONSISTENCY CHECK")
    print("="*60)
    
    # Check if we can properly inverse transform
    test_targets = data['test_targets']
    
    try:
        # Try inverse transform
        targets_real = target_scaler.inverse_transform(test_targets)
        print(f"Inverse transform successful!")
        print(f"  Normalized targets range: [{test_targets.min():.3f}, {test_targets.max():.3f}]")
        print(f"  Real targets range: [{targets_real.min():.1f}, {targets_real.max():.1f}]")
        print(f"  Real targets mean: {targets_real.mean(axis=0)}")
        
        # Check if re-normalizing gives back original
        targets_renorm = target_scaler.transform(targets_real)
        diff = np.abs(test_targets - targets_renorm).max()
        print(f"  Round-trip error (max): {diff:.6f}")
        
    except Exception as e:
        print(f"Error with target scaler: {e}")
    
    # Check feature scaler
    test_sequences = data['test_sequences']
    try:
        # Reshape to 2D for inverse transform
        seq_2d = test_sequences.reshape(-1, test_sequences.shape[-1])
        features_real = feature_scaler.inverse_transform(seq_2d)
        print(f"Feature inverse transform successful!")
        print(f"  Normalized features range: [{seq_2d.min():.3f}, {seq_2d.max():.3f}]")
        print(f"  Real features range: [{features_real.min():.1f}, {features_real.max():.1f}]")
        
    except Exception as e:
        print(f"Error with feature scaler: {e}")

def main():
    """Run comprehensive data analysis"""
    print("POWER FORECASTING DATA DEBUG ANALYSIS")
    print("="*60)
    
    try:
        # Load data
        data, metadata, feature_scaler, target_scaler = load_data()
        
        # Run analyses
        analyze_data_quality(data, metadata, feature_scaler, target_scaler)
        check_temporal_alignment(data)
        check_scaling_consistency(data, feature_scaler, target_scaler)
        baseline_results = test_baseline_models(data, target_scaler)
        
        print(f"\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        print(f"Baseline model performance:")
        for name, metrics in baseline_results.items():
            r2 = metrics.get('r2_real', metrics.get('r2_norm', 'N/A'))
            print(f"  {name}: R² = {r2:.4f}")
        
        print(f"\nIf baseline models achieve positive R² but neural networks don't,")
        print(f"the issue is likely with:")
        print(f"  1. Neural network architecture/hyperparameters")
        print(f"  2. Training procedure (learning rate, optimizer)")
        print(f"  3. Data loading/batching for sequences")
        print(f"  4. Loss function or metric calculation")
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()