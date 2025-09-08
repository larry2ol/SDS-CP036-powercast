"""
Quick BiLSTM Optimization - Focused on best performing architecture
Based on Week 4 results showing BiLSTM achieved R² = 0.203
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from pathlib import Path

from advanced_models import BidirectionalLSTM
from training_fixed import MetricsCalculator
from week2_feature_engineering_fixed import PowerConsumptionDataset
import pickle
import json


def create_optimized_bilstm_configs():
    """Create focused BiLSTM configurations based on successful results"""
    configs = []
    
    # Base configuration that worked well
    base_config = {
        'model_class': BidirectionalLSTM,
        'model_name': 'BiLSTM',
        'input_size': 11,
        'output_size': 3,
        'optimizer_name': 'AdamW',
        'scheduler': 'ReduceLROnPlateau',
        'early_stopping_patience': 15,
        'max_epochs': 100,
        'batch_size': 64
    }
    
    # Systematic grid around successful parameters
    hidden_sizes_options = [
        [64, 32],      # Smaller, faster
        [128, 64],     # Medium (likely what worked)
        [256, 128],    # Larger
        [128, 64, 32]  # Deeper
    ]
    
    learning_rates = [0.001, 0.0005, 0.002]
    dropout_rates = [0.1, 0.2, 0.3]
    
    config_id = 0
    for hidden_sizes in hidden_sizes_options:
        for lr in learning_rates:
            for dropout in dropout_rates:
                config = base_config.copy()
                config.update({
                    'config_id': config_id,
                    'hidden_sizes': hidden_sizes,
                    'dropout_rate': dropout,
                    'learning_rate': lr,
                    'model_params': {
                        'hidden_sizes': hidden_sizes,
                        'dropout_rate': dropout
                    }
                })
                configs.append(config)
                config_id += 1
    
    print(f"Created {len(configs)} BiLSTM configurations")
    return configs


def train_single_model(config, train_loader, val_loader, device='cpu'):
    """Train a single BiLSTM model"""
    
    # Create model
    model = config['model_class'](
        input_size=config['input_size'],
        output_size=config['output_size'],
        **config['model_params']
    ).to(device)
    
    # Setup training
    if config['optimizer_name'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': []}
    
    model.train()
    for epoch in range(config['max_epochs']):
        # Training
        train_loss = 0.0
        for batch_seq, batch_target in train_loader:
            batch_seq, batch_target = batch_seq.to(device), batch_target.to(device)
            
            optimizer.zero_grad()
            output = model(batch_seq)
            loss = criterion(output, batch_target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_seq, batch_target in val_loader:
                batch_seq, batch_target = batch_seq.to(device), batch_target.to(device)
                output = model(batch_seq)
                loss = criterion(output, batch_target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        model.train()
        
        # Record history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= config['early_stopping_patience']:
            print(f"   Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, best_val_loss, training_history


def evaluate_model(model, test_loader, target_scaler, device='cpu'):
    """Evaluate model and return metrics"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_seq, batch_target in test_loader:
            batch_seq, batch_target = batch_seq.to(device), batch_target.to(device)
            
            output = model(batch_seq)
            all_predictions.append(output.cpu().numpy())
            all_targets.append(batch_target.cpu().numpy())
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Denormalize for evaluation
    predictions_denorm = target_scaler.inverse_transform(predictions)
    targets_denorm = target_scaler.inverse_transform(targets)
    
    # Calculate metrics
    metrics_calc = MetricsCalculator(target_scaler=target_scaler)
    
    # Convert back to normalized for metrics calculator
    predictions_torch = torch.FloatTensor(predictions)
    targets_torch = torch.FloatTensor(targets)
    
    metrics = metrics_calc.calculate_all_metrics(
        targets, predictions
    )
    
    return metrics, predictions_denorm, targets_denorm


def run_quick_optimization():
    """Run focused BiLSTM optimization"""
    
    print("QUICK BiLSTM OPTIMIZATION")
    print("=" * 60)
    print("Based on Week 4 results: BiLSTM achieved R² = 0.203")
    print("Focusing on BiLSTM variants for faster results")
    print()
    
    # Load data
    print("1. Loading data...")
    
    # Load metadata
    with open('dataset_metadata_fixed.json', 'r') as f:
        metadata = json.load(f)
    
    # Load scalers
    with open('feature_scaler.pkl', 'rb') as f:
        feature_scaler = pickle.load(f)
    with open('target_scaler.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    
    # Load data
    train_sequences = np.load('train_sequences_fixed.npy')
    train_targets = np.load('train_targets_fixed.npy')
    val_sequences = np.load('val_sequences_fixed.npy')
    val_targets = np.load('val_targets_fixed.npy')
    test_sequences = np.load('test_sequences_fixed.npy')
    test_targets = np.load('test_targets_fixed.npy')
    
    print(f"   Train: {len(train_sequences)} sequences")
    print(f"   Val: {len(val_sequences)} sequences")
    print(f"   Test: {len(test_sequences)} sequences")
    
    # Create DataLoaders
    train_dataset = PowerConsumptionDataset(train_sequences, train_targets)
    val_dataset = PowerConsumptionDataset(val_sequences, val_targets)
    test_dataset = PowerConsumptionDataset(test_sequences, test_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Create configurations
    configs = create_optimized_bilstm_configs()
    
    # Track results
    results = []
    
    # MLflow experiment
    mlflow.set_experiment("Quick_BiLSTM_Optimization")
    
    print(f"\n2. Training {len(configs)} BiLSTM configurations...")
    
    for i, config in enumerate(configs):
        print(f"\n   Config {i+1}/{len(configs)}: {config['hidden_sizes']}, lr={config['learning_rate']}, dropout={config['dropout_rate']}")
        
        with mlflow.start_run(run_name=f"BiLSTM_Config_{i+1}"):
            # Log parameters
            mlflow.log_params({
                'model_type': 'BiLSTM',
                'hidden_sizes': str(config['hidden_sizes']),
                'learning_rate': config['learning_rate'],
                'dropout_rate': config['dropout_rate'],
                'optimizer': config['optimizer_name'],
                'batch_size': config['batch_size']
            })
            
            # Train model
            model, best_val_loss, history = train_single_model(
                config, train_loader, val_loader, device
            )
            
            # Evaluate on test set
            metrics, predictions, targets = evaluate_model(
                model, test_loader, target_scaler, device
            )
            
            # Log metrics
            mlflow.log_metrics({
                'val_loss': best_val_loss,
                'test_rmse': metrics['RMSE_Overall'],
                'test_mae': metrics['MAE_Overall'],
                'test_r2': metrics['R2_Overall'],
                'test_mape': metrics['MAPE_Overall']
            })
            
            # Log training curves
            for epoch, (train_loss, val_loss) in enumerate(zip(history['train_loss'], history['val_loss'])):
                mlflow.log_metrics({
                    'epoch_train_loss': train_loss,
                    'epoch_val_loss': val_loss
                }, step=epoch)
            
            # Save model
            model_path = f"bilstm_config_{i+1}.pth"
            torch.save(model, model_path)
            mlflow.log_artifact(model_path)
            
            # Store results
            result = {
                'config_id': i+1,
                'config': config,
                'metrics': metrics,
                'model_path': model_path,
                'val_loss': best_val_loss
            }
            results.append(result)
            
            print(f"      R² = {metrics['R2_Overall']:.4f}, RMSE = {metrics['RMSE_Overall']:.1f}")
    
    # Find best model
    best_result = max(results, key=lambda x: x['metrics']['R2_Overall'])
    
    print(f"\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    print(f"Best Model: Config {best_result['config_id']}")
    print(f"Architecture: BiLSTM {best_result['config']['hidden_sizes']}")
    print(f"Learning Rate: {best_result['config']['learning_rate']}")
    print(f"Dropout: {best_result['config']['dropout_rate']}")
    print()
    
    best_metrics = best_result['metrics']
    print("Performance:")
    print(f"  R² = {best_metrics['R2_Overall']:.4f}")
    print(f"  RMSE = {best_metrics['RMSE_Overall']:.1f}")
    print(f"  MAE = {best_metrics['MAE_Overall']:.1f}")
    print(f"  MAPE = {best_metrics['MAPE_Overall']:.1f}%")
    
    print(f"\nBest model saved as: {best_result['model_path']}")
    
    # Save optimization results
    optimization_summary = {
        'best_config_id': int(best_result['config_id']),
        'best_model_path': best_result['model_path'],
        'best_metrics': {k: float(v) for k, v in best_result['metrics'].items()},
        'all_results': [{
            'config_id': int(r['config_id']),
            'hidden_sizes': r['config']['hidden_sizes'],
            'learning_rate': float(r['config']['learning_rate']),
            'dropout_rate': float(r['config']['dropout_rate']),
            'r2': float(r['metrics']['R2_Overall']),
            'rmse': float(r['metrics']['RMSE_Overall'])
        } for r in results]
    }
    
    with open('quick_optimization_results.json', 'w') as f:
        json.dump(optimization_summary, f, indent=2)
    
    print(f"Optimization summary saved to: quick_optimization_results.json")
    
    return best_result


if __name__ == "__main__":
    best_model_info = run_quick_optimization()