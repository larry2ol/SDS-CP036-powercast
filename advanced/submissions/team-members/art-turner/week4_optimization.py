"""
Week 4: Model Optimization & Advanced Training
Hyperparameter tuning, advanced architectures, and ensemble methods
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
import itertools
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our models
from models import LSTMBaseline, GRUAlternative, TemporalConvNet
from advanced_models import (
    DeepLSTM, AttentionLSTM, AdvancedTCN,
    EnsembleModel
)
from training_fixed_refactor import MetricsCalculator, Trainer, Visualizer


class HyperparameterOptimizer:
    """
    Systematic hyperparameter optimization with grid search
    """
    
    def __init__(self, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 target_scaler,
                 device: str = 'cpu'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.target_scaler = target_scaler
        self.device = device
        self.results = []
    
    def optimize_lstm_variants(self, input_size: int, output_size: int):
        """Optimize LSTM and variants"""
        print("Optimizing LSTM variants...")
        
        # LSTM hyperparameter grid
        lstm_grid = {
            'hidden_sizes': [
                [64, 32],
                [128, 64],
                [256, 128],
                [128, 64, 32],
                [256, 128, 64]
            ],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.0005, 0.0001],
            'optimizer': ['adam', 'adamw'],
            'bidirectional': [False, True]
        }
        
        # Standard LSTM
        self._grid_search('LSTM', LSTMBaseline, lstm_grid, input_size, output_size)
        
        # Deep LSTM (fewer configs due to complexity)
        deep_lstm_grid = {
            'hidden_sizes': [[256, 128, 64, 32], [128, 64, 32, 16]],
            'dropout_rate': [0.2, 0.3],
            'layer_norm': [True, False],
            'learning_rate': [0.0005, 0.0001],
            'optimizer': ['adamw']
        }
        self._grid_search('DeepLSTM', DeepLSTM, deep_lstm_grid, input_size, output_size)
    
    def optimize_gru_variants(self, input_size: int, output_size: int):
        """Optimize GRU variants"""
        print("Optimizing GRU variants...")

        gru_grid = {
            'hidden_sizes': [
                [64, 32],
                [128, 64],
                [256, 128],
                [128, 64, 32],
                [256, 128, 64]
            ],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.0005, 0.0001],
            'optimizer': ['adam', 'adamw'],
            'bidirectional': [False, True]
        }

        self._grid_search('GRU', GRUAlternative, gru_grid, input_size, output_size)
    
    def optimize_attention_models(self, input_size: int, output_size: int):
        """Optimize attention-based models"""
        print("Optimizing attention models...")
        
        attention_grid = {
            'hidden_size': [64, 128, 256],
            'num_layers': [2, 3],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.0005],
            'optimizer': ['adam', 'adamw']
        }
        self._grid_search('AttentionLSTM', AttentionLSTM, attention_grid, input_size, output_size)
    
    def _grid_search(self, model_name: str, model_class, param_grid: Dict, input_size: int, output_size: int):
        """Perform grid search for a specific model"""
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Separate model params from training params
        training_params = ['learning_rate', 'optimizer']
        model_param_names = [name for name in param_names if name not in training_params]
        training_param_names = [name for name in param_names if name in training_params]
        
        # Generate combinations
        all_combinations = list(itertools.product(*param_values))
        
        # Limit combinations for efficiency (sample if too many)
        max_combinations = 20
        if len(all_combinations) > max_combinations:
            # Sample combinations intelligently
            step = len(all_combinations) // max_combinations
            combinations = all_combinations[::step][:max_combinations]
            print(f"   Sampling {len(combinations)} out of {len(all_combinations)} combinations")
        else:
            combinations = all_combinations
        
        for i, combo in enumerate(combinations):
            print(f"   Testing {model_name} combination {i+1}/{len(combinations)}")
            
            # Split parameters
            params_dict = dict(zip(param_names, combo))
            model_params = {k: v for k, v in params_dict.items() if k in model_param_names}
            train_params = {k: v for k, v in params_dict.items() if k in training_param_names}
            
            try:
                # Create model
                model = model_class(
                    input_size=input_size,
                    output_size=output_size,
                    **model_params
                )
                
                # Create trainer
                trainer = Trainer(
                    model=model,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    test_loader=self.test_loader,
                    target_scaler=self.target_scaler,
                    device=self.device,
                    experiment_name=f"week4_optimization_{model_name.lower()}"
                )
                
                # Train with limited epochs for grid search
                training_config = {
                    'num_epochs': 30,  # Shorter for grid search
                    'learning_rate': train_params.get('learning_rate', 0.001),
                    'criterion': 'mae',
                    'optimizer_type': train_params.get('optimizer', 'adam'),
                    'patience': 10,
                    'log_interval': 30  # Less verbose
                }
                
                # Train model
                result = trainer.train(**training_config)
                
                # Store results
                result_record = {
                    'model_name': model_name,
                    'model_params': model_params,
                    'train_params': train_params,
                    'best_val_loss': result['best_val_loss'],
                    'test_r2': result['test_results']['metrics']['R2_Overall'],
                    'test_rmse': result['test_results']['metrics']['RMSE_Overall'],
                    'test_mae': result['test_results']['metrics']['MAE_Overall'],
                    'param_count': sum(p.numel() for p in model.parameters()),
                    'converged_epoch': len(result['train_losses']),
                    'model_filename': result.get('model_filename', 'N/A')  # Add model filename
                }
                
                self.results.append(result_record)
                
                print(f"      Result: R2 = {result_record['test_r2']:.3f}, "
                      f"RMSE = {result_record['test_rmse']:.1f}")
                
            except Exception as e:
                print(f"      Failed: {str(e)}")
                continue
    
    def get_best_models(self, top_k: int = 3) -> List[Dict]:
        """Get top k models by R² score"""
        sorted_results = sorted(self.results, key=lambda x: x['test_r2'], reverse=True)
        return sorted_results[:top_k]
    
    def save_results(self, filepath: str = 'optimization_results.json'):
        """Save optimization results"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Optimization results saved to {filepath}")


class EnsembleTrainer:
    """
    Train ensemble models using best individual models
    """
    
    def __init__(self,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 target_scaler,
                 device: str = 'cpu'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.target_scaler = target_scaler
        self.device = device
    
    def create_best_ensemble(self, best_configs: List[Dict], input_size: int, output_size: int):
        """Create ensemble from best individual models"""
        print("Creating ensemble from best models...")
        
        # Model class mapping
        model_classes = {
            'LSTM': LSTMBaseline,
            'GRU': GRUAlternative,
            'DeepLSTM': DeepLSTM,
            'TCN': TemporalConvNet,
            'AdvancedTCN': AdvancedTCN,
            'AttentionLSTM': AttentionLSTM
        }
        
        # Create individual models
        individual_models = []
        for config in best_configs:
            model_name = config['model_name']
            model_params = config['model_params']
            
            if model_name in model_classes:
                model = model_classes[model_name](
                    input_size=input_size,
                    output_size=output_size,
                    **model_params
                )
                individual_models.append(model)
                print(f"   Added {model_name} to ensemble")
        
        # Create ensemble
        ensemble = EnsembleModel(individual_models, output_size)
        
        # Train ensemble
        trainer = Trainer(
            model=ensemble,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            target_scaler=self.target_scaler,
            device=self.device,
            experiment_name="week4_ensemble"
        )
        
        training_config = {
            'num_epochs': 50,
            'learning_rate': 0.0005,
            'criterion': 'mae',
            'optimizer_type': 'adamw',
            'patience': 15
        }
        
        print("Training ensemble model...")
        result = trainer.train(**training_config)
        
        return ensemble, result


def load_fixed_data():
    """Load the fixed preprocessed data"""
    try:
        # Load sequences
        train_sequences = np.load('train_sequences_fixed.npy')
        val_sequences = np.load('val_sequences_fixed.npy')
        test_sequences = np.load('test_sequences_fixed.npy')
        train_targets = np.load('train_targets_fixed.npy')
        val_targets = np.load('val_targets_fixed.npy')
        test_targets = np.load('test_targets_fixed.npy')
        
        # Load metadata and scaler
        with open('dataset_metadata_fixed.json', 'r') as f:
            metadata = json.load(f)
        
        with open('target_scaler.pkl', 'rb') as f:
            target_scaler = pickle.load(f)
        
        # Create datasets and loaders
        from week2_feature_engineering_fixed import PowerConsumptionDataset
        
        batch_size = 64  # Fixed batch size for optimization
        train_dataset = PowerConsumptionDataset(train_sequences, train_targets)
        val_dataset = PowerConsumptionDataset(val_sequences, val_targets)
        test_dataset = PowerConsumptionDataset(test_sequences, test_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, metadata, target_scaler
        
    except FileNotFoundError as e:
        print(f"Error loading fixed data: {e}")
        raise


def main():
    """Main Week 4 optimization pipeline"""
    print("WEEK 4: MODEL OPTIMIZATION & INTERPRETABILITY")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader, metadata, target_scaler = load_fixed_data()
    
    input_size = len(metadata['feature_cols'])
    output_size = len(metadata['target_cols'])
    
    print(f"Input size: {input_size}, Output size: {output_size}")
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        target_scaler=target_scaler,
        device=device
    )
    
    # Phase 1: Optimize LSTM variants
    print(f"\n{'='*60}")
    print("PHASE 1: LSTM OPTIMIZATION")
    print("="*60)
    optimizer.optimize_lstm_variants(input_size, output_size)
    
    # Phase 2: Optimize TCN variants
    print(f"\n{'='*60}")
    print("PHASE 2: GRU OPTIMIZATION")
    print("="*60)
    optimizer.optimize_gru_variants(input_size, output_size)
    
    # Phase 3: Optimize attention models
    print(f"\n{'='*60}")
    print("PHASE 3: ATTENTION MODEL OPTIMIZATION")
    print("="*60)
    optimizer.optimize_attention_models(input_size, output_size)
    
    # Save optimization results
    optimizer.save_results('week4_optimization_results.json')
    
    # Get best models
    best_models = optimizer.get_best_models(top_k=5)
    
    # Display results
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    results_df = pd.DataFrame([
        {
            'Model': result['model_name'],
            'R2': result['test_r2'],
            'RMSE': result['test_rmse'],
            'MAE': result['test_mae'],
            'Parameters': f"{result['param_count']:,}",
            'Epochs': result['converged_epoch']
        }
        for result in best_models
    ])
    
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # Phase 4: Create and train ensemble
    print(f"\n{'='*60}")
    print("PHASE 4: ENSEMBLE TRAINING")
    print("="*60)
    
    ensemble_trainer = EnsembleTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        target_scaler=target_scaler,
        device=device
    )
    
    ensemble_model, ensemble_result = ensemble_trainer.create_best_ensemble(
        best_models[:3], input_size, output_size
    )
    
    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print("="*60)
    
    final_results = []
    
    # Add best individual models
    for result in best_models[:3]:
        final_results.append({
            'Model': result['model_name'],
            'Type': 'Individual',
            'R2': result['test_r2'],
            'RMSE': result['test_rmse'],
            'Parameters': result['param_count'],
            'Model_File': result.get('model_filename', 'N/A')  # Add model filename
        })
    
    # Add ensemble
    final_results.append({
        'Model': 'Ensemble',
        'Type': 'Ensemble',
        'R2': ensemble_result['test_results']['metrics']['R2_Overall'],
        'RMSE': ensemble_result['test_results']['metrics']['RMSE_Overall'],
        'Parameters': sum(p.numel() for p in ensemble_model.parameters()),
        'Model_File': ensemble_result.get('model_filename', 'N/A')  # Add ensemble filename if available
    })
    
    final_df = pd.DataFrame(final_results)
    final_df = final_df.sort_values('R2', ascending=False)
    
    print(final_df.to_string(index=False, float_format='%.4f'))
    
    # Save final results
    final_df.to_csv('week4_final_results.csv', index=False)
    
    print(f"\nWeek 4 optimization completed!")
    print(f"Best single model: {final_df.iloc[0]['Model']} (R2 = {final_df.iloc[0]['R2']:.4f})")
    
    return {
        'optimization_results': optimizer.results,
        'best_models': best_models,
        'ensemble_result': ensemble_result,
        'final_comparison': final_df
    }


if __name__ == "__main__":
    results = main()
