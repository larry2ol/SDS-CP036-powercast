"""
Week 3: Training Pipeline with FIXED Target Normalization
Training neural networks with properly normalized targets and denormalization for evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import mlflow
from mlflow_utils import log_test_metrics, log_pytorch_model
import mlflow.pytorch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import json
import pickle
import os
from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

from models import LSTMBaseline, GRUAlternative, TemporalConvNet, create_models, get_model_summary


class MetricsCalculator:
    """Calculate evaluation metrics for forecasting models with denormalization"""
    
    def __init__(self, target_scaler=None):
        self.target_scaler = target_scaler
    
    def denormalize_if_needed(self, y_normalized):
        """Denormalize predictions if scaler is available"""
        if self.target_scaler is not None:
            return self.target_scaler.inverse_transform(y_normalized)
        return y_normalized
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared coefficient"""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def calculate_all_metrics(self, y_true_norm: np.ndarray, y_pred_norm: np.ndarray) -> Dict[str, float]:
        """Calculate all metrics with proper denormalization"""
        # Denormalize for meaningful metrics
        y_true = self.denormalize_if_needed(y_true_norm)
        y_pred = self.denormalize_if_needed(y_pred_norm)
        
        n_outputs = y_true.shape[1] if len(y_true.shape) > 1 else 1
        
        if n_outputs == 1:
            return {
                'RMSE': self.calculate_rmse(y_true, y_pred),
                'MAE': self.calculate_mae(y_true, y_pred),
                'R2': self.calculate_r2(y_true, y_pred),
                'MAPE': self.calculate_mape(y_true, y_pred)
            }
        
        metrics = {}
        zone_names = ['Zone1', 'Zone2', 'Zone3']
        
        # Calculate metrics for each zone
        for i in range(min(n_outputs, 3)):
            zone_name = zone_names[i]
            metrics[f'RMSE_{zone_name}'] = self.calculate_rmse(y_true[:, i], y_pred[:, i])
            metrics[f'MAE_{zone_name}'] = self.calculate_mae(y_true[:, i], y_pred[:, i])
            metrics[f'R2_{zone_name}'] = self.calculate_r2(y_true[:, i], y_pred[:, i])
            metrics[f'MAPE_{zone_name}'] = self.calculate_mape(y_true[:, i], y_pred[:, i])
        
        # Calculate overall metrics (average across zones)
        metrics['RMSE_Overall'] = np.mean([metrics[f'RMSE_{zone}'] for zone in zone_names[:n_outputs]])
        metrics['MAE_Overall'] = np.mean([metrics[f'MAE_{zone}'] for zone in zone_names[:n_outputs]])
        metrics['R2_Overall'] = np.mean([metrics[f'R2_{zone}'] for zone in zone_names[:n_outputs]])
        metrics['MAPE_Overall'] = np.mean([metrics[f'MAPE_{zone}'] for zone in zone_names[:n_outputs]])
        
        return metrics


class Trainer:
    """Training class for neural network models with fixed normalization"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 target_scaler=None,
                 device: str = 'cpu',
                 experiment_name: str = 'power_forecasting_fixed'):
        """
        Initialize trainer with target scaler for proper evaluation
        
        Args:
            model: Neural network model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            target_scaler: Scaler for denormalizing targets during evaluation
            device: Device to train on ('cpu' or 'cuda')
            experiment_name: MLflow experiment name
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.target_scaler = target_scaler
        self.device = device
        self.experiment_name = experiment_name
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Metrics calculator with scaler
        self.metrics_calc = MetricsCalculator(target_scaler)
    
    def train_epoch(self, optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_seq, batch_target in self.train_loader:
            batch_seq = batch_seq.to(self.device)
            batch_target = batch_target.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = self.model(batch_seq)
            loss = criterion(predictions, batch_target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_seq, batch_target in self.val_loader:
                batch_seq = batch_seq.to(self.device)
                batch_target = batch_target.to(self.device)
                
                predictions = self.model(batch_seq)
                loss = criterion(predictions, batch_target)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and targets for metrics
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_target.cpu().numpy())
        
        # Calculate metrics with denormalization
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        metrics = self.metrics_calc.calculate_all_metrics(all_targets, all_predictions)
        
        avg_loss = total_loss / num_batches
        return avg_loss, metrics
    
    def train(self, 
              num_epochs: int = 100,
              learning_rate: float = 0.001,
              criterion: str = 'mae',
              optimizer_type: str = 'adam',
              patience: int = 15,
              log_interval: int = 10) -> Dict:
        """
        Train the model with proper denormalization for evaluation
        """
        
        # Set up loss function
        if criterion.lower() == 'mae':
            criterion_fn = nn.L1Loss()
        else:
            criterion_fn = nn.MSELoss()
        
        # Set up optimizer
        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=max(1, patience//2)
        )
        

        # Start MLflow run with a unique name and capture its run_id
        run_timestamp = getattr(self, 'session_id', datetime.now().strftime('%Y%m%d-%H%M%S'))
        with mlflow.start_run(run_name=f"{self.model.__class__.__name__}-{run_timestamp}") as _run:
            run_id = _run.info.run_id
            # Group runs by session (if provided from main())
            mlflow.set_tag("session_id", getattr(self, "session_id", "n/a"))

            # Log parameters
            mlflow.log_param("model_type", self.model.__class__.__name__)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("criterion", criterion)
            mlflow.log_param("optimizer", optimizer_type)
            mlflow.log_param("patience", patience)
            mlflow.log_param("target_normalized", True)  # NEW FLAG
            
            # Log model architecture info
            model_summary = get_model_summary(self.model)
            mlflow.log_param("total_parameters", model_summary['total_parameters'])
            
            # Training loop
            epochs_without_improvement = 0
            
            print(f"Training {self.model.__class__.__name__} (FIXED)")
            print(f"Parameters: {model_summary['trainable_params']}")
            print("="*50)
            
            for epoch in range(num_epochs):
                # Train epoch
                train_loss = self.train_epoch(optimizer, criterion_fn)
                self.train_losses.append(train_loss)
                
                # Validate epoch
                val_loss, val_metrics = self.validate_epoch(criterion_fn)
                self.val_losses.append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
#                   self.best_model_state = self.model.state_dict().copy()
                    # deep-copy weights to CPU to avoid aliasing / VRAM retention
                    self.best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                # Logging
                if epoch % log_interval == 0 or epoch == num_epochs - 1:
                    print(f"Epoch {epoch+1:3d}: Train Loss = {train_loss:.4f}, "
                          f"Val Loss = {val_loss:.4f}, "
                          f"Val R² = {val_metrics['R2_Overall']:.3f}")
                
                # MLflow logging
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                for metric_name, metric_value in val_metrics.items():
#                   mlflow.log_metric(f"val_{metric_name.lower()}", metric_value, step=epoch)
                    mlflow.log_metric(f"val_{metric_name.lower()}", float(metric_value), step=epoch)
                # Early stopping
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
            
            # Final evaluation on test set
            test_results = self.evaluate_test_set()

            # Metrics
            log_test_metrics(test_results.get("metrics", {}), prefix="test_")

            # Optional traceability (git tagging removed for Colab compatibility)

            # Save best model as .pth file for analysis
            model_filename = f"best_{self.model.__class__.__name__.lower()}_{run_timestamp}.pth"
            
            # Extract model configuration for reconstruction
            model_config = self._extract_model_config(self.model)
            
            torch.save({
                'model_state_dict': self.best_model_state,
                'model_class': self.model.__class__.__name__,
                'model_config': model_config,
                'test_metrics': test_results['metrics'],
                'best_val_loss': self.best_val_loss,
                'training_config': {
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'criterion': criterion,
                    'optimizer_type': optimizer_type,
                    'patience': patience
                }
            }, model_filename)
            print(f"Best model saved as: {model_filename}")
            
            # Log the model to MLflow (handles name=..., signature, CUDA pip reqs)
            log_pytorch_model(
                model=self.model,
                name="model",                      # replaces artifact_path
                loader=self.test_loader,           # used to build a tiny CPU input_example
                fallback_shape=(1, 36, 11),        # (batch, seq_len, features) for power data
                # registered_model_name="my-registry-name",  # optional
            )
            
            # Log .pth file as artifact
            mlflow.log_artifact(model_filename)
            print(f"\nTraining completed!")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Test R² Overall: {test_results['metrics']['R2_Overall']:.3f}")
            print(f"Test RMSE Overall: {test_results['metrics']['RMSE_Overall']:.1f}")
            
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss,
                'test_results': test_results,
                'model_state': self.best_model_state,
                'mlflow_run_id': run_id,  # <- return the run id so we can safely log artifacts later
                'model_filename': model_filename,  # <- add model filename to results
            }
    
    def evaluate_test_set(self) -> Dict:
        """Evaluate model on test set with denormalization"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_seq, batch_target in self.test_loader:
                batch_seq = batch_seq.to(self.device)
                batch_target = batch_target.to(self.device)
                
                predictions = self.model(batch_seq)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_target.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        metrics = self.metrics_calc.calculate_all_metrics(targets, predictions)
        
        # Also return denormalized values for visualization
        predictions_denorm = self.metrics_calc.denormalize_if_needed(predictions)
        targets_denorm = self.metrics_calc.denormalize_if_needed(targets)
        
        return {
            'predictions': predictions_denorm,  # Denormalized for visualization
            'targets': targets_denorm,          # Denormalized for visualization
            'predictions_norm': predictions,    # Normalized for loss calculation
            'targets_norm': targets,           # Normalized for loss calculation
            'metrics': metrics
        }
    
    def _extract_model_config(self, model):
        """Extract model configuration for checkpoint saving"""
        config = {}
        
        # Get model class name
        model_class_name = model.__class__.__name__
        
        # Extract configuration based on model type
        if hasattr(model, 'hidden_size') and hasattr(model, 'num_layers'):
            # AttentionLSTM
            config = {
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'dropout_rate': getattr(model, 'dropout', 0.2)
            }
        elif hasattr(model, 'hidden_sizes'):
            # LSTM, GRU, BiLSTM variants
            config = {
                'hidden_sizes': model.hidden_sizes,
                'dropout_rate': getattr(model, 'dropout_rate', 0.2)
            }
            if hasattr(model, 'layer_norm'):
                config['layer_norm'] = model.layer_norm
        elif hasattr(model, 'num_channels'):
            # TCN variants
            config = {
                'num_channels': model.num_channels,
                'dropout': getattr(model, 'dropout', 0.2)
            }
            if hasattr(model, 'use_attention'):
                config['use_attention'] = model.use_attention
        
        return config


class Visualizer:
    """Visualization utilities for model results"""
    
    @staticmethod
    def plot_training_history(train_losses: List[float], 
                            val_losses: List[float], 
                            save_path: Optional[str] = None):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', alpha=0.8)
        plt.plot(val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Normalized Scale)')
        plt.title('Training History - FIXED')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_predictions_vs_actual(targets: np.ndarray, 
                                 predictions: np.ndarray, 
                                 zone_names: List[str] = ['Zone 1', 'Zone 2', 'Zone 3'],
                                 save_path: Optional[str] = None):
        """Plot predictions vs actual for each zone (using denormalized values)"""
        n_zones = min(targets.shape[1], len(zone_names))
        
        fig, axes = plt.subplots(1, n_zones, figsize=(15, 5))
        if n_zones == 1:
            axes = [axes]
        
        for i in range(n_zones):
            axes[i].scatter(targets[:, i], predictions[:, i], alpha=0.6, s=1)
            
            # Perfect prediction line
            min_val = min(targets[:, i].min(), predictions[:, i].min())
            max_val = max(targets[:, i].max(), predictions[:, i].max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            axes[i].set_xlabel('Actual Power (kW)')
            axes[i].set_ylabel('Predicted Power (kW)')
            axes[i].set_title(f'{zone_names[i]} Predictions - FIXED')
            axes[i].grid(True, alpha=0.3)
            
            # Calculate R²
            r2 = r2_score(targets[:, i], predictions[:, i])
            axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', 
                        transform=axes[i].transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_time_series_forecast(targets: np.ndarray, 
                                predictions: np.ndarray,
                                n_samples: int = 200,
                                zone_names: List[str] = ['Zone 1', 'Zone 2', 'Zone 3'],
                                save_path: Optional[str] = None):
        """Plot time series forecast for each zone (using denormalized values)"""
        n_zones = min(targets.shape[1], len(zone_names))
        
        fig, axes = plt.subplots(n_zones, 1, figsize=(15, 4*n_zones))
        if n_zones == 1:
            axes = [axes]
        
        # Select subset for visualization
        end_idx = min(n_samples, len(targets))
        time_idx = np.arange(end_idx)
        
        for i in range(n_zones):
            axes[i].plot(time_idx, targets[:end_idx, i], label='Actual', alpha=0.8)
            axes[i].plot(time_idx, predictions[:end_idx, i], label='Predicted', alpha=0.8)
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Power Consumption (kW)')
            axes[i].set_title(f'{zone_names[i]} Forecast - FIXED')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def load_preprocessed_fixed_data() -> Tuple[DataLoader, DataLoader, DataLoader, Dict, object]:
    """Load preprocessed FIXED data from Week 2"""
    try:
        # Load sequences
        train_sequences = np.load('train_sequences_fixed.npy')
        val_sequences = np.load('val_sequences_fixed.npy')
        test_sequences = np.load('test_sequences_fixed.npy')
        train_targets = np.load('train_targets_fixed.npy')
        val_targets = np.load('val_targets_fixed.npy')
        test_targets = np.load('test_targets_fixed.npy')
        
        # Load metadata
        with open('dataset_metadata_fixed.json', 'r') as f:
            metadata = json.load(f)
        
        # Load target scaler for denormalization
        with open('target_scaler.pkl', 'rb') as f:
            target_scaler = pickle.load(f)
        
        # Create datasets and loaders
        from week2_feature_engineering_fixed import PowerConsumptionDataset
        
        batch_size = metadata['batch_size']
        train_dataset = PowerConsumptionDataset(train_sequences, train_targets)
        val_dataset = PowerConsumptionDataset(val_sequences, val_targets)
        test_dataset = PowerConsumptionDataset(test_sequences, test_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Loaded FIXED preprocessed data:")
        print(f"  Train: {len(train_sequences)} sequences")
        print(f"  Val: {len(val_sequences)} sequences")
        print(f"  Test: {len(test_sequences)} sequences")
        print(f"  Input features: {len(metadata['feature_cols'])}")
        print(f"  Output targets: {len(metadata['target_cols'])}")
        print(f"  Target normalized: {metadata.get('target_normalized', False)}")
        
        return train_loader, val_loader, test_loader, metadata, target_scaler
        
    except FileNotFoundError as e:
        print(f"Error loading FIXED preprocessed data: {e}")
        print("Please run week2_feature_engineering_fixed.py first!")
        raise


def main():
    """Main training script with FIXED data"""
    print("WEEK 3: NEURAL NETWORK TRAINING WITH FIXED NORMALIZED TARGETS")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    
    # Create a session id so this whole program invocation is grouped and filenames are unique
    session_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # Load FIXED data
    train_loader, val_loader, test_loader, metadata, target_scaler = load_preprocessed_fixed_data()
    
    input_size = len(metadata['feature_cols'])
    output_size = len(metadata['target_cols'])
    
    # Create models to train
    models_to_train = {
        'LSTM_Medium_Fixed': LSTMBaseline(input_size=input_size, 
                                        hidden_sizes=[128, 64], 
                                        output_size=output_size, 
                                        dropout_rate=0.2),
        'GRU_Medium_Fixed': GRUAlternative(input_size=input_size, 
                                         hidden_sizes=[128, 64], 
                                         output_size=output_size, 
                                         dropout_rate=0.2),
        'TCN_Medium_Fixed': TemporalConvNet(input_size=input_size, 
                                          num_channels=[64, 128, 64], 
                                          output_size=output_size, 
                                          dropout=0.2)
    }
    
    # Training configurations
    training_config = {
        'num_epochs': 100,
        'learning_rate': 0.001,
        'criterion': 'mae',
        'optimizer_type': 'adam',
        'patience': 15
    }
    
    results = {}
    
    # Train each model
    for model_name, model in models_to_train.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # Create trainer with target scaler
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            target_scaler=target_scaler,  # CRITICAL for proper evaluation
            device=device,
            experiment_name=f"power_forecasting_fixed_{model_name.lower()}_refactor"
        )
        # Pass session context to the trainer so the run can be tagged
        trainer.session_id = session_id

        # Train model
        result = trainer.train(**training_config)
        results[model_name] = result
        
        # Visualizations
        print("Creating visualizations...")
        
        # Use the run_id from training to make filenames unique and log into the same run
        run_id = result.get("mlflow_run_id")

        # Training history
        Visualizer.plot_training_history(
            result['train_losses'], 
            result['val_losses'],
            save_path=f'{model_name}_training_history_{run_id}.png'
        )
        # Reopen the same MLflow run to attach artifacts
        with mlflow.start_run(run_id=run_id):
             mlflow.log_artifact(f'{model_name}_training_history_{run_id}.png')

        # Predictions vs actual (using denormalized values)
        test_results = result['test_results']
        Visualizer.plot_predictions_vs_actual(
            test_results['targets'],        # Already denormalized
            test_results['predictions'],    # Already denormalized
            save_path=f'{model_name}_predictions_vs_actual_{run_id}.png'
        )

        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(f'{model_name}_predictions_vs_actual_{run_id}.png')

        # Time series forecast
        Visualizer.plot_time_series_forecast(
            test_results['targets'],
            test_results['predictions'],
            save_path=f'{model_name}_time_series_forecast_{run_id}.png'
        )
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(f'{model_name}_time_series_forecast_{run_id}.png')

    # Compare models
    print(f"\n{'='*60}")
    print("FIXED MODEL COMPARISON")
    print(f"{'='*60}")
    
    comparison_df = []
    for model_name, result in results.items():
        test_metrics = result['test_results']['metrics']
        comparison_df.append({
            'Model': model_name,
            'RMSE': test_metrics['RMSE_Overall'],
            'MAE': test_metrics['MAE_Overall'],
            'R²': test_metrics['R2_Overall'],
            'MAPE': test_metrics['MAPE_Overall']
        })
    
    comparison_df = pd.DataFrame(comparison_df)
    comparison_df = comparison_df.sort_values('R²', ascending=False)
    
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Save comparison with a session-specific filename so it never overwrites
    comp_path = f'model_comparison_fixed_{session_id}.csv'
    comparison_df.to_csv(comp_path, index=False)
    # (Optional) attach it to the last model's run as a convenience
    if results:
        last_run_id = results[list(results.keys())[-1]].get("mlflow_run_id")
        if last_run_id:
            with mlflow.start_run(run_id=last_run_id):
                mlflow.log_artifact(comp_path)
 
    print(f"\nFIXED training completed! Results saved to MLflow and local files (session {session_id}).")

    print("Expected improvement: RNN models should now have positive R² scores!")
    
    return results


if __name__ == "__main__":
    results = main()