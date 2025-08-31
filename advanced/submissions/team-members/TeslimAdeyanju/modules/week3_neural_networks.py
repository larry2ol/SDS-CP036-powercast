"""
Week 3: Neural Networks Module
==============================

This module contains neural network architectures, training infrastructure,
evaluation tools, and model interpretation utilities for time series forecasting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Neural network functionality will be limited.")


class PowerCastModelBuilder:
    """Neural network model builder for power consumption forecasting"""
    
    def __init__(self, input_shape: Tuple[int, int]):
        self.input_shape = input_shape
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for neural network models")
    
    def build_lstm_model(self, lstm_units: int = 50, dropout_rate: float = 0.2,
                        dense_units: int = 25) -> keras.Model:
        """Build LSTM model for time series forecasting"""
        model = keras.Sequential([
            layers.LSTM(lstm_units, return_sequences=True, input_shape=self.input_shape),
            layers.Dropout(dropout_rate),
            layers.LSTM(lstm_units // 2, return_sequences=False),
            layers.Dropout(dropout_rate),
            layers.Dense(dense_units, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1)
        ], name='LSTM_PowerCast')
        
        return model
    
    def build_gru_model(self, gru_units: int = 50, dropout_rate: float = 0.2,
                       dense_units: int = 25) -> keras.Model:
        """Build GRU model for time series forecasting"""
        model = keras.Sequential([
            layers.GRU(gru_units, return_sequences=True, input_shape=self.input_shape),
            layers.Dropout(dropout_rate),
            layers.GRU(gru_units // 2, return_sequences=False),
            layers.Dropout(dropout_rate),
            layers.Dense(dense_units, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1)
        ], name='GRU_PowerCast')
        
        return model
    
    def build_dense_model(self, hidden_units: List[int] = [128, 64, 32],
                         dropout_rate: float = 0.3) -> keras.Model:
        """Build Dense (feedforward) model for time series forecasting"""
        model = keras.Sequential(name='Dense_PowerCast')
        
        # Flatten input
        model.add(layers.Flatten(input_shape=self.input_shape))
        
        # Hidden layers
        for units in hidden_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1))
        
        return model
    
    def build_cnn_lstm_model(self, cnn_filters: int = 64, kernel_size: int = 3,
                            lstm_units: int = 50, dropout_rate: float = 0.2) -> keras.Model:
        """Build CNN-LSTM hybrid model"""
        model = keras.Sequential([
            layers.Conv1D(filters=cnn_filters, kernel_size=kernel_size, 
                         activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(dropout_rate),
            
            layers.LSTM(lstm_units, return_sequences=False),
            layers.Dropout(dropout_rate),
            layers.Dense(50, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1)
        ], name='CNN_LSTM_PowerCast')
        
        return model
    
    def get_model_summary(self, model: keras.Model) -> Dict[str, Any]:
        """Get detailed model summary"""
        return {
            "name": model.name,
            "total_params": model.count_params(),
            "trainable_params": np.sum([np.prod(v.shape) for v in model.trainable_weights]),
            "layers": len(model.layers),
            "input_shape": self.input_shape,
            "output_shape": model.output_shape
        }


class PowerCastTrainer:
    """Training infrastructure for PowerCast models"""
    
    def __init__(self, model_builder: PowerCastModelBuilder):
        self.model_builder = model_builder
        self.trained_models = {}
        self.training_history = {}
        
    def compile_model(self, model: keras.Model, learning_rate: float = 0.001) -> keras.Model:
        """Compile model with appropriate optimizer and loss"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_callbacks(self, model_name: str, patience: int = 10) -> List[keras.callbacks.Callback]:
        """Create training callbacks"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=f'best_{model_name.lower()}_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, epochs: int = 50,
                   batch_size: int = 32, verbose: int = 1) -> keras.callbacks.History:
        """Train a single model"""
        
        model_name = model.name
        print(f"\n🚀 Training {model_name}...")
        print(f"   Parameters: {model.count_params():,}")
        
        # Compile model
        self.compile_model(model)
        
        # Create callbacks
        callbacks = self.create_callbacks(model_name)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Store results
        self.trained_models[model_name] = model
        self.training_history[model_name] = history
        
        # Print final metrics
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_mae = history.history['mae'][-1]
        final_val_mae = history.history['val_mae'][-1]
        
        print(f"   ✅ Training completed:")
        print(f"      Final Loss: {final_loss:.0f}")
        print(f"      Final Val Loss: {final_val_loss:.0f}")
        print(f"      Final MAE: {final_mae:.0f}")
        print(f"      Final Val MAE: {final_val_mae:.0f}")
        
        return history
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, keras.callbacks.History]:
        """Train all available model architectures"""
        
        print("🏗️ Building and training neural network models...")
        
        # Build models
        models = {
            'LSTM': self.model_builder.build_lstm_model(),
            'GRU': self.model_builder.build_gru_model(),
            'Dense': self.model_builder.build_dense_model()
        }
        
        # Train each model
        all_histories = {}
        
        for model_name, model in models.items():
            history = self.train_model(
                model, X_train, y_train, X_val, y_val,
                epochs=30, batch_size=32
            )
            all_histories[model_name] = history
        
        return all_histories
    
    def plot_training_history(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot training history for all models"""
        
        if not self.training_history:
            print("No training history available")
            return
        
        n_models = len(self.training_history)
        fig, axes = plt.subplots(2, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for i, (model_name, history) in enumerate(self.training_history.items()):
            # Loss plot
            axes[0, i].plot(history.history['loss'], 'b-', label='Training Loss', alpha=0.7)
            axes[0, i].plot(history.history['val_loss'], 'r-', label='Validation Loss', alpha=0.7)
            axes[0, i].set_title(f'{model_name} - Training Loss')
            axes[0, i].set_xlabel('Epoch')
            axes[0, i].set_ylabel('Loss (MSE)')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # MAE plot
            axes[1, i].plot(history.history['mae'], 'b-', label='Training MAE', alpha=0.7)
            axes[1, i].plot(history.history['val_mae'], 'r-', label='Validation MAE', alpha=0.7)
            axes[1, i].set_title(f'{model_name} - Training MAE')
            axes[1, i].set_xlabel('Epoch')
            axes[1, i].set_ylabel('MAE')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_training_summary(self) -> str:
        """Generate training summary"""
        if not self.trained_models:
            return "No models trained yet"
        
        summary = []
        summary.append("=" * 60)
        summary.append("TRAINING SUMMARY")
        summary.append("=" * 60)
        
        for model_name, model in self.trained_models.items():
            history = self.training_history[model_name]
            
            summary.append(f"\n{model_name} Model:")
            summary.append(f"   Parameters: {model.count_params():,}")
            summary.append(f"   Epochs Trained: {len(history.history['loss'])}")
            summary.append(f"   Final Training Loss: {history.history['loss'][-1]:.0f}")
            summary.append(f"   Final Validation Loss: {history.history['val_loss'][-1]:.0f}")
            summary.append(f"   Final Training MAE: {history.history['mae'][-1]:.0f}")
            summary.append(f"   Final Validation MAE: {history.history['val_mae'][-1]:.0f}")
        
        return "\n".join(summary)


class ModelEvaluator:
    """Comprehensive model evaluation for time series forecasting"""
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Simple evaluation method that returns basic metrics"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        # Basic regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        # Basic regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Percentage errors
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Power system specific metrics
        max_error = np.max(np.abs(y_true - y_pred))
        mean_bias = np.mean(y_pred - y_true)
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R²': r2,
            'Max_Error': max_error,
            'Mean_Bias': mean_bias
        }
        
        self.evaluation_results[model_name] = metrics
        return metrics
    
    def evaluate_all_models(self, models: Dict[str, keras.Model], 
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Evaluate all trained models on test data"""
        results = {}
        
        print("Model Evaluation Results")
        print("=" * 60)
        
        for model_name, model in models.items():
            # Make predictions
            y_pred = model.predict(X_test, verbose=0).flatten()
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, model_name)
            results[model_name] = {
                'predictions': y_pred,
                'metrics': metrics
            }
            
            # Print results
            print(f"\n{model_name} Model Performance:")
            print("-" * 30)
            for metric_name, value in metrics.items():
                if metric_name in ['MAE', 'RMSE', 'Max_Error', 'Mean_Bias']:
                    print(f"  {metric_name}: {value:,.1f} kW")
                elif metric_name == 'MAPE':
                    print(f"  {metric_name}: {value:.2f}%")
                elif metric_name == 'R²':
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value:,.0f}")
        
        return results
    
    def plot_predictions_comparison(self, results: Dict[str, Dict[str, Any]], 
                                  y_test: np.ndarray, n_samples: int = 100) -> None:
        """Plot prediction vs actual comparison for all models"""
        n_models = len(results)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 8))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        # Use only first n_samples for clarity
        plot_indices = range(min(n_samples, len(y_test)))
        y_test_plot = y_test[plot_indices]
        
        for i, (model_name, result) in enumerate(results.items()):
            y_pred_plot = result['predictions'][plot_indices]
            
            # Actual vs Predicted time series
            axes[0, i].plot(plot_indices, y_test_plot, 'b-', label='Actual', alpha=0.7)
            axes[0, i].plot(plot_indices, y_pred_plot, 'r-', label='Predicted', alpha=0.7)
            axes[0, i].set_title(f'{model_name} - Time Series Comparison')
            axes[0, i].set_xlabel('Sample Index')
            axes[0, i].set_ylabel('Power Consumption (kW)')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Scatter plot (Actual vs Predicted)
            axes[1, i].scatter(y_test_plot, y_pred_plot, alpha=0.5, s=20)
            
            # Perfect prediction line
            min_val = min(y_test_plot.min(), y_pred_plot.min())
            max_val = max(y_test_plot.max(), y_pred_plot.max())
            axes[1, i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            axes[1, i].set_title(f'{model_name} - Actual vs Predicted')
            axes[1, i].set_xlabel('Actual Power (kW)')
            axes[1, i].set_ylabel('Predicted Power (kW)')
            axes[1, i].grid(True, alpha=0.3)
            
            # Add R² to scatter plot
            r2 = result['metrics']['R²']
            axes[1, i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[1, i].transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def create_metrics_comparison(self) -> pd.DataFrame:
        """Create comprehensive metrics comparison table"""
        if not self.evaluation_results:
            print("No evaluation results available")
            return pd.DataFrame()
        
        # Create comparison DataFrame
        metrics_df = pd.DataFrame(self.evaluation_results).T
        
        # Sort by RMSE (primary metric for time series)
        metrics_df = metrics_df.sort_values('RMSE')
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*80)
        
        # Format and display table
        print(metrics_df.round(2))
        
        # Identify best model for each metric
        print("\n" + "="*80)
        print("BEST PERFORMING MODELS BY METRIC")
        print("="*80)
        
        for metric in metrics_df.columns:
            if metric in ['MAE', 'MSE', 'RMSE', 'MAPE', 'Max_Error']:
                best_model = metrics_df[metric].idxmin()
                best_value = metrics_df.loc[best_model, metric]
            else:  # Higher is better (R², etc.)
                best_model = metrics_df[metric].idxmax()
                best_value = metrics_df.loc[best_model, metric]
            
            if metric in ['MAPE']:
                print(f"{metric:12}: {best_model:8} ({best_value:.2f}%)")
            elif metric in ['R²']:
                print(f"{metric:12}: {best_model:8} ({best_value:.4f})")
            else:
                print(f"{metric:12}: {best_model:8} ({best_value:,.1f})")
        
        return metrics_df


class ModelInterpreter:
    """Model interpretation and analysis for neural networks"""
    
    def __init__(self, models: Dict[str, keras.Model], feature_names: List[str] = None):
        self.models = models
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(7)]
        
    def analyze_model_architecture(self) -> None:
        """Analyze and compare model architectures"""
        
        print("🔍 MODEL ARCHITECTURE ANALYSIS")
        print("=" * 60)
        
        for model_name, model in self.models.items():
            print(f"\n{model_name} Model Architecture:")
            print("-" * 30)
            
            # Count parameters
            total_params = model.count_params()
            trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
            
            print(f"Total Parameters: {total_params:,}")
            print(f"Trainable Parameters: {trainable_params:,}")
            print(f"Model Depth: {len(model.layers)} layers")
            
            # Layer analysis
            print("Layer Structure:")
            for i, layer in enumerate(model.layers):
                layer_type = type(layer).__name__
                try:
                    units = layer.units if hasattr(layer, 'units') else 'N/A'
                    print(f"  {i+1}. {layer_type}: {units} units")
                except:
                    print(f"  {i+1}. {layer_type}")
    
    def calculate_prediction_confidence(self, X_sample: np.ndarray, n_predictions: int = 10) -> Dict[str, Dict[str, float]]:
        """Calculate prediction confidence using Monte Carlo approach"""
        
        print("\n🎯 PREDICTION CONFIDENCE ANALYSIS")
        print("=" * 60)
        
        # Use first sample for analysis
        single_sample = X_sample[:1]
        confidence_results = {}
        
        for model_name, model in self.models.items():
            predictions = []
            
            # Make multiple predictions
            for _ in range(n_predictions):
                pred = model.predict(single_sample, verbose=0)
                predictions.append(pred[0, 0])
            
            predictions = np.array(predictions)
            
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            cv = (std_pred / np.maximum(np.abs(mean_pred), 1e-8)) * 100
            
            confidence_results[model_name] = {
                'mean': mean_pred,
                'std': std_pred,
                'cv': cv,
                'confidence_interval': (mean_pred - 1.96*std_pred, mean_pred + 1.96*std_pred)
            }
            
            print(f"\n{model_name} Prediction Confidence:")
            print(f"  Mean Prediction: {mean_pred:,.1f} kW")
            print(f"  Std Deviation: {std_pred:.1f} kW")
            print(f"  Coefficient of Variation: {cv:.2f}%")
            print(f"  95% Confidence Interval: [{mean_pred - 1.96*std_pred:,.1f}, {mean_pred + 1.96*std_pred:,.1f}] kW")
        
        return confidence_results
    
    def generate_insights_summary(self) -> None:
        """Generate comprehensive insights and recommendations"""
        
        print("\n" + "="*80)
        print("💡 MODEL INSIGHTS & INTERPRETATION SUMMARY")
        print("="*80)
        
        print("\n🏗️ ARCHITECTURE INSIGHTS:")
        print("-" * 50)
        print("• Dense Model: Simple feedforward architecture with strong baseline performance")
        print("• LSTM Model: Recurrent architecture designed for temporal dependencies")
        print("• GRU Model: Simplified recurrent unit with faster training")
        print("• Parameter Efficiency: Analyze parameter count vs. performance trade-offs")
        
        print("\n📊 PERFORMANCE INSIGHTS:")
        print("-" * 50)
        print("• Model comparison reveals effectiveness of different architectures")
        print("• Training convergence patterns indicate proper optimization")
        print("• Evaluation metrics provide comprehensive performance assessment")
        print("• Error analysis helps identify areas for improvement")
        
        print("\n🚀 RECOMMENDATIONS FOR IMPROVEMENT:")
        print("-" * 50)
        print("1. 📈 Advanced Feature Engineering")
        print("   • Weather data integration")
        print("   • Calendar effects (holidays, seasons)")
        print("   • Lag features and rolling statistics")
        
        print("\n2. 🎛️ Model Optimization")
        print("   • Hyperparameter tuning")
        print("   • Ensemble methods")
        print("   • Advanced architectures (Attention, Transformer)")
        
        print("\n3. 📋 Data Quality Enhancement")
        print("   • Real power consumption data")
        print("   • Data cleaning and outlier handling")
        print("   • Multiple data sources integration")
