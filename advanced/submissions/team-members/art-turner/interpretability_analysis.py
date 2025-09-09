"""
Model Interpretability Analysis for Power Forecasting Models
Provides SHAP analysis, attention visualization, and saliency mapping
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP, provide fallback if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

from models import LSTMBaseline, GRUAlternative, TemporalConvNet
from advanced_models import AttentionLSTM


class ModelInterpreter:
    """Comprehensive model interpretability analysis"""
    
    def __init__(self, model, feature_scaler, target_scaler, feature_names, target_names):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.feature_names = feature_names
        self.target_names = target_names
        self.model.eval()
        
    def gradient_saliency(self, input_sequences, target_zone=0):
        """
        Compute gradient-based saliency maps
        Shows which input features/timesteps are most important
        """
        self.model.eval()
        saliency_maps = []
        
        for seq in input_sequences:
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).requires_grad_(True)
            
            # Forward pass
            output = self.model(seq_tensor)
            
            # Compute gradient w.r.t. target zone
            target_output = output[0, target_zone]
            target_output.backward()
            
            # Get gradients
            gradients = seq_tensor.grad.data.abs()
            saliency_maps.append(gradients.squeeze().numpy())
        
        return np.array(saliency_maps)
    
    def attention_analysis(self, input_sequences):
        """
        Extract attention weights if model has attention mechanism
        """
        if not hasattr(self.model, 'attention'):
            return None
            
        self.model.eval()
        attention_weights = []
        
        with torch.no_grad():
            for seq in input_sequences:
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
                
                # Forward pass and extract attention
                output = self.model(seq_tensor)
                if hasattr(self.model, 'last_attention_weights'):
                    attention_weights.append(
                        self.model.last_attention_weights.squeeze().numpy()
                    )
        
        return np.array(attention_weights) if attention_weights else None
    
    def temporal_importance_analysis(self, input_sequences, n_samples=100):
        """
        Analyze importance of different temporal positions
        """
        # Sample sequences for analysis
        indices = np.random.choice(len(input_sequences), min(n_samples, len(input_sequences)), replace=False)
        sample_sequences = input_sequences[indices]
        
        # Compute saliency for each zone
        temporal_importance = {}
        
        for zone_idx, zone_name in enumerate(self.target_names):
            saliency_maps = self.gradient_saliency(sample_sequences, target_zone=zone_idx)
            
            # Average across samples and features to get temporal importance
            temporal_scores = np.mean(saliency_maps, axis=(0, 2))  # Average over samples and features
            temporal_importance[zone_name] = temporal_scores
        
        return temporal_importance
    
    def feature_importance_analysis(self, input_sequences, n_samples=100):
        """
        Analyze importance of different input features
        """
        indices = np.random.choice(len(input_sequences), min(n_samples, len(input_sequences)), replace=False)
        sample_sequences = input_sequences[indices]
        
        feature_importance = {}
        
        for zone_idx, zone_name in enumerate(self.target_names):
            saliency_maps = self.gradient_saliency(sample_sequences, target_zone=zone_idx)
            
            # Average across samples and time to get feature importance
            feature_scores = np.mean(saliency_maps, axis=(0, 1))  # Average over samples and time
            feature_importance[zone_name] = dict(zip(self.feature_names, feature_scores))
        
        return feature_importance
    
    def shap_analysis(self, background_sequences, test_sequences, n_samples=50):
        """
        SHAP analysis for model interpretability
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Skipping SHAP analysis.")
            return None
        
        # Prepare data
        background_data = background_sequences[:min(100, len(background_sequences))]
        test_data = test_sequences[:min(n_samples, len(test_sequences))]
        
        # Create wrapper function for SHAP
        def model_predict(x):
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x)
                return self.model(x_tensor).numpy()
        
        # Initialize SHAP explainer
        explainer = shap.DeepExplainer(self.model, torch.FloatTensor(background_data))
        
        # Compute SHAP values
        shap_values = explainer.shap_values(torch.FloatTensor(test_data))
        
        return {
            'shap_values': shap_values,
            'background_data': background_data,
            'test_data': test_data
        }
    
    def plot_temporal_importance(self, temporal_importance, save_path=None):
        """
        Plot temporal importance across time steps
        """
        fig, axes = plt.subplots(len(temporal_importance), 1, 
                               figsize=(12, 4 * len(temporal_importance)))
        
        if len(temporal_importance) == 1:
            axes = [axes]
        
        for i, (zone_name, importance) in enumerate(temporal_importance.items()):
            axes[i].plot(importance, marker='o', linewidth=2)
            axes[i].set_title(f'Temporal Importance - {zone_name}')
            axes[i].set_xlabel('Time Step (past → future)')
            axes[i].set_ylabel('Importance Score')
            axes[i].grid(True, alpha=0.3)
            
            # Highlight most important time steps
            top_indices = np.argsort(importance)[-3:]
            axes[i].scatter(top_indices, importance[top_indices], 
                          color='red', s=100, alpha=0.7, zorder=5)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_importance, save_path=None):
        """
        Plot feature importance for each target zone
        """
        n_zones = len(feature_importance)
        fig, axes = plt.subplots(1, n_zones, figsize=(6 * n_zones, 8))
        
        if n_zones == 1:
            axes = [axes]
        
        for i, (zone_name, importance_dict) in enumerate(feature_importance.items()):
            features = list(importance_dict.keys())
            scores = list(importance_dict.values())
            
            # Sort by importance
            sorted_indices = np.argsort(scores)
            features_sorted = [features[i] for i in sorted_indices]
            scores_sorted = [scores[i] for i in sorted_indices]
            
            # Create horizontal bar plot
            bars = axes[i].barh(range(len(features_sorted)), scores_sorted)
            axes[i].set_yticks(range(len(features_sorted)))
            axes[i].set_yticklabels(features_sorted)
            axes[i].set_xlabel('Importance Score')
            axes[i].set_title(f'Feature Importance - {zone_name}')
            axes[i].grid(True, alpha=0.3, axis='x')
            
            # Color bars by importance
            colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_saliency_heatmap(self, input_sequences, n_samples=10, save_path=None):
        """
        Plot saliency heatmaps showing importance across time and features
        """
        # Sample sequences
        indices = np.random.choice(len(input_sequences), min(n_samples, len(input_sequences)), replace=False)
        sample_sequences = input_sequences[indices]
        
        fig, axes = plt.subplots(len(self.target_names), 1, 
                               figsize=(15, 5 * len(self.target_names)))
        
        if len(self.target_names) == 1:
            axes = [axes]
        
        for zone_idx, zone_name in enumerate(self.target_names):
            saliency_maps = self.gradient_saliency(sample_sequences, target_zone=zone_idx)
            
            # Average saliency across samples
            avg_saliency = np.mean(saliency_maps, axis=0)
            
            # Create heatmap
            sns.heatmap(avg_saliency.T, 
                       xticklabels=[f'T-{i}' for i in range(len(avg_saliency))],
                       yticklabels=self.feature_names,
                       cmap='viridis',
                       ax=axes[zone_idx],
                       cbar_kws={'label': 'Saliency Score'})
            
            axes[zone_idx].set_title(f'Saliency Heatmap - {zone_name}')
            axes[zone_idx].set_xlabel('Time Steps (past → future)')
            axes[zone_idx].set_ylabel('Features')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ResidualAnalyzer:
    """Analyze model residuals and error patterns"""
    
    def __init__(self, model, feature_scaler, target_scaler, target_names):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.target_names = target_names
        
    def compute_residuals(self, sequences, targets):
        """Compute model residuals"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for seq in sequences:
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
                pred = self.model(seq_tensor)
                predictions.append(pred.numpy())
        
        predictions = np.vstack(predictions)
        
        # Denormalize for analysis
        predictions_denorm = self.target_scaler.inverse_transform(predictions)
        targets_denorm = self.target_scaler.inverse_transform(targets)
        
        residuals = targets_denorm - predictions_denorm
        
        return {
            'predictions': predictions_denorm,
            'targets': targets_denorm,
            'residuals': residuals
        }
    
    def analyze_error_patterns(self, residuals_data, timestamps=None):
        """Analyze patterns in model errors"""
        residuals = residuals_data['residuals']
        predictions = residuals_data['predictions']
        targets = residuals_data['targets']
        
        analysis = {}
        
        for i, zone_name in enumerate(self.target_names):
            zone_residuals = residuals[:, i]
            zone_predictions = predictions[:, i]
            zone_targets = targets[:, i]
            
            # Basic error statistics
            rmse = np.sqrt(mean_squared_error(zone_targets, zone_predictions))
            mae = mean_absolute_error(zone_targets, zone_predictions)
            r2 = r2_score(zone_targets, zone_predictions)
            
            # Error distribution analysis
            error_stats = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'residual_mean': np.mean(zone_residuals),
                'residual_std': np.std(zone_residuals),
                'residual_skew': float(pd.Series(zone_residuals).skew()),
                'residual_kurtosis': float(pd.Series(zone_residuals).kurtosis())
            }
            
            # Error quantiles
            error_quantiles = np.percentile(np.abs(zone_residuals), [25, 50, 75, 90, 95, 99])
            error_stats['error_quantiles'] = dict(zip([25, 50, 75, 90, 95, 99], error_quantiles))
            
            analysis[zone_name] = error_stats
        
        return analysis
    
    def plot_residual_analysis(self, residuals_data, save_path=None):
        """Create comprehensive residual plots"""
        residuals = residuals_data['residuals']
        predictions = residuals_data['predictions']
        targets = residuals_data['targets']
        
        n_zones = len(self.target_names)
        fig, axes = plt.subplots(n_zones, 3, figsize=(18, 6 * n_zones))
        
        if n_zones == 1:
            axes = axes.reshape(1, -1)
        
        for i, zone_name in enumerate(self.target_names):
            zone_residuals = residuals[:, i]
            zone_predictions = predictions[:, i]
            zone_targets = targets[:, i]
            
            # Residual vs Fitted
            axes[i, 0].scatter(zone_predictions, zone_residuals, alpha=0.6)
            axes[i, 0].axhline(y=0, color='red', linestyle='--')
            axes[i, 0].set_xlabel('Fitted Values')
            axes[i, 0].set_ylabel('Residuals')
            axes[i, 0].set_title(f'Residuals vs Fitted - {zone_name}')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Q-Q Plot (residual distribution)
            from scipy import stats
            stats.probplot(zone_residuals, dist="norm", plot=axes[i, 1])
            axes[i, 1].set_title(f'Q-Q Plot - {zone_name}')
            axes[i, 1].grid(True, alpha=0.3)
            
            # Residual histogram
            axes[i, 2].hist(zone_residuals, bins=30, density=True, alpha=0.7)
            axes[i, 2].set_xlabel('Residuals')
            axes[i, 2].set_ylabel('Density')
            axes[i, 2].set_title(f'Residual Distribution - {zone_name}')
            axes[i, 2].grid(True, alpha=0.3)
            
            # Overlay normal distribution
            x_norm = np.linspace(zone_residuals.min(), zone_residuals.max(), 100)
            y_norm = stats.norm.pdf(x_norm, np.mean(zone_residuals), np.std(zone_residuals))
            axes[i, 2].plot(x_norm, y_norm, 'r-', label='Normal')
            axes[i, 2].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_by_magnitude(self, residuals_data, save_path=None):
        """Plot error patterns by prediction magnitude"""
        residuals = residuals_data['residuals']
        predictions = residuals_data['predictions']
        
        fig, axes = plt.subplots(1, len(self.target_names), 
                               figsize=(6 * len(self.target_names), 5))
        
        if len(self.target_names) == 1:
            axes = [axes]
        
        for i, zone_name in enumerate(self.target_names):
            zone_residuals = np.abs(residuals[:, i])
            zone_predictions = predictions[:, i]
            
            # Create magnitude bins
            n_bins = 10
            bin_edges = np.percentile(zone_predictions, np.linspace(0, 100, n_bins + 1))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Calculate mean absolute error for each bin
            bin_errors = []
            for j in range(n_bins):
                mask = (zone_predictions >= bin_edges[j]) & (zone_predictions < bin_edges[j + 1])
                if mask.sum() > 0:
                    bin_errors.append(np.mean(zone_residuals[mask]))
                else:
                    bin_errors.append(0)
            
            axes[i].bar(range(n_bins), bin_errors, alpha=0.7)
            axes[i].set_xlabel('Prediction Magnitude Bin')
            axes[i].set_ylabel('Mean Absolute Error')
            axes[i].set_title(f'Error by Magnitude - {zone_name}')
            axes[i].set_xticks(range(n_bins))
            axes[i].set_xticklabels([f'{x:.0f}' for x in bin_centers], rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def run_interpretability_analysis(model_path, data_path, output_dir='interpretability_results'):
    """
    Run comprehensive interpretability analysis
    """
    print("RUNNING INTERPRETABILITY ANALYSIS")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load model and data
    print("1. Loading model and data...")
    
    # Load metadata
    with open('dataset_metadata_fixed.json', 'r') as f:
        metadata = json.load(f)
    
    # Load scalers
    with open('feature_scaler.pkl', 'rb') as f:
        feature_scaler = pickle.load(f)
    with open('target_scaler.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    
    # Load test data
    test_sequences = np.load('test_sequences_fixed.npy')
    test_targets = np.load('test_targets_fixed.npy')
    
    # Load model using our checkpoint loader
    from model_loader import load_model_from_checkpoint
    
    model, checkpoint_data = load_model_from_checkpoint(
        model_path=model_path,
        input_size=len(metadata['feature_cols']),
        output_size=len(metadata['target_cols']),
        device='cpu'
    )
    
    print(f"   Model: {type(model).__name__}")
    print(f"   Test sequences: {test_sequences.shape}")
    print(f"   Features: {len(metadata['feature_cols'])}")
    print(f"   Targets: {len(metadata['target_cols'])}")
    
    # Initialize analyzers
    interpreter = ModelInterpreter(
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_names=metadata['feature_cols'],
        target_names=metadata['target_cols']
    )
    
    residual_analyzer = ResidualAnalyzer(
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        target_names=metadata['target_cols']
    )
    
    # Run temporal importance analysis
    print("\n2. Analyzing temporal importance...")
    temporal_importance = interpreter.temporal_importance_analysis(test_sequences)
    interpreter.plot_temporal_importance(
        temporal_importance, 
        save_path=output_path / 'temporal_importance.png'
    )
    
    # Run feature importance analysis
    print("\n3. Analyzing feature importance...")
    feature_importance = interpreter.feature_importance_analysis(test_sequences)
    interpreter.plot_feature_importance(
        feature_importance,
        save_path=output_path / 'feature_importance.png'
    )
    
    # Generate saliency heatmaps
    print("\n4. Generating saliency heatmaps...")
    interpreter.plot_saliency_heatmap(
        test_sequences,
        save_path=output_path / 'saliency_heatmap.png'
    )
    
    # Analyze residuals
    print("\n5. Analyzing residuals...")
    residuals_data = residual_analyzer.compute_residuals(test_sequences, test_targets)
    error_analysis = residual_analyzer.analyze_error_patterns(residuals_data)
    
    # Plot residual analysis
    residual_analyzer.plot_residual_analysis(
        residuals_data,
        save_path=output_path / 'residual_analysis.png'
    )
    
    residual_analyzer.plot_error_by_magnitude(
        residuals_data,
        save_path=output_path / 'error_by_magnitude.png'
    )
    
    # Save analysis results
    print("\n6. Saving results...")
    
    # Save numerical results
    results = {
        'temporal_importance': {k: v.tolist() for k, v in temporal_importance.items()},
        'feature_importance': feature_importance,
        'error_analysis': error_analysis,
        'model_type': type(model).__name__
    }
    
    with open(output_path / 'interpretability_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("INTERPRETABILITY ANALYSIS COMPLETE")
    print("=" * 60)
    
    print("Key Insights:")
    for zone_name in metadata['target_cols']:
        temporal = temporal_importance[zone_name]
        most_important_timestep = np.argmax(temporal)
        
        feature_imp = feature_importance[zone_name]
        most_important_feature = max(feature_imp.items(), key=lambda x: x[1])
        
        error_stats = error_analysis[zone_name]
        
        print(f"\n{zone_name}:")
        print(f"  Most important timestep: T-{most_important_timestep} (score: {temporal[most_important_timestep]:.3f})")
        print(f"  Most important feature: {most_important_feature[0]} (score: {most_important_feature[1]:.3f})")
        print(f"  RMSE: {error_stats['rmse']:.1f}")
        print(f"  R²: {error_stats['r2']:.3f}")
    
    return results


if __name__ == "__main__":
    # Example usage - you'll need to specify the model path
    print("Please specify model path to run analysis:")
    print("python interpretability_analysis.py")
    print()
    print("Or use in notebook:")
    print("from interpretability_analysis import run_interpretability_analysis")
    print("results = run_interpretability_analysis('best_model.pth', 'data_path')")