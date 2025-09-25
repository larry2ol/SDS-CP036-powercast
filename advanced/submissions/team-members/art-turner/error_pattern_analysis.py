"""
Advanced Error Pattern Analysis for Power Forecasting Models
Analyzes residuals, temporal patterns, and systematic biases
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats import jarque_bera, shapiro
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from models import LSTMBaseline, GRUAlternative, TemporalConvNet


class AdvancedErrorAnalyzer:
    """Advanced analysis of model errors and residual patterns"""
    
    def __init__(self, model, feature_scaler, target_scaler, feature_names, target_names):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.feature_names = feature_names
        self.target_names = target_names
        self.model.eval()
        
    def compute_comprehensive_residuals(self, sequences, targets, features=None):
        """Compute residuals with additional context information"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for seq in sequences:
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
                pred = self.model(seq_tensor)
                predictions.append(pred.numpy())
        
        predictions = np.vstack(predictions)
        
        # Denormalize
        predictions_denorm = self.target_scaler.inverse_transform(predictions)
        targets_denorm = self.target_scaler.inverse_transform(targets)
        residuals = targets_denorm - predictions_denorm
        
        # Calculate relative errors
        relative_errors = residuals / (targets_denorm + 1e-8)  # Avoid division by zero
        
        # Calculate absolute percentage errors
        percentage_errors = np.abs(relative_errors) * 100
        
        return {
            'predictions': predictions_denorm,
            'targets': targets_denorm,
            'residuals': residuals,
            'relative_errors': relative_errors,
            'percentage_errors': percentage_errors,
            'absolute_errors': np.abs(residuals)
        }
    
    def temporal_error_analysis(self, residuals_data, window_size=24):
        """Analyze error patterns over time"""
        residuals = residuals_data['residuals']
        n_samples, n_zones = residuals.shape
        
        temporal_patterns = {}
        
        for i, zone_name in enumerate(self.target_names):
            zone_residuals = residuals[:, i]
            zone_errors = residuals_data['absolute_errors'][:, i]
            
            # Rolling statistics
            rolling_mean = pd.Series(zone_residuals).rolling(window=window_size, center=True).mean()
            rolling_std = pd.Series(zone_residuals).rolling(window=window_size, center=True).std()
            rolling_mae = pd.Series(zone_errors).rolling(window=window_size, center=True).mean()
            
            # Detect periods of high/low error
            high_error_threshold = np.percentile(zone_errors, 90)
            low_error_threshold = np.percentile(zone_errors, 10)
            
            high_error_periods = zone_errors > high_error_threshold
            low_error_periods = zone_errors < low_error_threshold
            
            temporal_patterns[zone_name] = {
                'rolling_mean': rolling_mean.values,
                'rolling_std': rolling_std.values,
                'rolling_mae': rolling_mae.values,
                'high_error_periods': high_error_periods,
                'low_error_periods': low_error_periods,
                'high_error_threshold': high_error_threshold,
                'low_error_threshold': low_error_threshold
            }
        
        return temporal_patterns
    
    def magnitude_based_error_analysis(self, residuals_data, n_bins=10):
        """Analyze errors based on prediction magnitude"""
        predictions = residuals_data['predictions']
        absolute_errors = residuals_data['absolute_errors']
        percentage_errors = residuals_data['percentage_errors']
        
        magnitude_analysis = {}
        
        for i, zone_name in enumerate(self.target_names):
            zone_preds = predictions[:, i]
            zone_abs_errors = absolute_errors[:, i]
            zone_pct_errors = percentage_errors[:, i]
            
            # Create magnitude bins
            bin_edges = np.percentile(zone_preds, np.linspace(0, 100, n_bins + 1))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Calculate statistics for each bin
            bin_stats = []
            for j in range(n_bins):
                mask = (zone_preds >= bin_edges[j]) & (zone_preds < bin_edges[j + 1])
                if mask.sum() > 0:
                    bin_stats.append({
                        'bin_center': bin_centers[j],
                        'count': mask.sum(),
                        'mean_abs_error': np.mean(zone_abs_errors[mask]),
                        'std_abs_error': np.std(zone_abs_errors[mask]),
                        'mean_pct_error': np.mean(zone_pct_errors[mask]),
                        'median_abs_error': np.median(zone_abs_errors[mask])
                    })
                else:
                    bin_stats.append({
                        'bin_center': bin_centers[j],
                        'count': 0,
                        'mean_abs_error': 0,
                        'std_abs_error': 0,
                        'mean_pct_error': 0,
                        'median_abs_error': 0
                    })
            
            magnitude_analysis[zone_name] = {
                'bin_edges': bin_edges,
                'bin_stats': bin_stats
            }
        
        return magnitude_analysis
    
    def residual_distribution_analysis(self, residuals_data):
        """Analyze distribution properties of residuals"""
        residuals = residuals_data['residuals']
        
        distribution_analysis = {}
        
        for i, zone_name in enumerate(self.target_names):
            zone_residuals = residuals[:, i]
            
            # Basic statistics
            mean_residual = np.mean(zone_residuals)
            std_residual = np.std(zone_residuals)
            skewness = stats.skew(zone_residuals)
            kurtosis_val = stats.kurtosis(zone_residuals)
            
            # Normality tests
            shapiro_stat, shapiro_p = shapiro(zone_residuals[:5000])  # Limit for computational efficiency
            jb_stat, jb_p = jarque_bera(zone_residuals)
            
            # Quantile analysis
            quantiles = np.percentile(zone_residuals, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            
            # Outlier detection
            q1, q3 = np.percentile(zone_residuals, [25, 75])
            iqr = q3 - q1
            outlier_threshold_lower = q1 - 1.5 * iqr
            outlier_threshold_upper = q3 + 1.5 * iqr
            
            outliers = (zone_residuals < outlier_threshold_lower) | (zone_residuals > outlier_threshold_upper)
            
            distribution_analysis[zone_name] = {
                'mean': mean_residual,
                'std': std_residual,
                'skewness': skewness,
                'kurtosis': kurtosis_val,
                'shapiro_stat': shapiro_stat,
                'shapiro_p_value': shapiro_p,
                'jarque_bera_stat': jb_stat,
                'jarque_bera_p_value': jb_p,
                'quantiles': dict(zip([1, 5, 10, 25, 50, 75, 90, 95, 99], quantiles)),
                'outlier_count': outliers.sum(),
                'outlier_percentage': (outliers.sum() / len(zone_residuals)) * 100
            }
        
        return distribution_analysis
    
    def conditional_error_analysis(self, residuals_data, sequences, feature_indices=None):
        """Analyze errors conditioned on input features"""
        if feature_indices is None:
            # Analyze all features, but focus on environmental ones
            env_features = ['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']
            feature_indices = [i for i, name in enumerate(self.feature_names) if name in env_features]
        
        absolute_errors = residuals_data['absolute_errors']
        
        # Extract features (use last timestep for analysis)
        last_timestep_features = sequences[:, -1, :]  # Last timestep
        
        conditional_analysis = {}
        
        for feat_idx in feature_indices:
            feature_name = self.feature_names[feat_idx]
            feature_values = last_timestep_features[:, feat_idx]
            
            # Denormalize feature values for interpretation
            feature_values_denorm = self.feature_scaler.inverse_transform(
                np.zeros((len(feature_values), len(self.feature_names)))
            )
            # This is a simplification - ideally we'd store feature-specific scalers
            
            conditional_analysis[feature_name] = {}
            
            for i, zone_name in enumerate(self.target_names):
                zone_errors = absolute_errors[:, i]
                
                # Bin feature values
                n_bins = 5
                feature_bins = np.percentile(feature_values, np.linspace(0, 100, n_bins + 1))
                
                bin_errors = []
                for j in range(n_bins):
                    mask = (feature_values >= feature_bins[j]) & (feature_values < feature_bins[j + 1])
                    if mask.sum() > 0:
                        bin_errors.append({
                            'bin_range': (feature_bins[j], feature_bins[j + 1]),
                            'mean_error': np.mean(zone_errors[mask]),
                            'count': mask.sum()
                        })
                
                conditional_analysis[feature_name][zone_name] = bin_errors
        
        return conditional_analysis
    
    def error_clustering_analysis(self, residuals_data, sequences, n_clusters=5):
        """Cluster samples based on error patterns"""
        absolute_errors = residuals_data['absolute_errors']
        
        # Use PCA to reduce dimensionality of error patterns
        if len(self.target_names) > 1:
            pca = PCA(n_components=min(3, len(self.target_names)))
            error_pca = pca.fit_transform(absolute_errors)
        else:
            error_pca = absolute_errors
            pca = None
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(error_pca)
        
        # Analyze each cluster
        cluster_analysis = {}
        
        for cluster_id in range(n_clusters):
            mask = clusters == cluster_id
            cluster_errors = absolute_errors[mask]
            cluster_sequences = sequences[mask]
            
            # Compute statistics for this cluster
            cluster_stats = {}
            for i, zone_name in enumerate(self.target_names):
                zone_cluster_errors = cluster_errors[:, i]
                cluster_stats[zone_name] = {
                    'mean_error': np.mean(zone_cluster_errors),
                    'std_error': np.std(zone_cluster_errors),
                    'median_error': np.median(zone_cluster_errors)
                }
            
            # Analyze input characteristics of this cluster
            cluster_features = cluster_sequences[:, -1, :]  # Last timestep
            feature_stats = {}
            for j, feature_name in enumerate(self.feature_names):
                feature_stats[feature_name] = {
                    'mean': np.mean(cluster_features[:, j]),
                    'std': np.std(cluster_features[:, j])
                }
            
            cluster_analysis[f'Cluster_{cluster_id}'] = {
                'size': mask.sum(),
                'percentage': (mask.sum() / len(clusters)) * 100,
                'error_stats': cluster_stats,
                'feature_stats': feature_stats
            }
        
        return {
            'cluster_analysis': cluster_analysis,
            'clusters': clusters,
            'pca_components': pca.components_ if pca else None,
            'pca_explained_variance': pca.explained_variance_ratio_ if pca else None
        }
    
    def plot_comprehensive_error_analysis(self, residuals_data, temporal_patterns, 
                                        magnitude_analysis, save_dir=None):
        """Create comprehensive error analysis plots"""
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True)
        
        # 1. Temporal error patterns
        fig, axes = plt.subplots(len(self.target_names), 2, figsize=(15, 5 * len(self.target_names)))
        if len(self.target_names) == 1:
            axes = axes.reshape(1, -1)
        
        for i, zone_name in enumerate(self.target_names):
            patterns = temporal_patterns[zone_name]
            
            # Rolling error statistics
            axes[i, 0].plot(patterns['rolling_mae'], label='Rolling MAE', linewidth=2)
            axes[i, 0].fill_between(range(len(patterns['rolling_mae'])), 
                                  patterns['rolling_mae'], alpha=0.3)
            axes[i, 0].set_title(f'Temporal Error Patterns - {zone_name}')
            axes[i, 0].set_xlabel('Time')
            axes[i, 0].set_ylabel('Mean Absolute Error')
            axes[i, 0].grid(True, alpha=0.3)
            
            # High/low error periods
            high_periods = np.where(patterns['high_error_periods'])[0]
            low_periods = np.where(patterns['low_error_periods'])[0]
            
            axes[i, 1].scatter(high_periods, 
                             residuals_data['absolute_errors'][high_periods, i], 
                             c='red', alpha=0.6, label='High Error')
            axes[i, 1].scatter(low_periods, 
                             residuals_data['absolute_errors'][low_periods, i], 
                             c='green', alpha=0.6, label='Low Error')
            axes[i, 1].set_title(f'Error Distribution - {zone_name}')
            axes[i, 1].set_xlabel('Time')
            axes[i, 1].set_ylabel('Absolute Error')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_path / 'temporal_error_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Magnitude-based error analysis
        fig, axes = plt.subplots(1, len(self.target_names), figsize=(6 * len(self.target_names), 5))
        if len(self.target_names) == 1:
            axes = [axes]
        
        for i, zone_name in enumerate(self.target_names):
            analysis = magnitude_analysis[zone_name]
            bin_centers = [stat['bin_center'] for stat in analysis['bin_stats']]
            mean_errors = [stat['mean_abs_error'] for stat in analysis['bin_stats']]
            std_errors = [stat['std_abs_error'] for stat in analysis['bin_stats']]
            
            axes[i].errorbar(bin_centers, mean_errors, yerr=std_errors, 
                           marker='o', capsize=5, capthick=2, linewidth=2)
            axes[i].set_title(f'Error vs Prediction Magnitude - {zone_name}')
            axes[i].set_xlabel('Prediction Magnitude')
            axes[i].set_ylabel('Mean Absolute Error')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_path / 'magnitude_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Residual distribution analysis
        fig, axes = plt.subplots(len(self.target_names), 2, figsize=(12, 4 * len(self.target_names)))
        if len(self.target_names) == 1:
            axes = axes.reshape(1, -1)
        
        for i, zone_name in enumerate(self.target_names):
            zone_residuals = residuals_data['residuals'][:, i]
            
            # Histogram with normal overlay
            axes[i, 0].hist(zone_residuals, bins=50, density=True, alpha=0.7, 
                          color='skyblue', edgecolor='black')
            
            # Overlay normal distribution
            mu, sigma = np.mean(zone_residuals), np.std(zone_residuals)
            x = np.linspace(zone_residuals.min(), zone_residuals.max(), 100)
            axes[i, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                          label=f'Normal(μ={mu:.1f}, σ={sigma:.1f})')
            
            axes[i, 0].set_title(f'Residual Distribution - {zone_name}')
            axes[i, 0].set_xlabel('Residuals')
            axes[i, 0].set_ylabel('Density')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Box plot
            axes[i, 1].boxplot(zone_residuals, vert=True, patch_artist=True,
                             boxprops=dict(facecolor='lightblue'))
            axes[i, 1].set_title(f'Residual Box Plot - {zone_name}')
            axes[i, 1].set_ylabel('Residuals')
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_path / 'residual_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_comprehensive_error_analysis(model_path, output_dir='error_analysis_results'):
    """
    Run comprehensive error pattern analysis
    """
    print("COMPREHENSIVE ERROR PATTERN ANALYSIS")
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
    print(f"   Test samples: {len(test_sequences)}")
    
    # Initialize analyzer
    analyzer = AdvancedErrorAnalyzer(
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_names=metadata['feature_cols'],
        target_names=metadata['target_cols']
    )
    
    # Compute comprehensive residuals
    print("2. Computing comprehensive residuals...")
    residuals_data = analyzer.compute_comprehensive_residuals(test_sequences, test_targets)
    
    # Temporal error analysis
    print("3. Analyzing temporal error patterns...")
    temporal_patterns = analyzer.temporal_error_analysis(residuals_data)
    
    # Magnitude-based error analysis
    print("4. Analyzing magnitude-based errors...")
    magnitude_analysis = analyzer.magnitude_based_error_analysis(residuals_data)
    
    # Distribution analysis
    print("5. Analyzing residual distributions...")
    distribution_analysis = analyzer.residual_distribution_analysis(residuals_data)
    
    # Conditional error analysis
    print("6. Analyzing conditional errors...")
    conditional_analysis = analyzer.conditional_error_analysis(residuals_data, test_sequences)
    
    # Error clustering
    print("7. Performing error clustering analysis...")
    clustering_analysis = analyzer.error_clustering_analysis(residuals_data, test_sequences)
    
    # Generate plots
    print("8. Generating comprehensive plots...")
    analyzer.plot_comprehensive_error_analysis(
        residuals_data, temporal_patterns, magnitude_analysis, save_dir=output_path
    )
    
    # Save results
    print("9. Saving analysis results...")
    results = {
        'model_type': type(model).__name__,
        'distribution_analysis': distribution_analysis,
        'magnitude_analysis': {k: {'bin_stats': v['bin_stats']} for k, v in magnitude_analysis.items()},
        'conditional_analysis': conditional_analysis,
        'clustering_summary': {k: {
            'size': v['size'], 
            'percentage': v['percentage'],
            'error_stats': v['error_stats']
        } for k, v in clustering_analysis['cluster_analysis'].items()}
    }
    
    with open(output_path / 'comprehensive_error_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 60)
    
    print("Key Findings:")
    for zone_name in metadata['target_cols']:
        dist_stats = distribution_analysis[zone_name]
        
        print(f"\n{zone_name}:")
        print(f"  Mean residual: {dist_stats['mean']:.3f} (bias indicator)")
        print(f"  Residual std: {dist_stats['std']:.1f}")
        print(f"  Skewness: {dist_stats['skewness']:.3f} (0 = symmetric)")
        print(f"  Outliers: {dist_stats['outlier_percentage']:.1f}% of predictions")
        
        if dist_stats['shapiro_p_value'] < 0.05:
            print(f"  Distribution: Non-normal (p={dist_stats['shapiro_p_value']:.4f})")
        else:
            print(f"  Distribution: Approximately normal (p={dist_stats['shapiro_p_value']:.4f})")
    
    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    # Example usage
    print("Please specify model path to run analysis:")
    print("python error_pattern_analysis.py")
    print()
    print("Or use in notebook:")
    print("from error_pattern_analysis import run_comprehensive_error_analysis")
    print("results = run_comprehensive_error_analysis('best_model.pth')")