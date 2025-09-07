"""
Week 2: Feature Engineering Module
==================================

This module contains feature engineering utilities, data preprocessing,
and sequence generation tools for time series forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any, Optional

class SequenceGenerator:
    """Generate sequences for time series forecasting"""
    
    def __init__(self, lookback_window: int = 144, forecast_horizon: int = 1):
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training/testing"""
        if target is None:
            target = data[:, -1]  # Use last column as target
        
        X, y = [], []
        
        for i in range(len(data) - self.lookback_window - self.forecast_horizon + 1):
            # Input sequence
            sequence = data[i:i + self.lookback_window]
            X.append(sequence)
            
            # Target (forecast horizon steps ahead)
            target_idx = i + self.lookback_window + self.forecast_horizon - 1
            y.append(target[target_idx])
        
        return np.array(X), np.array(y)
    
    def validate_sequence_alignment(self, X: np.ndarray, y: np.ndarray, 
                                   original_data: np.ndarray) -> Dict[str, Any]:
        """Validate that sequences are properly aligned"""
        validation_results = {
            "sequence_shape": X.shape,
            "target_shape": y.shape,
            "lookback_window": self.lookback_window,
            "forecast_horizon": self.forecast_horizon,
            "alignment_check": True,
            "sample_validation": {}
        }
        
        # Check a few samples for alignment
        for i in [0, len(X)//2, -1]:
            if i == -1:
                i = len(X) - 1
            
            # Get the sequence
            sequence = X[i]
            target = y[i]
            
            # Calculate original indices
            start_idx = i
            end_idx = i + self.lookback_window
            target_idx = end_idx + self.forecast_horizon - 1
            
            # Validate alignment
            expected_sequence = original_data[start_idx:end_idx]
            expected_target = original_data[target_idx, -1] if len(original_data.shape) > 1 else original_data[target_idx]
            
            is_aligned = (
                np.allclose(sequence, expected_sequence, rtol=1e-5) and
                np.allclose(target, expected_target, rtol=1e-5)
            )
            
            validation_results["sample_validation"][f"sample_{i}"] = {
                "aligned": is_aligned,
                "sequence_start": start_idx,
                "sequence_end": end_idx,
                "target_index": target_idx,
                "target_value": float(target),
                "expected_target": float(expected_target)
            }
            
            if not is_aligned:
                validation_results["alignment_check"] = False
        
        return validation_results


class FeatureScaler:
    """Feature scaling utilities for time series data"""
    
    def __init__(self, method: str = 'standard'):
        self.method = method
        self.scalers = {}
        self.is_fitted = False
    
    def fit_transform(self, data: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """Fit scalers and transform data"""
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data.shape[1])]
        
        scaled_data = np.zeros_like(data)
        
        for i, feature_name in enumerate(feature_names):
            # Create scaler for each feature
            if self.method == 'standard':
                scaler = StandardScaler()
            elif self.method == 'minmax':
                scaler = MinMaxScaler()
            elif self.method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.method}")
            
            # Fit and transform
            scaled_data[:, i] = scaler.fit_transform(data[:, i].reshape(-1, 1)).flatten()
            self.scalers[feature_name] = scaler
        
        self.is_fitted = True
        return scaled_data
    
    def transform(self, data: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """Transform data using fitted scalers"""
        if not self.is_fitted:
            raise ValueError("Scalers not fitted. Call fit_transform first.")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data.shape[1])]
        
        scaled_data = np.zeros_like(data)
        
        for i, feature_name in enumerate(feature_names):
            if feature_name in self.scalers:
                scaled_data[:, i] = self.scalers[feature_name].transform(data[:, i].reshape(-1, 1)).flatten()
            else:
                raise ValueError(f"Scaler for feature '{feature_name}' not found")
        
        return scaled_data
    
    def inverse_transform(self, data: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """Inverse transform scaled data"""
        if not self.is_fitted:
            raise ValueError("Scalers not fitted. Call fit_transform first.")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data.shape[1])]
        
        original_data = np.zeros_like(data)
        
        for i, feature_name in enumerate(feature_names):
            if feature_name in self.scalers:
                original_data[:, i] = self.scalers[feature_name].inverse_transform(data[:, i].reshape(-1, 1)).flatten()
            else:
                raise ValueError(f"Scaler for feature '{feature_name}' not found")
        
        return original_data
    
    def get_scaling_stats(self) -> Dict[str, Dict[str, float]]:
        """Get scaling statistics for each feature"""
        stats = {}
        
        for feature_name, scaler in self.scalers.items():
            if self.method == 'standard':
                stats[feature_name] = {
                    'mean': float(scaler.mean_[0]),
                    'std': float(scaler.scale_[0])
                }
            elif self.method == 'minmax':
                stats[feature_name] = {
                    'min': float(scaler.data_min_[0]),
                    'max': float(scaler.data_max_[0]),
                    'scale': float(scaler.scale_[0])
                }
            elif self.method == 'robust':
                stats[feature_name] = {
                    'median': float(scaler.center_[0]),
                    'scale': float(scaler.scale_[0])
                }
        
        return stats


class DataSplitter:
    """Time series aware data splitting utilities"""
    
    @staticmethod
    def temporal_split(data: np.ndarray, train_ratio: float = 0.7, 
                      val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data temporally (preserving time order)"""
        n_samples = len(data)
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return train_data, val_data, test_data
    
    @staticmethod
    def sequence_split(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7,
                      val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split sequences temporally"""
        n_samples = len(X)
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def get_split_info(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Dict[str, Any]:
        """Get information about data splits"""
        total_samples = len(X_train) + len(X_val) + len(X_test)
        
        return {
            "total_samples": total_samples,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "train_ratio": len(X_train) / total_samples,
            "val_ratio": len(X_val) / total_samples,
            "test_ratio": len(X_test) / total_samples,
            "train_shape": X_train.shape,
            "val_shape": X_val.shape,
            "test_shape": X_test.shape
        }


class PreprocessingPipeline:
    """Complete preprocessing pipeline for time series data"""
    
    def __init__(self, lookback_window: int = 144, forecast_horizon: int = 1, 
                 scaling_method: str = 'standard'):
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.scaling_method = scaling_method
        
        self.sequence_generator = SequenceGenerator(lookback_window, forecast_horizon)
        self.feature_scaler = FeatureScaler(scaling_method)
        self.is_fitted = False
        self.pipeline_stats = {}
    
    def fit_transform(self, data: np.ndarray, target_column: int = -1,
                     feature_names: List[str] = None) -> Dict[str, np.ndarray]:
        """Complete preprocessing pipeline"""
        print("🔄 Starting preprocessing pipeline...")
        
        # 1. Scale features
        print("   Scaling features...")
        scaled_data = self.feature_scaler.fit_transform(data, feature_names)
        
        # 2. Create sequences
        print("   Creating sequences...")
        X, y = self.sequence_generator.create_sequences(scaled_data, scaled_data[:, target_column])
        
        # 3. Split data
        print("   Splitting data temporally...")
        X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.sequence_split(X, y)
        
        # 4. Validate alignment
        print("   Validating sequence alignment...")
        validation = self.sequence_generator.validate_sequence_alignment(X, y, scaled_data)
        
        # Store pipeline statistics
        self.pipeline_stats = {
            "scaling_stats": self.feature_scaler.get_scaling_stats(),
            "split_info": DataSplitter.get_split_info(X_train, X_val, X_test),
            "sequence_validation": validation,
            "lookback_window": self.lookback_window,
            "forecast_horizon": self.forecast_horizon,
            "scaling_method": self.scaling_method
        }
        
        self.is_fitted = True
        print("✅ Preprocessing pipeline completed!")
        
        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "scaled_data": scaled_data,
            "original_data": data
        }
    
    def get_pipeline_summary(self) -> str:
        """Generate preprocessing pipeline summary"""
        if not self.is_fitted:
            return "Pipeline not fitted yet"
        
        stats = self.pipeline_stats
        
        summary = []
        summary.append("=" * 60)
        summary.append("PREPROCESSING PIPELINE SUMMARY")
        summary.append("=" * 60)
        
        # Sequence settings
        summary.append(f"\n🔧 SEQUENCE CONFIGURATION:")
        summary.append(f"   Lookback Window: {stats['lookback_window']} steps")
        summary.append(f"   Forecast Horizon: {stats['forecast_horizon']} step(s)")
        summary.append(f"   Scaling Method: {stats['scaling_method']}")
        
        # Data splits
        split_info = stats['split_info']
        summary.append(f"\n📊 DATA SPLITS:")
        summary.append(f"   Total Samples: {split_info['total_samples']:,}")
        summary.append(f"   Train: {split_info['train_samples']:,} ({split_info['train_ratio']:.1%})")
        summary.append(f"   Validation: {split_info['val_samples']:,} ({split_info['val_ratio']:.1%})")
        summary.append(f"   Test: {split_info['test_samples']:,} ({split_info['test_ratio']:.1%})")
        
        # Sequence validation
        validation = stats['sequence_validation']
        summary.append(f"\n✅ SEQUENCE VALIDATION:")
        summary.append(f"   Alignment Check: {'PASSED' if validation['alignment_check'] else 'FAILED'}")
        summary.append(f"   Sequence Shape: {validation['sequence_shape']}")
        summary.append(f"   Target Shape: {validation['target_shape']}")
        
        # Scaling statistics
        scaling_stats = stats['scaling_stats']
        summary.append(f"\n📏 SCALING STATISTICS:")
        for feature, stat in scaling_stats.items():
            if stats['scaling_method'] == 'standard':
                summary.append(f"   {feature}: μ={stat['mean']:.3f}, σ={stat['std']:.3f}")
            elif stats['scaling_method'] == 'minmax':
                summary.append(f"   {feature}: min={stat['min']:.3f}, max={stat['max']:.3f}")
            elif stats['scaling_method'] == 'robust':
                summary.append(f"   {feature}: median={stat['median']:.3f}, scale={stat['scale']:.3f}")
        
        return "\n".join(summary)


def analyze_lookback_windows(data: np.ndarray, target_column: int = -1,
                           windows: List[int] = None) -> Dict[int, Dict[str, float]]:
    """Analyze different lookback window sizes"""
    if windows is None:
        windows = [24, 48, 72, 96, 144, 168]  # 1 day to 1 week in hours
    
    results = {}
    
    print("🔍 Analyzing lookback window sizes...")
    
    for window in windows:
        print(f"   Testing window size: {window}")
        
        # Create sequences
        seq_gen = SequenceGenerator(lookback_window=window)
        X, y = seq_gen.create_sequences(data, data[:, target_column])
        
        if len(X) < 100:  # Skip if too few samples
            continue
        
        # Calculate metrics
        target_var = np.var(y)
        sequence_var = np.mean([np.var(seq) for seq in X])
        coverage = len(X) / (len(data) - window)
        
        results[window] = {
            "samples_generated": len(X),
            "target_variance": target_var,
            "sequence_variance": sequence_var,
            "data_coverage": coverage,
            "efficiency_score": coverage * (1 / (1 + abs(target_var - sequence_var) / target_var))
        }
    
    return results


def create_sample_training_data(n_samples: int = 1000, seq_length: int = 144, 
                              n_features: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Create sample training data for testing"""
    np.random.seed(42)
    
    # Create synthetic time series with patterns
    time_steps = np.arange(n_samples + seq_length)
    
    # Base features with different patterns
    features = []
    
    for i in range(n_features):
        # Different seasonal patterns for each feature
        seasonal = np.sin(2 * np.pi * time_steps / (24 + i * 12)) * (10 + i * 5)
        trend = time_steps * (0.1 + i * 0.05)
        noise = np.random.normal(0, 2 + i, len(time_steps))
        
        feature = 50 + i * 10 + seasonal + trend + noise
        features.append(feature)
    
    # Stack features
    data = np.column_stack(features)
    
    # Create sequences
    seq_gen = SequenceGenerator(lookback_window=seq_length)
    X, y = seq_gen.create_sequences(data, data[:, -1])  # Use last feature as target
    
    return X, y
