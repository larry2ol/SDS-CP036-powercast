"""
Week 1: Exploratory Data Analysis Module
=======================================

This module contains utilities for data exploration, visualization, and initial analysis
for the PowerCast time series forecasting project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

class DataExplorer:
    """Comprehensive data exploration and analysis tools"""
    
    def __init__(self, data: pd.DataFrame = None):
        self.data = data
        self.analysis_results = {}
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(file_path)
            print(f"✅ Data loaded successfully: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def basic_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset"""
        if self.data is None:
            return {"error": "No data loaded"}
        
        info = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "memory_usage": self.data.memory_usage(deep=True).sum(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "duplicate_rows": self.data.duplicated().sum()
        }
        
        self.analysis_results["basic_info"] = info
        return info
    
    def statistical_summary(self) -> pd.DataFrame:
        """Generate comprehensive statistical summary"""
        if self.data is None:
            return None
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        summary = numeric_data.describe()
        
        # Add additional statistics
        summary.loc['variance'] = numeric_data.var()
        summary.loc['skewness'] = numeric_data.skew()
        summary.loc['kurtosis'] = numeric_data.kurtosis()
        
        self.analysis_results["statistical_summary"] = summary
        return summary
    
    def plot_distributions(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot distribution of all numeric variables"""
        if self.data is None:
            print("No data to plot")
            return
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)
        
        if n_cols == 0:
            print("No numeric columns found")
            return
        
        # Calculate subplot dimensions
        n_rows = (n_cols + 2) // 3
        n_subplot_cols = min(3, n_cols)
        
        fig, axes = plt.subplots(n_rows, n_subplot_cols, figsize=figsize)
        if n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(numeric_cols):
            row = i // n_subplot_cols
            col_idx = i % n_subplot_cols
            
            ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
            
            # Plot histogram with KDE
            self.data[col].hist(bins=30, alpha=0.7, ax=ax, density=True)
            self.data[col].plot.kde(ax=ax, color='red', linewidth=2)
            
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_cols, n_rows * n_subplot_cols):
            row = i // n_subplot_cols
            col_idx = i % n_subplot_cols
            if n_rows > 1:
                axes[row, col_idx].set_visible(False)
            elif n_subplot_cols > 1:
                axes[col_idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def correlation_analysis(self, method: str = 'pearson') -> pd.DataFrame:
        """Perform correlation analysis"""
        if self.data is None:
            return None
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr(method=method)
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title(f'Correlation Matrix ({method.capitalize()})')
        plt.tight_layout()
        plt.show()
        
        self.analysis_results["correlation_matrix"] = correlation_matrix
        return correlation_matrix
    
    def detect_outliers(self, method: str = 'iqr') -> Dict[str, List]:
        """Detect outliers using specified method"""
        if self.data is None:
            return {}
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        outliers = {}
        
        for col in numeric_data.columns:
            if method == 'iqr':
                Q1 = numeric_data[col].quantile(0.25)
                Q3 = numeric_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_indices = numeric_data[
                    (numeric_data[col] < lower_bound) | 
                    (numeric_data[col] > upper_bound)
                ].index.tolist()
                
            elif method == 'zscore':
                z_scores = np.abs((numeric_data[col] - numeric_data[col].mean()) / numeric_data[col].std())
                outlier_indices = numeric_data[z_scores > 3].index.tolist()
            
            outliers[col] = outlier_indices
        
        self.analysis_results["outliers"] = outliers
        return outliers
    
    def generate_eda_report(self) -> str:
        """Generate comprehensive EDA report"""
        if self.data is None:
            return "No data available for analysis"
        
        report = []
        report.append("=" * 60)
        report.append("EXPLORATORY DATA ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Basic info
        info = self.basic_info()
        report.append(f"\n📊 DATASET OVERVIEW:")
        report.append(f"Shape: {info['shape']}")
        report.append(f"Columns: {len(info['columns'])}")
        report.append(f"Memory Usage: {info['memory_usage'] / 1024:.1f} KB")
        
        # Missing values
        missing = info['missing_values']
        if any(missing.values()):
            report.append(f"\n⚠️  MISSING VALUES:")
            for col, count in missing.items():
                if count > 0:
                    pct = (count / len(self.data)) * 100
                    report.append(f"  {col}: {count} ({pct:.1f}%)")
        else:
            report.append(f"\n✅ NO MISSING VALUES DETECTED")
        
        # Duplicates
        if info['duplicate_rows'] > 0:
            pct = (info['duplicate_rows'] / len(self.data)) * 100
            report.append(f"\n⚠️  DUPLICATE ROWS: {info['duplicate_rows']} ({pct:.1f}%)")
        else:
            report.append(f"\n✅ NO DUPLICATE ROWS DETECTED")
        
        # Data types
        report.append(f"\n📋 DATA TYPES:")
        for col, dtype in info['dtypes'].items():
            report.append(f"  {col}: {dtype}")
        
        return "\n".join(report)


def create_sample_power_data(n_samples: int = 1000, include_anomalies: bool = True) -> pd.DataFrame:
    """Create sample power consumption data for demonstration"""
    np.random.seed(42)
    
    # Base time series
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='h')
    
    # Seasonal patterns
    hour_pattern = np.sin(2 * np.pi * dates.hour / 24) * 5000
    daily_pattern = np.sin(2 * np.pi * dates.dayofyear / 365) * 3000
    
    # Base consumption with trend
    base_consumption = 25000 + np.arange(n_samples) * 2
    
    # Add noise
    noise = np.random.normal(0, 1000, n_samples)
    
    # Combine patterns
    power_consumption = base_consumption + hour_pattern + daily_pattern + noise
    
    # Add anomalies
    if include_anomalies:
        anomaly_indices = np.random.choice(n_samples, size=n_samples//50, replace=False)
        anomaly_multipliers = np.random.uniform(1.5, 2.0, len(anomaly_indices))
        for i, idx in enumerate(anomaly_indices):
            power_consumption[idx] *= anomaly_multipliers[i]
    
    # Weather data
    temperature = 20 + 15 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 3, n_samples)
    humidity = 50 + 20 * np.sin(2 * np.pi * dates.dayofyear / 365 + np.pi/4) + np.random.normal(0, 5, n_samples)
    wind_speed = np.abs(np.random.normal(10, 5, n_samples))
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'power_consumption': power_consumption,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'hour': dates.hour,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'is_weekend': (dates.dayofweek >= 5).astype(int)
    })
    
    return data


class TimeSeriesAnalyzer:
    """Specialized time series analysis tools"""
    
    def __init__(self, data: pd.DataFrame, time_col: str, value_col: str):
        self.data = data
        self.time_col = time_col
        self.value_col = value_col
    
    def plot_time_series(self, figsize: Tuple[int, int] = (15, 6)) -> None:
        """Plot time series with trend"""
        plt.figure(figsize=figsize)
        plt.plot(self.data[self.time_col], self.data[self.value_col], alpha=0.7)
        plt.title(f'Time Series: {self.value_col}')
        plt.xlabel(self.time_col)
        plt.ylabel(self.value_col)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def seasonal_decomposition(self) -> Dict[str, np.ndarray]:
        """Perform seasonal decomposition"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Set time column as index
            ts_data = self.data.set_index(self.time_col)[self.value_col]
            
            # Perform decomposition
            decomposition = seasonal_decompose(ts_data, model='additive', period=24)
            
            # Plot decomposition
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            decomposition.observed.plot(ax=axes[0], title='Original')
            decomposition.trend.plot(ax=axes[1], title='Trend')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
            decomposition.resid.plot(ax=axes[3], title='Residual')
            
            plt.tight_layout()
            plt.show()
            
            return {
                'trend': decomposition.trend.values,
                'seasonal': decomposition.seasonal.values,
                'residual': decomposition.resid.values
            }
        except ImportError:
            print("⚠️ statsmodels not available for seasonal decomposition")
            return {}
    
    def autocorrelation_analysis(self, max_lags: int = 50) -> None:
        """Plot autocorrelation and partial autocorrelation"""
        try:
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            
            fig, axes = plt.subplots(2, 1, figsize=(15, 8))
            
            plot_acf(self.data[self.value_col].dropna(), lags=max_lags, ax=axes[0])
            axes[0].set_title('Autocorrelation Function (ACF)')
            
            plot_pacf(self.data[self.value_col].dropna(), lags=max_lags, ax=axes[1])
            axes[1].set_title('Partial Autocorrelation Function (PACF)')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("⚠️ statsmodels not available for autocorrelation analysis")
