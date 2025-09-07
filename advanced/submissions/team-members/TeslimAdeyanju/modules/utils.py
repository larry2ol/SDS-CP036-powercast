"""
Common Utilities Module
======================

This module contains common utilities, helper functions, and shared tools
used across different weeks of the PowerCast project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def setup_matplotlib():
    """Setup matplotlib with consistent styling"""
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['figure.facecolor'] = 'white'

def print_section_header(title: str, level: int = 1) -> None:
    """Print formatted section headers"""
    if level == 1:
        print("\n" + "=" * 80)
        print(f"{title}")
        print("=" * 80)
    elif level == 2:
        print(f"\n{title}")
        print("-" * len(title))
    else:
        print(f"\n{title}:")

def print_success(message: str) -> None:
    """Print success message with formatting"""
    print(f"✅ {message}")

def print_warning(message: str) -> None:
    """Print warning message with formatting"""
    print(f"⚠️ {message}")

def print_error(message: str) -> None:
    """Print error message with formatting"""
    print(f"❌ {message}")

def print_info(message: str) -> None:
    """Print info message with formatting"""
    print(f"ℹ️ {message}")

def format_number(number: float, decimals: int = 2) -> str:
    """Format numbers with thousands separators"""
    if abs(number) >= 1000:
        return f"{number:,.{decimals}f}"
    else:
        return f"{number:.{decimals}f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage values"""
    return f"{value:.{decimals}f}%"

def create_progress_indicator(current: int, total: int, width: int = 50) -> str:
    """Create a progress bar string"""
    progress = current / total
    filled_width = int(width * progress)
    bar = "█" * filled_width + "░" * (width - filled_width)
    percentage = progress * 100
    return f"[{bar}] {percentage:.1f}% ({current}/{total})"

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def check_missing_values(data: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values in DataFrame"""
        missing_info = {}
        missing_count = data.isnull().sum()
        missing_percentage = (missing_count / len(data)) * 100
        
        for col in data.columns:
            if missing_count[col] > 0:
                missing_info[col] = {
                    'count': missing_count[col],
                    'percentage': missing_percentage[col]
                }
        
        return missing_info
    
    @staticmethod
    def check_data_types(data: pd.DataFrame) -> Dict[str, str]:
        """Check data types of DataFrame columns"""
        return data.dtypes.to_dict()
    
    @staticmethod
    def check_duplicates(data: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate rows"""
        duplicate_count = data.duplicated().sum()
        return {
            'count': duplicate_count,
            'percentage': (duplicate_count / len(data)) * 100
        }
    
    @staticmethod
    def validate_time_series_data(data: pd.DataFrame, time_col: str, value_col: str) -> Dict[str, Any]:
        """Validate time series data structure"""
        validation_results = {
            'time_column_exists': time_col in data.columns,
            'value_column_exists': value_col in data.columns,
            'chronological_order': False,
            'missing_timestamps': 0,
            'duplicate_timestamps': 0
        }
        
        if validation_results['time_column_exists']:
            # Check chronological order
            time_series = pd.to_datetime(data[time_col])
            validation_results['chronological_order'] = time_series.is_monotonic_increasing
            
            # Check for duplicate timestamps
            validation_results['duplicate_timestamps'] = time_series.duplicated().sum()
        
        return validation_results

class PerformanceTracker:
    """Track and compare model performance across experiments"""
    
    def __init__(self):
        self.experiments = {}
    
    def add_experiment(self, name: str, metrics: Dict[str, float], 
                      metadata: Dict[str, Any] = None) -> None:
        """Add experiment results"""
        self.experiments[name] = {
            'metrics': metrics,
            'metadata': metadata or {},
            'timestamp': pd.Timestamp.now()
        }
    
    def get_best_model(self, metric: str = 'RMSE', minimize: bool = True) -> Tuple[str, Dict[str, Any]]:
        """Get best performing model based on specified metric"""
        if not self.experiments:
            return None, {}
        
        best_name = None
        best_value = float('inf') if minimize else float('-inf')
        
        for name, experiment in self.experiments.items():
            if metric in experiment['metrics']:
                value = experiment['metrics'][metric]
                if minimize and value < best_value:
                    best_value = value
                    best_name = name
                elif not minimize and value > best_value:
                    best_value = value
                    best_name = name
        
        return best_name, self.experiments.get(best_name, {})
    
    def compare_experiments(self, metrics: List[str] = None) -> pd.DataFrame:
        """Compare all experiments"""
        if not self.experiments:
            return pd.DataFrame()
        
        comparison_data = []
        
        for name, experiment in self.experiments.items():
            row = {'Experiment': name}
            row.update(experiment['metrics'])
            if experiment['metadata']:
                row.update(experiment['metadata'])
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        if metrics:
            # Keep only specified metrics (plus experiment name)
            available_metrics = [m for m in metrics if m in df.columns]
            df = df[['Experiment'] + available_metrics]
        
        return df
    
    def plot_comparison(self, metric: str, figsize: Tuple[int, int] = (12, 6)) -> None:
        """Plot metric comparison across experiments"""
        df = self.compare_experiments()
        
        if metric not in df.columns:
            print(f"Metric '{metric}' not found in experiments")
            return
        
        plt.figure(figsize=figsize)
        
        # Sort by metric value
        df_sorted = df.sort_values(metric)
        
        plt.barh(df_sorted['Experiment'], df_sorted[metric])
        plt.xlabel(metric)
        plt.title(f'Model Comparison - {metric}')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (exp, value) in enumerate(zip(df_sorted['Experiment'], df_sorted[metric])):
            plt.text(value, i, f' {value:.2f}', va='center')
        
        plt.tight_layout()
        plt.show()

class ConfigManager:
    """Manage configuration settings for experiments"""
    
    def __init__(self):
        self.config = {}
    
    def set_config(self, section: str, **kwargs) -> None:
        """Set configuration for a section"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section].update(kwargs)
    
    def get_config(self, section: str, key: str = None, default=None):
        """Get configuration value"""
        if section not in self.config:
            return default if key else {}
        
        if key is None:
            return self.config[section]
        
        return self.config[section].get(key, default)
    
    def save_config(self, filepath: str) -> None:
        """Save configuration to file"""
        import json
        
        # Convert numpy types to native Python types
        config_serializable = {}
        for section, settings in self.config.items():
            config_serializable[section] = {}
            for key, value in settings.items():
                if isinstance(value, np.integer):
                    config_serializable[section][key] = int(value)
                elif isinstance(value, np.floating):
                    config_serializable[section][key] = float(value)
                elif isinstance(value, np.ndarray):
                    config_serializable[section][key] = value.tolist()
                else:
                    config_serializable[section][key] = value
        
        with open(filepath, 'w') as f:
            json.dump(config_serializable, f, indent=2)
    
    def load_config(self, filepath: str) -> None:
        """Load configuration from file"""
        import json
        
        with open(filepath, 'r') as f:
            self.config = json.load(f)
    
    def print_config(self) -> None:
        """Print current configuration"""
        print_section_header("CONFIGURATION SETTINGS", level=1)
        
        for section, settings in self.config.items():
            print_section_header(section, level=2)
            for key, value in settings.items():
                print(f"  {key}: {value}")

def create_experiment_summary(week: int, objectives: List[str], 
                            achievements: List[str], metrics: Dict[str, float] = None) -> str:
    """Create formatted experiment summary"""
    
    summary = []
    summary.append(f"WEEK {week} COMPLETION SUMMARY")
    summary.append("=" * 60)
    
    summary.append(f"\n🎯 OBJECTIVES:")
    for obj in objectives:
        summary.append(f"✅ {obj}")
    
    summary.append(f"\n🏆 KEY ACHIEVEMENTS:")
    for achievement in achievements:
        summary.append(f"• {achievement}")
    
    if metrics:
        summary.append(f"\n📊 PERFORMANCE METRICS:")
        for metric_name, value in metrics.items():
            if 'error' in metric_name.lower() or 'loss' in metric_name.lower():
                summary.append(f"• {metric_name}: {format_number(value)}")
            elif 'accuracy' in metric_name.lower() or 'r2' in metric_name.lower():
                summary.append(f"• {metric_name}: {format_percentage(value * 100)}")
            else:
                summary.append(f"• {metric_name}: {format_number(value)}")
    
    return "\n".join(summary)

def setup_project_environment():
    """Setup project environment with necessary configurations"""
    print_section_header("POWERCAST PROJECT ENVIRONMENT SETUP", level=1)
    
    # Setup matplotlib
    setup_matplotlib()
    print_success("Matplotlib styling configured")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    print_success("Random seeds set for reproducibility")
    
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print_success("Pandas display options configured")
    
    print_success("Project environment setup complete!")

# Initialize environment when module is imported
setup_project_environment()
