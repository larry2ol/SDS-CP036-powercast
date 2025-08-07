import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import logging
from pathlib import Path
import time
import subprocess
import json

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit', 'pandas', 'joblib', 'matplotlib', 
        'seaborn', 'numpy', 'sklearn', 'nbformat', 'nbconvert'
    ]
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info("Please install them using: pip install " + " ".join(missing_packages))
        return False
    return True

def create_directory_structure(script_dir):
    """Create necessary directories if they don't exist."""
    directories = ['data/raw', 'models', 'results']
    for directory in directories:
        dir_path = os.path.join(script_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created/verified directory: {dir_path}")

def print_directory_tree(script_dir):
    """Print directory tree using 'tree' command or Python fallback."""
    try:
        result = subprocess.run(
            ['tree', '-a', '-I', '__pycache__|venv', script_dir],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("Directory tree:\n" + result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("tree command not available. Using Python directory listing.")
        try:
            def list_dir(path, indent=""):
                output = []
                for item in sorted(Path(path).iterdir()):
                    output.append(f"{indent}{item.name}")
                    if item.is_dir() and item.name not in ['__pycache__', 'venv']:
                        output.extend(list_dir(item, indent + "  "))
                return output
            tree_output = "\n".join(list_dir(script_dir))
            logger.info(f"Directory tree (Python fallback):\n{tree_output}")
        except Exception as e:
            logger.error(f"Failed to generate directory tree: {e}")

def create_fallback_json(script_dir):
    """Create fallback JSON files if they are missing or invalid."""
    json_files = [
        'results/time_results.json',
        'results/temporal_results.json',
        'results/correlation_results.json',
        'results/lagged_results.json',
        'results/outlier_results.json',
        'results/eda_questions.json'
    ]
    fallback_data = {
        'time_results.json': {
            'timestamp_consistency': {'is_monotonic': False, 'irregular_timestamps': 'N/A'},
            'sampling_frequency': {'frequency_minutes': 'N/A', 'is_consistent': False},
            'duplicates': {'duplicate_count': 0, 'duplicates': []}
        },
        'temporal_results.json': {
            'hourly': {t: {'mean': {}, 'std': {}} for t in ['Zone_1_Power_Consumption', 'Zone_2_Power_Consumption', 'Zone_3_Power_Consumption']},
            'daily': {t: {'mean': {}, 'std': {}} for t in ['Zone_1_Power_Consumption', 'Zone_2_Power_Consumption', 'Zone_3_Power_Consumption']},
            'weekly': {t: {'mean': {}, 'std': {}} for t in ['Zone_1_Power_Consumption', 'Zone_2_Power_Consumption', 'Zone_3_Power_Consumption']}
        },
        'correlation_results.json': {
            t: {f: 'N/A' for f in ['Temperature', 'Humidity', 'Wind_Speed', 'general_diffuse_flows', 'diffuse_flows']}
            for t in ['Zone_1_Power_Consumption', 'Zone_2_Power_Consumption', 'Zone_3_Power_Consumption']
        },
        'lagged_results.json': {
            t: {str(lag): 'N/A' for lag in range(1, 25)}
            for t in ['Zone_1_Power_Consumption', 'Zone_2_Power_Consumption', 'Zone_3_Power_Consumption']
        },
        'outlier_results.json': {
            col: {'outlier_count': 0, 'outlier_indices': []}
            for col in ['Temperature', 'Humidity', 'Wind_Speed', 'general_diffuse_flows', 'diffuse_flows',
                        'Zone_1_Power_Consumption', 'Zone_2_Power_Consumption', 'Zone_3_Power_Consumption']
        },
        'eda_questions.json': {
            'time_consistency': {'questions': [{'question': 'Is the data timestamp consistent?', 'answer': 'Timestamps are {irregular_timestamps} irregular.'}]},
            'temporal_trends': {'questions': [{'question': 'Are there temporal patterns?', 'answer': 'Hourly, daily, and weekly patterns analyzed.'}]},
            'environmental_relationships': {'questions': [{'question': 'How do features correlate with consumption?', 'answer': 'Correlations computed for each zone.'}]},
            'lag_effects': {'questions': [{'question': 'Are there lag effects?', 'answer': 'Lag correlations analyzed up to 24 hours.'}]},
            'data_quality': {'questions': [{'question': 'Are there outliers?', 'answer': '{outlier_count} columns have outliers: {outlier_details}'}]}
        }
    }
    for json_file in json_files:
        file_path = os.path.join(script_dir, json_file)
        try:
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                logger.warning(f"Creating fallback for {json_file}")
                with open(file_path, 'w') as f:
                    json.dump(fallback_data[os.path.basename(json_file)], f, indent=2)
            else:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        json.loads(content)
                    else:
                        logger.warning(f"Empty JSON file {json_file}. Creating fallback.")
                        with open(file_path, 'w') as f:
                            json.dump(fallback_data[os.path.basename(json_file)], f, indent=2)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {json_file}: {e}. Creating fallback.")
            with open(file_path, 'w') as f:
                json.dump(fallback_data[os.path.basename(json_file)], f, indent=2)
        except Exception as e:
            logger.error(f"Error checking {json_file}: {e}. Creating fallback.")
            with open(file_path, 'w') as f:
                json.dump(fallback_data[os.path.basename(json_file)], f, indent=2)

def check_json_files(script_dir):
    """Check if JSON files are valid."""
    json_files = [
        'results/time_results.json',
        'results/temporal_results.json',
        'results/correlation_results.json',
        'results/lagged_results.json',
        'results/outlier_results.json',
        'results/eda_questions.json'
    ]
    invalid_files = []
    for json_file in json_files:
        file_path = os.path.join(script_dir, json_file)
        try:
            if not os.path.exists(file_path):
                logger.warning(f"JSON file missing: {file_path}")
                invalid_files.append(json_file)
            else:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        logger.warning(f"JSON file is empty: {file_path}")
                        invalid_files.append(json_file)
                    else:
                        json.loads(content)
                        logger.debug(f"JSON file valid: {file_path}")
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {file_path}: {e}")
            invalid_files.append(json_file)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            invalid_files.append(json_file)
    return invalid_files

def execute_notebook(notebook_path, timeout=600):
    """Execute notebook using nbconvert's ExecutePreprocessor."""
    try:
        logger.info(f"Starting execution of {notebook_path}")
        if not os.path.exists(notebook_path):
            logger.error(f"Notebook {notebook_path} not found")
            return False
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3', allow_errors=True)
        notebook_dir = os.path.dirname(os.path.abspath(notebook_path))
        ep.preprocess(nb, {'metadata': {'path': notebook_dir}})
        executed_path = notebook_path.replace('.ipynb', '_executed.ipynb')
        with open(executed_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        logger.info(f"Notebook executed successfully, saved as {executed_path}")
        return True
    except Exception as e:
        logger.error(f"Notebook execution failed: {e}")
        return False

def check_data_file(script_dir):
    """Check if the data file exists."""
    data_path = os.path.join(script_dir, "data/raw/Tetuan City power consumption.csv")
    if not os.path.exists(data_path):
        logger.warning(f"Data file not found at {data_path}")
        logger.info("The app will use sample data for demonstration")
        return False
    logger.info(f"Data file found at {data_path}")
    return True

def check_models(script_dir):
    """Check if model files exist."""
    model_paths = [
        'models/best_model_Zone_1_Power_Consumption.pkl',
        'models/best_model_Zone_2_Power_Consumption.pkl',
        'models/best_model_Zone_3_Power_Consumption.pkl'
    ]
    missing_models = []
    for model_path in model_paths:
        full_path = os.path.join(script_dir, model_path)
        if not os.path.exists(full_path):
            missing_models.append(model_path)
    if missing_models:
        logger.warning(f"Missing model files: {missing_models}")
        return False
    logger.info("All model files found")
    return True

def run_streamlit_app(app_path):
    """Run the Streamlit application."""
    try:
        logger.info("Launching Streamlit app...")
        try:
            import streamlit
        except ImportError:
            logger.error("Streamlit not installed. Please install it using: pip install streamlit")
            return False
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Streamlit app failed to start: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("Streamlit app stopped by user")
        return True
    except Exception as e:
        logger.error(f"Error running Streamlit app: {e}")
        return False

def create_simple_notebook_if_missing(script_dir):
    """Create a simple data analysis notebook if none exists."""
    notebook_path = os.path.join(script_dir, "data-analysis.ipynb")
    if os.path.exists(notebook_path):
        return notebook_path
    logger.info("Creating simple data analysis notebook...")
    nb = nbformat.v4.new_notebook()
    cells = [
        nbformat.v4.new_markdown_cell("# Tetouan City Power Consumption Analysis"),
        nbformat.v4.new_code_cell("""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'RAW_DATA_PATH': 'data/raw/Tetuan City power consumption.csv',
    'MODEL_DIR': 'models',
    'RESULTS_DIR': 'results',
    'MODEL_PATHS': {
        'Zone_1_Power_Consumption': 'models/best_model_Zone_1_Power_Consumption.pkl',
        'Zone_2_Power_Consumption': 'models/best_model_Zone_2_Power_Consumption.pkl',
        'Zone_3_Power_Consumption': 'models/best_model_Zone_3_Power_Consumption.pkl',
    },
    'FEATURES': ['Temperature', 'Humidity', 'Wind_Speed', 'general_diffuse_flows', 'diffuse_flows'],
    'TARGETS': ['Zone_1_Power_Consumption', 'Zone_2_Power_Consumption', 'Zone_3_Power_Consumption']
}

# Create directories
os.makedirs(CONFIG['RESULTS_DIR'], exist_ok=True)
os.makedirs(CONFIG['MODEL_DIR'], exist_ok=True)
print("Setup complete!")

def json_serializable(obj):
    if pd.isna(obj) or obj is np.nan:
        return None
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    return str(obj)
        """),
        nbformat.v4.new_markdown_cell("## Load and Clean Data"),
        nbformat.v4.new_code_cell("""
def load_data(file_path):
    \"\"\"Load the dataset from a CSV file.\"\"\"
    if not os.path.exists(file_path):
        print(f"Dataset not found at {file_path}")
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='10min')
        np.random.seed(42)
        return pd.DataFrame({
            'Temperature': np.random.normal(20, 5, len(dates)),
            'Humidity': np.random.uniform(30, 80, len(dates)),
            'Wind_Speed': np.random.exponential(2, len(dates)),
            'general_diffuse_flows': np.random.uniform(0, 500, len(dates)),
            'diffuse_flows': np.random.uniform(0, 300, len(dates)),
            'Zone_1_Power_Consumption': np.random.normal(100, 20, len(dates)),
            'Zone_2_Power_Consumption': np.random.normal(80, 15, len(dates)),
            'Zone_3_Power_Consumption': np.random.normal(60, 10, len(dates))
        }, index=dates)
    try:
        df = pd.read_csv(file_path, parse_dates=['DateTime'], index_col='DateTime')
        print(f"Loaded dataset with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    \"\"\"Clean the dataset by handling missing values, standardizing column names, and removing duplicates.\"\"\"
    if df is None or df.empty:
        return None, {'duplicate_count': 0, 'duplicates': []}
    column_mapping = {
        'Zone 1 Power Consumption': 'Zone_1_Power_Consumption',
        'Zone 2  Power Consumption': 'Zone_2_Power_Consumption',
        'Zone 3  Power Consumption': 'Zone_3_Power_Consumption',
        'Wind Speed': 'Wind_Speed',
        'general diffuse flows': 'general_diffuse_flows',
        'diffuse flows': 'diffuse_flows'
    }
    df = df.rename(columns=column_mapping)
    df.columns = [col.replace(' ', '_') for col in df.columns]
    if df.isnull().any().any():
        print(f"Found {df.isnull().sum().sum()} missing values. Filling with mean.")
        df = df.fillna(df.mean(numeric_only=True))
    for zone in CONFIG['TARGETS']:
        if zone in df.columns and (df[zone] < 0).any():
            print(f"Negative values found in {zone}. Replacing with 0.")
            df[zone] = df[zone].clip(lower=0)
    if df.index.duplicated().any():
        duplicate_count = df.index.duplicated().sum()
        print(f"Found {duplicate_count} duplicate timestamps. Aggregating by mean.")
        duplicate_indices = df.index[df.index.duplicated()].strftime('%Y-%m-%d %H:%M:%S').tolist()
        df = df.groupby(df.index).mean()
        print(f"After aggregation, dataset shape: {df.shape}")
        return df, {'duplicate_count': duplicate_count, 'duplicates': duplicate_indices[:5]}
    else:
        print("No duplicate timestamps found.")
        return df, {'duplicate_count': 0, 'duplicates': []}
try:
    df = load_data(CONFIG['RAW_DATA_PATH'])
    if df is not None:
        df, duplicate_info = clean_data(df)
        print("Data loaded and cleaned successfully!")
        print(df.head())
    else:
        print("No data file found. Creating sample data for demonstration.")
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='10min')
        np.random.seed(42)
        df = pd.DataFrame({
            'Temperature': np.random.normal(20, 5, len(dates)),
            'Humidity': np.random.uniform(30, 80, len(dates)),
            'Wind_Speed': np.random.exponential(2, len(dates)),
            'general_diffuse_flows': np.random.uniform(0, 500, len(dates)),
            'diffuse_flows': np.random.uniform(0, 300, len(dates)),
            'Zone_1_Power_Consumption': np.random.normal(100, 20, len(dates)),
            'Zone_2_Power_Consumption': np.random.normal(80, 15, len(dates)),
            'Zone_3_Power_Consumption': np.random.normal(60, 10, len(dates))
        }, index=dates)
        print("Sample data created!")
        duplicate_info = {'duplicate_count': 0, 'duplicates': []}
except Exception as e:
    print(f"Error in data processing: {e}")
        """),
        nbformat.v4.new_markdown_cell("## Model Training"),
        nbformat.v4.new_code_cell("""
def train_models():
    \"\"\"Train machine learning models for power consumption prediction.\"\"\"
    if df is None or df.empty:
        print("No data available for training!")
        return
    missing_features = [col for col in CONFIG['FEATURES'] if col not in df.columns]
    missing_targets = [col for col in CONFIG['TARGETS'] if col not in df.columns]
    if missing_features:
        print(f"Missing feature columns: {missing_features}")
        return
    if missing_targets:
        print(f"Missing target columns: {missing_targets}")
        return
    scaler = StandardScaler()
    models = {
        'Linear_Regression': LinearRegression(),
        'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    results = {}
    best_models = {}
    for zone in CONFIG['TARGETS']:
        print(f"\\nTraining models for {zone}")
        X = df[CONFIG['FEATURES']]
        y = df[zone]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        best_r2 = -float('inf')
        best_model = None
        results[zone] = {}
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results[zone][name] = {'MSE': mse, 'R2': r2}
                print(f"  {name} - MSE: {mse:.2f}, R2: {r2:.3f}")
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
                    best_models[zone] = (model, scaler)
            except Exception as e:
                print(f"  Error training {name} for {zone}: {e}")
        if best_model is not None:
            try:
                joblib.dump(best_models[zone], CONFIG['MODEL_PATHS'][zone])
                print(f"  Best model saved to {CONFIG['MODEL_PATHS'][zone]}")
            except Exception as e:
                print(f"  Error saving model: {e}")
    try:
        with open(os.path.join(CONFIG['RESULTS_DIR'], 'model_performance.json'), 'w') as f:
            json.dump(results, f, indent=2, default=json_serializable)
        print("\\nModel performance saved to results/model_performance.json")
    except Exception as e:
        print(f"Error saving results: {e}")
    return results, best_models
try:
    results, best_models = train_models()
    print("\\nModel training completed!")
except Exception as e:
    print(f"Error in model training: {e}")
        """)
    ]
    nb.cells = cells
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    logger.info(f"Simple notebook created at {notebook_path}")
    return notebook_path

def main():
    try:
        if not check_requirements():
            sys.exit(1)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Project directory: {script_dir}")
        create_directory_structure(script_dir)
        print_directory_tree(script_dir)
        data_exists = check_data_file(script_dir)
        models_exist = check_models(script_dir)
        if not models_exist:
            logger.info("Models not found. Looking for notebook to execute...")
            notebook_path = create_simple_notebook_if_missing(script_dir)
            if notebook_path and os.path.exists(notebook_path):
                logger.info(f"Executing notebook: {notebook_path}")
                if not execute_notebook(notebook_path):
                    logger.warning("Notebook execution failed, but continuing with app launch")
            else:
                logger.warning("No notebook found and couldn't create one")
        else:
            logger.info("Models already exist. Skipping notebook execution.")
        create_fallback_json(script_dir)
        invalid_json_files = check_json_files(script_dir)
        if invalid_json_files:
            logger.warning(f"Invalid or missing JSON files: {invalid_json_files}")
            logger.info("Streamlit app may display partial data")
        app_path = os.path.join(script_dir, "app.py")
        if not os.path.exists(app_path):
            logger.error(f"Streamlit app file {app_path} not found")
            sys.exit(1)
        logger.info("=== System Status ===")
        logger.info(f"Data file: {'✓ Found' if data_exists else '✗ Missing (will use sample data)'}")
        logger.info(f"Models: {'✓ Found' if models_exist else '✗ Missing (prediction may not work)'}")
        logger.info(f"JSON files: {'✓ All valid' if not invalid_json_files else f'✗ Issues with {invalid_json_files}'}")
        logger.info(f"App file: {'✓ Found' if os.path.exists(app_path) else '✗ Missing'}")
        logger.info("====================")
        time.sleep(2)
        run_streamlit_app(app_path)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()