import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration for file paths and key settings
CONFIG = {
    'RAW_DATA_PATH': 'data/raw/Tetuan City power consumption.csv',
    'MODEL_PATHS': {
        'Zone_1_Power_Consumption': 'models/best_model_Zone_1_Power_Consumption.pkl',
        'Zone_2_Power_Consumption': 'models/best_model_Zone_2_Power_Consumption.pkl',
        'Zone_3_Power_Consumption': 'models/best_model_Zone_3_Power_Consumption.pkl',
    },
    'FEATURES': ['Temperature', 'Humidity', 'Wind_Speed', 'general_diffuse_flows', 'diffuse_flows'],
    'TARGETS': ['Zone_1_Power_Consumption', 'Zone_2_Power_Consumption', 'Zone_3_Power_Consumption'],
    'RESULTS_DIR': 'results'
}

# Default JSON structures for fallbacks
DEFAULT_JSON = {
    'time_results.json': {
        'timestamp_consistency': {'is_monotonic': False, 'irregular_timestamps': 'N/A'},
        'sampling_frequency': {'frequency_minutes': 'N/A', 'is_consistent': False},
        'duplicates': {'duplicate_count': 0, 'duplicates': []}
    },
    'temporal_results.json': {
        'hourly': {t: {'mean': {}, 'std': {}} for t in CONFIG['TARGETS']},
        'daily': {t: {'mean': {}, 'std': {}} for t in CONFIG['TARGETS']},
        'weekly': {t: {'mean': {}, 'std': {}} for t in CONFIG['TARGETS']}
    },
    'correlation_results.json': {
        t: {f: 'N/A' for f in CONFIG['FEATURES']} for t in CONFIG['TARGETS']
    },
    'lagged_results.json': {
        t: {str(lag): 'N/A' for lag in range(1, 25)} for t in CONFIG['TARGETS']
    },
    'outlier_results.json': {
        col: {'outlier_count': 0, 'outlier_indices': [], 'summary': ''} for col in CONFIG['FEATURES'] + CONFIG['TARGETS']
    },
    'eda_questions.json': {
        'time_consistency': {'questions': [{'question': 'Is the data timestamp consistent?', 'answer': 'Timestamps are {irregular_timestamps} irregular.'}]},
        'temporal_trends': {'questions': [{'question': 'Are there temporal patterns?', 'answer': 'Hourly, daily, and weekly patterns analyzed.'}]},
        'environmental_relationships': {'questions': [{'question': 'How do features correlate with consumption?', 'answer': 'Correlations computed for each zone.'}]},
        'lag_effects': {'questions': [{'question': 'Are there lag effects?', 'answer': 'Lag correlations analyzed up to 24 hours.'}]},
        'data_quality': {'questions': [{'question': 'Are there outliers?', 'answer': '{outlier_count} columns have outliers: {outlier_details}'}]}
    }
}

@st.cache_data
def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        if not os.path.exists(file_path):
            st.error(f"Data file {file_path} not found. Please ensure the data file exists.")
            return None
        df = pd.read_csv(file_path, parse_dates=['DateTime'], index_col='DateTime')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def clean_data(df):
    """Clean the dataset by handling missing values and standardizing column names."""
    if df is None:
        return None
    column_mapping = {
        'Zone 1 Power Consumption': 'Zone_1_Power_Consumption',
        'Zone 2  Power Consumption': 'Zone_2_Power_Consumption',
        'Zone 3  Power Consumption': 'Zone_3_Power_Consumption'
    }
    df = df.rename(columns=column_mapping)
    df.columns = [col.replace(' ', '_') for col in df.columns]
    if df.isnull().any().any():
        df = df.fillna(df.mean(numeric_only=True))
    for zone in CONFIG['TARGETS']:
        if zone in df.columns and (df[zone] < 0).any():
            df[zone] = df[zone].clip(lower=0)
    return df

@st.cache_resource
def load_models():
    """Load all trained models."""
    models = {}
    missing_models = []
    for zone, path in CONFIG['MODEL_PATHS'].items():
        try:
            if not os.path.exists(path):
                missing_models.append(path)
                continue
            loaded = joblib.load(path)
            models[zone] = loaded if isinstance(loaded, tuple) else (loaded, None)
        except Exception as e:
            st.error(f"Error loading model for {zone}: {e}")
            missing_models.append(path)
    if missing_models:
        st.error(f"The following model files are missing: {missing_models}")
        st.info("Please run the data analysis notebook to train the models first.")
    return models

@st.cache_data
def load_eda_results(file_name):
    """Load EDA results from JSON files in results directory with fallback."""
    file_path = os.path.join(CONFIG['RESULTS_DIR'], file_name)
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            st.warning(f"Results file {file_name} is missing or empty. Using default data.")
            return DEFAULT_JSON.get(file_name, None)
        with open(file_path, 'r') as f:
            content = f.read().strip()
            if not content:
                st.warning(f"Results file {file_name} is empty. Using default data.")
                return DEFAULT_JSON.get(file_name, None)
            return json.load(f)
    except json.JSONDecodeError as e:
        st.warning(f"Invalid JSON in {file_name}: {e}. Using default data.")
        return DEFAULT_JSON.get(file_name, None)
    except Exception as e:
        st.error(f"Error loading {file_name}: {e}. Using default data.")
        return DEFAULT_JSON.get(file_name, None)

def create_sample_data():
    """Create sample data for demonstration if real data is unavailable."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    np.random.seed(42)
    data = {
        'Temperature': np.random.normal(20, 5, len(dates)),
        'Humidity': np.random.uniform(30, 80, len(dates)),
        'Wind_Speed': np.random.exponential(2, len(dates)),
        'general_diffuse_flows': np.random.uniform(0, 500, len(dates)),
        'diffuse_flows': np.random.uniform(0, 300, len(dates)),
        'Zone_1_Power_Consumption': np.random.normal(100, 20, len(dates)),
        'Zone_2_Power_Consumption': np.random.normal(80, 15, len(dates)),
        'Zone_3_Power_Consumption': np.random.normal(60, 10, len(dates))
    }
    df = pd.DataFrame(data, index=dates)
    for zone in CONFIG['TARGETS']:
        df[zone] = df[zone] + 0.5 * df['Temperature'] + np.random.normal(0, 5, len(dates))
        df[zone] = df[zone].clip(lower=0)
    return df

def display_eda_section(section_title, section_data, results_data, plots=None):
    """Display an EDA section with questions, answers, and optional plots."""
    st.subheader(section_title)
    if not section_data or 'questions' not in section_data:
        st.warning(f"No questions available for {section_title}.")
        return
    for qa in section_data['questions']:
        answer = qa['answer']
        try:
            if '{irregular_timestamps}' in answer:
                answer = answer.format(
                    irregular_timestamps=results_data.get('timestamp_consistency', {}).get('irregular_timestamps', 'N/A'),
                    frequency_minutes=results_data.get('sampling_frequency', {}).get('frequency_minutes', 'N/A'),
                    consistency_status='fully' if results_data.get('sampling_frequency', {}).get('is_consistent', False) else 'not',
                    duplicate_count=results_data.get('duplicates', {}).get('duplicate_count', 0),
                    duplicate_details=f": {results_data.get('duplicates', {}).get('duplicates', [])[:5]}" if results_data.get('duplicates', {}).get('duplicate_count', 0) > 0 else ''
                )
            elif '{zone_1_temp_corr}' in answer:
                answer = answer.format(
                    zone_1_temp_corr=results_data.get('Zone_1_Power_Consumption', {}).get('Temperature', 'N/A'),
                    zone_2_humidity_corr=results_data.get('Zone_2_Power_Consumption', {}).get('Humidity', 'N/A')
                )
            elif '{lag_1_temp_corr}' in answer:
                answer = answer.format(
                    lag_1_temp_corr=results_data.get('Zone_1_Power_Consumption', {}).get('1', 'N/A'),
                    lag_3_temp_corr=results_data.get('Zone_1_Power_Consumption', {}).get('3', 'N/A'),
                    zone='Zone_1_Power_Consumption'
                )
            elif '{outlier_count}' in answer:
                outlier_cols = [k for k, v in results_data.items() if k != 'boxplot' and v.get('outlier_count', 0) > 0]
                answer = answer.format(
                    outlier_count=len(outlier_cols),
                    outlier_details=', '.join([f"{k}: {v['outlier_count']} outliers ({v['summary']})" for k, v in results_data.items() if k != 'boxplot' and v.get('outlier_count', 0) > 0])
                ) if outlier_cols else "No outliers detected in any columns. All features and targets were analyzed using the IQR method."
            st.markdown(f"**Q: {qa['question']}**")
            st.markdown(f"- **Answer**: {answer}")
        except Exception as e:
            st.error(f"Error formatting answer: {e}")
    if plots:
        for plot_path in plots:
            if plot_path and os.path.exists(plot_path):
                st.image(plot_path, caption=os.path.basename(plot_path).replace('_', ' ').title())

def main():
    st.set_page_config(
        page_title="Tetouan City Power Consumption Prediction",
        page_icon="⚡",
        layout="wide"
    )
    st.title("⚡ Tetouan City Power Consumption Prediction")
    st.markdown("---")
    df = load_data(CONFIG['RAW_DATA_PATH'])
    if df is None:
        st.warning("Using sample data for demonstration purposes.")
        df = create_sample_data()
    else:
        df = clean_data(df)
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Data Visualization", "Model Predictions", "Data Statistics", "Week 1 EDA"])
    if page == "Data Visualization":
        st.header("📊 Power Consumption Trends")
        col1, col2 = st.columns([2, 1])
        with col2:
            zone = st.selectbox("Select Zone", CONFIG['TARGETS'])
            time_period = st.selectbox("Time Period", ["Last 7 Days", "Last 30 Days", "Last 3 Months", "All Data"])
        with col1:
            if time_period == "Last 7 Days":
                filtered_df = df.tail(24 * 7)
            elif time_period == "Last 30 Days":
                filtered_df = df.tail(24 * 30)
            elif time_period == "Last 3 Months":
                filtered_df = df.tail(24 * 90)
            else:
                filtered_df = df
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(filtered_df.index, filtered_df[zone], linewidth=1, color='steelblue')
            ax.set_title(f"{zone.replace('_', ' ')} Over Time", fontsize=16, fontweight='bold')
            ax.set_xlabel("DateTime", fontsize=12)
            ax.set_ylabel("Power Consumption (kWh)", fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        st.subheader("Consumption Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        for zone in CONFIG['TARGETS']:
            if zone in df.columns:
                ax.plot(df.tail(24*7).index, df.tail(24*7)[zone], label=zone.replace('_', ' '), alpha=0.8)
        ax.set_title("Power Consumption - All Zones (Last 7 Days)", fontsize=14)
        ax.set_xlabel("DateTime")
        ax.set_ylabel("Power Consumption (kWh)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    elif page == "Model Predictions":
        st.header("🔮 Predict Power Consumption")
        models = load_models()
        if not models:
            st.error("No models available for prediction. Please train the models first.")
            return
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.0)
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
            with col2:
                wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=20.0, value=2.0)
                general_diffuse_flows = st.number_input("General Diffuse Flows (W/m²)", min_value=0.0, max_value=1000.0, value=100.0)
            with col3:
                diffuse_flows = st.number_input("Diffuse Flows (W/m²)", min_value=0.0, max_value=1000.0, value=80.0)
            predict_button = st.form_submit_button("🚀 Predict Power Consumption")
        if predict_button:
            input_data = pd.DataFrame({
                'Temperature': [temperature],
                'Humidity': [humidity],
                'Wind_Speed': [wind_speed],
                'general_diffuse_flows': [general_diffuse_flows],
                'diffuse_flows': [diffuse_flows]
            })
            st.subheader("📈 Prediction Results")
            results_col1, results_col2, results_col3 = st.columns(3)
            for i, (zone, (model, scaler)) in enumerate(models.items()):
                try:
                    input_scaled = scaler.transform(input_data) if scaler else input_data
                    prediction = model.predict(input_scaled)[0]
                    with [results_col1, results_col2, results_col3][i]:
                        st.metric(label=f"{zone.replace('_', ' ')}", value=f"{prediction:.2f} kWh")
                except Exception as e:
                    st.error(f"Error predicting for {zone}: {e}")
            predictions = []
            zone_names = []
            for zone, (model, scaler) in models.items():
                try:
                    input_scaled = scaler.transform(input_data) if scaler else input_data
                    prediction = model.predict(input_scaled)[0]
                    predictions.append(prediction)
                    zone_names.append(zone.replace('_', ' '))
                except:
                    continue
            if predictions:
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(zone_names, predictions, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                ax.set_title("Predicted Power Consumption by Zone", fontsize=16, fontweight='bold')
                ax.set_ylabel("Power Consumption (kWh)")
                ax.grid(True, alpha=0.3, axis='y')
                for bar, pred in zip(bars, predictions):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height, f'{pred:.1f}', ha='center', va='bottom', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    elif page == "Data Statistics":
        st.header("📊 Data Statistics")
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", len(CONFIG['FEATURES']))
        with col3:
            st.metric("Target Variables", len(CONFIG['TARGETS']))
        with col4:
            st.metric("Date Range", f"{(df.index.max() - df.index.min()).days} days")
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
        st.subheader("Feature Correlation Matrix")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title("Feature Correlation Matrix")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    elif page == "Week 1 EDA":
        st.header("📋 Week 1 EDA Results")
        eda_questions = load_eda_results('eda_questions.json')
        if not eda_questions:
            return
        time_results = load_eda_results('time_results.json')
        if time_results:
            display_eda_section(
                "1. Time Consistency & Structure",
                eda_questions.get('time_consistency', {}),
                time_results
            )
        temporal_results = load_eda_results('temporal_results.json')
        if temporal_results:
            plots = [
                temporal_results.get('temporal_patterns', {}).get('hourly_plot'),
                temporal_results.get('temporal_patterns', {}).get('weekly_plot'),
                temporal_results.get('seasonal_patterns', {}).get('seasonal_plot')
            ]
            display_eda_section(
                "2. Temporal Trends & Seasonality",
                eda_questions.get('temporal_trends', {}),
                temporal_results,
                plots
            )
        correlation_results = load_eda_results('correlation_results.json')
        if correlation_results:
            plots = [correlation_results.get('correlation_plot')]
            for zone in CONFIG['TARGETS']:
                if zone in correlation_results:
                    plots.append(correlation_results.get(zone, {}).get('plot'))
            display_eda_section(
                "3. Environmental Feature Relationships",
                eda_questions.get('environmental_relationships', {}),
                correlation_results,
                plots
            )
        lagged_results = load_eda_results('lagged_results.json')
        if lagged_results:
            plots = []
            for zone in CONFIG['TARGETS']:
                if zone in lagged_results:
                    plots.append(lagged_results.get(zone, {}).get('plot'))
            display_eda_section(
                "4. Lag Effects & Time Dependency",
                eda_questions.get('lag_effects', {}),
                lagged_results,
                plots
            )
        outlier_results = load_eda_results('outlier_results.json')
        if outlier_results:
            plots = [outlier_results.get('boxplot')]
            display_eda_section(
                "5. Data Quality & Sensor Anomalies",
                eda_questions.get('data_quality', {}),
                outlier_results,
                plots
            )

if __name__ == "__main__":
    main()