"""
Data Preprocessing Utilities for PowerCast Advanced Track Project
Contains functions for data loading, cleaning, and preprocessing.
"""

import pandas as pd
import numpy as np


class DataLoader:
    """Handle data loading and initial inspection"""
    
    @staticmethod
    def load_and_inspect_data(file_path):
        """Load dataset and perform initial inspection"""
        df = pd.read_csv(file_path)
        
        print("📂 DATASET LOADED SUCCESSFULLY")
        print("="*40)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\n📊 Data Info:")
        df.info()
        
        return df


class DataCleaner:
    """Handle data cleaning and formatting operations"""
    
    @staticmethod
    def rename_columns(df):
        """Rename columns for clarity and consistency"""
        df.columns = [
            'datetime',
            'temperature', 
            'humidity',
            'wind_speed',
            'general_diffuse_flows',
            'diffuse_flows', 
            'zone1_power_consumption',
            'zone2_power_consumption',
            'zone3_power_consumption'
        ]
        
        print("✅ Columns renamed successfully:")
        print(df.columns.tolist())
        return df
    
    @staticmethod
    def check_data_quality(df):
        """Comprehensive data quality assessment"""
        print("🔍 DATA QUALITY ASSESSMENT")
        print("="*40)
        
        # Check data types
        print("Data types:")
        print(df.dtypes)
        print("\n" + "="*50)
        
        # Check for missing values
        print("Missing values:")
        missing_values = df.isnull().sum()
        print(missing_values)
        print(f"Total missing values: {missing_values.sum()}")
        print("\n" + "="*50)
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        print(f"Duplicate rows: {duplicate_count}")
        print("\n" + "="*50)
        
        # Basic statistics
        print(f"Dataset shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return {
            'missing_values': missing_values,
            'duplicate_count': duplicate_count,
            'shape': df.shape
        }
    
    @staticmethod
    def convert_datetime(df, datetime_col='datetime'):
        """Convert datetime column and set as index"""
        print("📅 Converting datetime column...")
        
        # Convert to datetime
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # Verify conversion
        print("✅ Datetime conversion successful!")
        print(f"Datetime range: {df[datetime_col].min()} to {df[datetime_col].max()}")
        print(f"Data types after conversion:")
        print(df.dtypes)
        
        return df
    
    @staticmethod
    def set_datetime_index(df, datetime_col='datetime'):
        """Set datetime column as index for time series analysis"""
        df.set_index(datetime_col, inplace=True)
        
        print("🕐 DATETIME INDEX SET")
        print("="*30)
        print(f"New shape: {df.shape}")
        print(f"Index type: {type(df.index)}")
        
        return df
    
    @staticmethod
    def analyze_time_series_properties(df):
        """Analyze time series characteristics"""
        print("📈 TIME SERIES ANALYSIS")
        print("="*40)
        
        # Sort by datetime if not already sorted
        if not df.index.is_monotonic_increasing:
            df.sort_index(inplace=True)
            print("✅ Data sorted by datetime")
        
        # Check for time gaps and frequency
        df_sorted = df.sort_index()
        time_diff = df_sorted.index.to_series().diff().dropna()
        
        print(f"Time differences (first 10): {time_diff.head(10).values}")
        print(f"Most common time interval: {time_diff.mode().iloc[0]}")
        print(f"Minimum time interval: {time_diff.min()}")
        print(f"Maximum time interval: {time_diff.max()}")
        
        # Check for irregular timestamps
        irregular_gaps = time_diff[time_diff != time_diff.mode().iloc[0]]
        print(f"Number of irregular time gaps: {len(irregular_gaps)}")
        
        return {
            'most_common_interval': time_diff.mode().iloc[0],
            'irregular_gaps': len(irregular_gaps),
            'sorted': df.index.is_monotonic_increasing
        }
    
    @staticmethod
    def validate_data_ranges(df, power_cols, env_features):
        """Validate data ranges for logical consistency"""
        print("✅ DATA RANGE VALIDATION")
        print("="*40)
        
        # Check for negative values in power consumption
        for col in power_cols:
            negative_count = (df[col] < 0).sum()
            print(f"Negative values in {col}: {negative_count}")
        
        # Check for zero values in power consumption
        for col in power_cols:
            zero_count = (df[col] == 0).sum()
            print(f"Zero values in {col}: {zero_count}")
        
        # Environmental features validation
        if 'temperature' in env_features:
            extreme_temp = ((df['temperature'] < -50) | (df['temperature'] > 60)).sum()
            print(f"Extreme temperature values: {extreme_temp}")
        
        if 'humidity' in env_features:
            invalid_humidity = ((df['humidity'] < 0) | (df['humidity'] > 100)).sum()
            print(f"Invalid humidity values: {invalid_humidity}")
        
        if 'wind_speed' in env_features:
            negative_wind = (df['wind_speed'] < 0).sum()
            print(f"Negative wind speed values: {negative_wind}")
        
        # Summary statistics
        print(f"\n📊 SUMMARY STATISTICS")
        print("="*30)
        print(df.describe())
        
        return df


class DataPreprocessor:
    """Complete data preprocessing pipeline"""
    
    @staticmethod
    def preprocess_pipeline(file_path):
        """Complete preprocessing pipeline"""
        print("🚀 STARTING DATA PREPROCESSING PIPELINE")
        print("="*50)
        
        # Step 1: Load data
        df = DataLoader.load_and_inspect_data(file_path)
        
        # Step 2: Rename columns
        df = DataCleaner.rename_columns(df)
        
        # Step 3: Check data quality
        quality_report = DataCleaner.check_data_quality(df)
        
        # Step 4: Convert datetime
        df = DataCleaner.convert_datetime(df)
        
        # Step 5: Set datetime index
        df = DataCleaner.set_datetime_index(df)
        
        # Step 6: Analyze time series properties
        ts_properties = DataCleaner.analyze_time_series_properties(df)
        
        # Step 7: Validate data ranges
        power_cols = ['zone1_power_consumption', 'zone2_power_consumption', 'zone3_power_consumption']
        env_features = ['temperature', 'humidity', 'wind_speed', 'general_diffuse_flows', 'diffuse_flows']
        df = DataCleaner.validate_data_ranges(df, power_cols, env_features)
        
        print("\n✅ PREPROCESSING PIPELINE COMPLETED")
        print("="*40)
        print(f"Final dataset shape: {df.shape}")
        print(f"Time period: {df.index.min()} to {df.index.max()}")
        print(f"Duration: {(df.index.max() - df.index.min()).days} days")
        
        return df, quality_report, ts_properties


# Define feature groups for easy access
POWER_COLS = ['zone1_power_consumption', 'zone2_power_consumption', 'zone3_power_consumption']
ENV_FEATURES = ['temperature', 'humidity', 'wind_speed', 'general_diffuse_flows', 'diffuse_flows']
ALL_FEATURES = ENV_FEATURES + POWER_COLS
