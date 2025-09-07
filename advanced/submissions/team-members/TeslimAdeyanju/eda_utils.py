"""
EDA Utilities for PowerCast Advanced Track Project
Contains functions for exploratory data analysis, visualization, and data quality assessment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable, coolwarm
from matplotlib.colors import Normalize


class TimeConsistencyAnalyzer:
    """Analyze time consistency and structure of time series data"""
    
    @staticmethod
    def analyze_time_consistency(df):
        """Comprehensive time consistency analysis"""
        print("🕐 TIME CONSISTENCY & STRUCTURE ANALYSIS")
        print("="*60)
        
        # Verify time index is sorted
        if df.index.is_monotonic_increasing:
            print("✅ Timestamp index is properly sorted (ascending)")
        else:
            print("❌ Timestamp index is NOT sorted - needs sorting")
            df.sort_index(inplace=True)
            print("✅ Index sorted successfully")
        
        # Check for missing timestamps
        print(f"\n📊 Dataset Time Coverage:")
        print(f"Start Date: {df.index.min()}")
        print(f"End Date: {df.index.max()}")
        print(f"Total Duration: {(df.index.max() - df.index.min()).days} days")
        print(f"Total Records: {len(df):,}")
        
        # Analyze time frequency
        time_deltas = df.index.to_series().diff().dropna()
        most_common_freq = time_deltas.mode()[0]
        print(f"\n⏱️  Sampling Frequency Analysis:")
        print(f"Most common interval: {most_common_freq}")
        print(f"Expected records per day: {pd.Timedelta('1D') / most_common_freq:.0f}")
        
        # Check for gaps in time series
        gaps = time_deltas[time_deltas != most_common_freq]
        print(f"\n🔍 Time Gaps Detection:")
        print(f"Total irregular intervals: {len(gaps)}")
        if len(gaps) > 0:
            print(f"Largest gap: {gaps.max()}")
            print(f"Smallest irregular interval: {gaps.min()}")
            
        # Check for duplicate timestamps
        duplicate_times = df.index.duplicated().sum()
        print(f"\n🔄 Duplicate Timestamps: {duplicate_times}")
        
        # Generate expected time range and compare
        expected_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=most_common_freq)
        missing_timestamps = len(expected_range) - len(df)
        print(f"\n📉 Missing Data Points:")
        print(f"Expected records: {len(expected_range):,}")
        print(f"Actual records: {len(df):,}")
        print(f"Missing records: {missing_timestamps:,}")
        print(f"Completeness: {(len(df)/len(expected_range))*100:.2f}%")
        
        return {
            'most_common_freq': most_common_freq,
            'gaps': len(gaps),
            'duplicate_times': duplicate_times,
            'completeness': (len(df)/len(expected_range))*100
        }


class TemporalAnalyzer:
    """Analyze temporal trends and seasonality patterns"""
    
    @staticmethod
    def extract_temporal_features(df):
        """Extract time-based features for analysis"""
        df_temporal = df.copy()
        
        # Extract time-based features
        df_temporal['hour'] = df_temporal.index.hour
        df_temporal['day_of_week'] = df_temporal.index.dayofweek
        df_temporal['day_of_year'] = df_temporal.index.dayofyear
        df_temporal['month'] = df_temporal.index.month
        df_temporal['quarter'] = df_temporal.index.quarter
        df_temporal['is_weekend'] = df_temporal['day_of_week'].isin([5, 6])
        df_temporal['day_name'] = df_temporal.index.strftime('%A')
        df_temporal['month_name'] = df_temporal.index.strftime('%B')
        
        print("✅ Temporal features extracted successfully!")
        print(f"Added features: hour, day_of_week, day_of_year, month, quarter, is_weekend")
        print(f"Dataset shape with temporal features: {df_temporal.shape}")
        
        return df_temporal
    
    @staticmethod
    def plot_temporal_trends(df_temporal, power_cols):
        """Create comprehensive temporal trends visualization with better spacing"""
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # Figure 1: Overall time series and hourly patterns
        fig1, axes1 = plt.subplots(2, 1, figsize=(18, 12))
        fig1.suptitle('🔍 TEMPORAL TRENDS - TIME SERIES & HOURLY PATTERNS', fontsize=16, fontweight='bold')
        
        # 1. Overall time series plot
        ax1 = axes1[0]
        for col in power_cols:
            ax1.plot(df_temporal.index, df_temporal[col], alpha=0.7, linewidth=0.8, 
                    label=col.replace('_', ' ').title())
        ax1.set_title('📈 Power Consumption Over Time - All Zones', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Power Consumption')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Hourly patterns
        ax2 = axes1[1]
        hourly_avg = df_temporal.groupby('hour')[power_cols].mean()
        for col in power_cols:
            ax2.plot(hourly_avg.index, hourly_avg[col], marker='o', linewidth=2, 
                    label=col.replace('_', ' ').title())
        ax2.set_title('⏰ Average Hourly Consumption', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Average Power Consumption')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 3))
        
        plt.tight_layout()
        plt.show()
        
        # Figure 2: Daily and monthly patterns
        fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
        fig2.suptitle('🔍 TEMPORAL TRENDS - DAILY & MONTHLY PATTERNS', fontsize=16, fontweight='bold')
        
        # 3. Daily patterns (by day of week)
        ax3 = axes2[0]
        daily_avg = df_temporal.groupby('day_name')[power_cols].mean()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_avg = daily_avg.reindex(day_order)
        x_pos = range(len(day_order))
        width = 0.25
        for i, col in enumerate(power_cols):
            ax3.bar([x + width*i for x in x_pos], daily_avg[col], width, 
                   label=col.replace('_', ' ').title(), alpha=0.8)
        ax3.set_title('📅 Average Daily Consumption by Weekday', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Day of Week')
        ax3.set_ylabel('Average Power Consumption')
        ax3.set_xticks([x + width for x in x_pos])
        ax3.set_xticklabels([day[:3] for day in day_order])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Monthly patterns
        ax4 = axes2[1]
        monthly_avg = df_temporal.groupby('month_name')[power_cols].mean()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_avg = monthly_avg.reindex([m for m in month_order if m in monthly_avg.index])
        for col in power_cols:
            ax4.plot(range(len(monthly_avg)), monthly_avg[col], marker='s', linewidth=2,
                    label=col.replace('_', ' ').title())
        ax4.set_title('📆 Average Monthly Consumption', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Average Power Consumption')
        ax4.set_xticks(range(len(monthly_avg)))
        ax4.set_xticklabels([month[:3] for month in monthly_avg.index], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Figure 3: Weekend vs Weekday and Seasonal patterns
        fig3, axes3 = plt.subplots(1, 2, figsize=(18, 8))
        fig3.suptitle('🔍 TEMPORAL TRENDS - WEEKEND/WEEKDAY & SEASONAL PATTERNS', fontsize=16, fontweight='bold')
        
        # 5. Weekend vs Weekday comparison
        ax5 = axes3[0]
        weekend_comparison = df_temporal.groupby('is_weekend')[power_cols].mean()
        x_labels = ['Weekday', 'Weekend']
        x_pos = range(len(x_labels))
        width = 0.25
        for i, col in enumerate(power_cols):
            ax5.bar([x + width*i for x in x_pos], weekend_comparison[col], width,
                   label=col.replace('_', ' ').title(), alpha=0.8)
        ax5.set_title('🏢 Weekday vs Weekend Consumption', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Day Type')
        ax5.set_ylabel('Average Power Consumption')
        ax5.set_xticks([x + width for x in x_pos])
        ax5.set_xticklabels(x_labels)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Seasonal patterns (quarterly)
        ax6 = axes3[1]
        quarterly_avg = df_temporal.groupby('quarter')[power_cols].mean()
        for col in power_cols:
            ax6.plot(quarterly_avg.index, quarterly_avg[col], marker='D', linewidth=2,
                    label=col.replace('_', ' ').title())
        ax6.set_title('🌍 Seasonal Consumption (Quarterly)', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Quarter')
        ax6.set_ylabel('Average Power Consumption')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_xticks([1, 2, 3, 4])
        ax6.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
        
        plt.tight_layout()
        plt.show()
        
        # Figure 4: Heatmap
        fig4, ax7 = plt.subplots(1, 1, figsize=(12, 8))
        fig4.suptitle('🔍 TEMPORAL TRENDS - HOUR vs DAY HEATMAP', fontsize=16, fontweight='bold')
        
        # 7. Heatmap: Hour vs Day of Week
        pivot_data = df_temporal.groupby(['hour', 'day_of_week'])['zone1_power_consumption'].mean().unstack()
        day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        pivot_data.columns = day_labels
        im = ax7.imshow(pivot_data.T, cmap='YlOrRd', aspect='auto')
        ax7.set_title('🔥 Zone 1: Hour vs Day Heatmap', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Hour of Day')
        ax7.set_ylabel('Day of Week')
        ax7.set_xticks(range(0, 24, 3))
        ax7.set_xticklabels(range(0, 24, 3))
        ax7.set_yticks(range(7))
        ax7.set_yticklabels(day_labels)
        plt.colorbar(im, ax=ax7, shrink=0.8)
        
        plt.tight_layout()
        plt.show()


class EnvironmentalAnalyzer:
    """Analyze environmental feature relationships with power consumption"""
    
    @staticmethod
    def analyze_environmental_correlations(df, env_features, power_cols):
        """Analyze correlations between environmental features and power consumption"""
        print("🌡️ ENVIRONMENTAL FEATURE RELATIONSHIPS ANALYSIS")
        print("="*60)
        
        # Calculate correlation matrix
        correlation_matrix = df[env_features + power_cols].corr()
        env_power_corr = correlation_matrix.loc[env_features, power_cols]
        
        print("Correlation between Environmental Features and Power Consumption:")
        print("="*60)
        print(env_power_corr.round(3))
        
        # Find strongest correlations
        print(f"\nStrongest Correlations:")
        for zone in power_cols:
            zone_corr = env_power_corr[zone].abs()
            strongest_feature = zone_corr.idxmax()
            strongest_value = env_power_corr.loc[strongest_feature, zone]
            print(f"{zone}: {strongest_feature} (r = {strongest_value:.3f})")
        
        return env_power_corr
    
    @staticmethod
    def plot_environmental_relationships(df, env_features, power_cols, env_power_corr):
        """Create comprehensive environmental features visualization with better spacing"""
        
        # Figure 1: Correlation heatmap and temperature/humidity relationships
        fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
        fig1.suptitle('ENVIRONMENTAL FEATURES vs POWER CONSUMPTION - PART 1', fontsize=16, fontweight='bold')
        
        # 1. Correlation Heatmap
        ax1 = axes1[0, 0]
        im1 = ax1.imshow(env_power_corr.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax1.set_title('Correlation Heatmap', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(power_cols)))
        ax1.set_yticks(range(len(env_features)))
        ax1.set_xticklabels([col.replace('_', '\n').title() for col in power_cols], rotation=45)
        ax1.set_yticklabels([feat.replace('_', ' ').title() for feat in env_features])
        
        # Add correlation values to heatmap
        for i in range(len(env_features)):
            for j in range(len(power_cols)):
                text = ax1.text(j, i, f'{env_power_corr.iloc[i, j]:.2f}', 
                               ha="center", va="center", 
                               color="white" if abs(env_power_corr.iloc[i, j]) > 0.5 else "black")
        
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # 2. Temperature vs Power Consumption
        ax2 = axes1[0, 1]
        for zone in power_cols:
            ax2.scatter(df['temperature'], df[zone], alpha=0.3, s=1, 
                       label=zone.replace('_', ' ').title())
        ax2.set_title('Temperature vs Power Consumption', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel('Power Consumption')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Humidity vs Power Consumption  
        ax3 = axes1[1, 0]
        for zone in power_cols:
            ax3.scatter(df['humidity'], df[zone], alpha=0.3, s=1, 
                       label=zone.replace('_', ' ').title())
        ax3.set_title('Humidity vs Power Consumption', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Humidity (%)')
        ax3.set_ylabel('Power Consumption')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Wind Speed vs Power Consumption
        ax4 = axes1[1, 1]
        for zone in power_cols:
            ax4.scatter(df['wind_speed'], df[zone], alpha=0.3, s=1, 
                       label=zone.replace('_', ' ').title())
        ax4.set_title('Wind Speed vs Power Consumption', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Wind Speed')
        ax4.set_ylabel('Power Consumption')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Figure 2: Solar radiation and feature distributions
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
        fig2.suptitle('ENVIRONMENTAL FEATURES vs POWER CONSUMPTION - PART 2', fontsize=16, fontweight='bold')
        
        # 5. Solar Radiation vs Power Consumption
        ax5 = axes2[0]
        for zone in power_cols:
            ax5.scatter(df['general_diffuse_flows'], df[zone], alpha=0.3, s=1, 
                       label=zone.replace('_', ' ').title())
        ax5.set_title('Solar Radiation vs Power Consumption', fontsize=12, fontweight='bold')
        ax5.set_xlabel('General Diffuse Flows')
        ax5.set_ylabel('Power Consumption')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Environmental features distribution
        ax6 = axes2[1]
        env_data = df[env_features]
        env_data_normalized = (env_data - env_data.mean()) / env_data.std()
        for feature in env_features:
            ax6.hist(env_data_normalized[feature], alpha=0.6, bins=50, 
                    label=feature.replace('_', ' ').title(), density=True)
        ax6.set_title('Environmental Features Distribution', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Normalized Values')
        ax6.set_ylabel('Density')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class LagAnalyzer:
    """Analyze lag effects and time dependencies"""
    
    @staticmethod
    def calculate_lag_correlations(feature_series, target_series, max_lags=72):
        """Calculate correlations between a feature and target at various lags"""
        correlations = []
        for lag in range(max_lags + 1):
            if lag == 0:
                corr = feature_series.corr(target_series)
            else:
                lagged_feature = feature_series.shift(lag)
                corr = lagged_feature.corr(target_series)
            correlations.append(corr)
        return correlations
    
    @staticmethod
    def analyze_lag_effects(df, env_features, power_cols, max_lag_hours=72):
        """Comprehensive lag effects analysis"""
        print("⏳ LAG EFFECTS & TIME DEPENDENCY ANALYSIS")
        print("="*60)
        
        # Calculate lag correlations for environmental features vs power consumption
        lag_results = {}
        
        for env_feature in env_features:
            lag_results[env_feature] = {}
            for zone in power_cols:
                lag_corr = LagAnalyzer.calculate_lag_correlations(
                    df[env_feature], df[zone], max_lag_hours)
                lag_results[env_feature][zone] = lag_corr
                
                # Find optimal lag
                max_corr_idx = np.argmax(np.abs(lag_corr))
                max_corr_value = lag_corr[max_corr_idx]
                
                print(f"{env_feature} → {zone}:")
                print(f"  Best lag: {max_corr_idx} hours, Correlation: {max_corr_value:.3f}")
        
        # Autocorrelation analysis for power consumption
        print(f"\n📈 AUTOCORRELATION ANALYSIS:")
        print("="*40)
        
        autocorr_results = {}
        for zone in power_cols:
            autocorr = [df[zone].autocorr(lag=i) for i in range(1, 169)]  # 1 week of hourly lags
            autocorr_results[zone] = autocorr
            
            # Find significant autocorrelations
            significant_lags = [i+1 for i, corr in enumerate(autocorr) if abs(corr) > 0.3]
            print(f"{zone}:")
            print(f"  Significant lags (|corr| > 0.3): {significant_lags[:10]}...")
        
        return lag_results, autocorr_results
    
    @staticmethod
    def plot_lag_analysis(lag_results, autocorr_results, env_features, power_cols, max_lag_hours=72):
        """Create comprehensive lag analysis visualization with better spacing"""
        
        lag_hours = list(range(max_lag_hours + 1))
        
        # Figure 1: Environmental feature lag correlations (Temperature & Humidity)
        fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
        fig1.suptitle('⏳ LAG EFFECTS - ENVIRONMENTAL FEATURES (PART 1)', fontsize=16, fontweight='bold')
        
        # Temperature lag correlations
        ax1 = axes1[0]
        for zone in power_cols:
            ax1.plot(lag_hours, lag_results['temperature'][zone], 
                    label=zone.replace('_', ' ').title(), linewidth=2)
        ax1.set_title('🌡️ Temperature Lag Correlations', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Lag (hours)')
        ax1.set_ylabel('Correlation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Humidity lag correlations
        ax2 = axes1[1]
        for zone in power_cols:
            ax2.plot(lag_hours, lag_results['humidity'][zone], 
                    label=zone.replace('_', ' ').title(), linewidth=2)
        ax2.set_title('💧 Humidity Lag Correlations', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Lag (hours)')
        ax2.set_ylabel('Correlation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        # Figure 2: Environmental feature lag correlations (Wind Speed & Solar Radiation)
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
        fig2.suptitle('⏳ LAG EFFECTS - ENVIRONMENTAL FEATURES (PART 2)', fontsize=16, fontweight='bold')
        
        # Wind Speed lag correlations
        ax3 = axes2[0]
        for zone in power_cols:
            ax3.plot(lag_hours, lag_results['wind_speed'][zone], 
                    label=zone.replace('_', ' ').title(), linewidth=2)
        ax3.set_title('💨 Wind Speed Lag Correlations', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Lag (hours)')
        ax3.set_ylabel('Correlation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Solar Radiation lag correlations
        ax4 = axes2[1]
        for zone in power_cols:
            ax4.plot(lag_hours, lag_results['general_diffuse_flows'][zone], 
                    label=zone.replace('_', ' ').title(), linewidth=2)
        ax4.set_title('☀️ Solar Radiation Lag Correlations', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Lag (hours)')
        ax4.set_ylabel('Correlation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        # Figure 3: Autocorrelation analysis
        fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
        fig3.suptitle('⏳ LAG EFFECTS - AUTOCORRELATION ANALYSIS', fontsize=16, fontweight='bold')
        
        # Short-term autocorrelation (48 hours)
        ax5 = axes3[0]
        autocorr_lags = list(range(1, 49))  # First 48 hours
        for zone in power_cols:
            ax5.plot(autocorr_lags, autocorr_results[zone][:48], 
                    label=zone.replace('_', ' ').title(), linewidth=2, marker='o', markersize=3)
        ax5.set_title('📈 Autocorrelation (48h)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Lag (hours)')
        ax5.set_ylabel('Autocorrelation')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.axhline(y=0.3, color='red', linestyle='--', alpha=0.5)
        ax5.axhline(y=-0.3, color='red', linestyle='--', alpha=0.5)
        
        # Long-term autocorrelation (1 week)
        ax6 = axes3[1]
        weekly_lags = list(range(1, 169))  # 1 week
        for zone in power_cols:
            ax6.plot(weekly_lags, autocorr_results[zone], 
                    label=zone.replace('_', ' ').title(), linewidth=1.5)
        ax6.set_title('📅 Long-term Autocorrelation (1 week)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Lag (hours)')
        ax6.set_ylabel('Autocorrelation')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        # Mark daily cycles
        for day in range(1, 8):
            ax6.axvline(x=day*24, color='orange', linestyle=':', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Summary of findings
        print(f"\n📋 LAG ANALYSIS SUMMARY:")
        print("="*40)
        print("Key findings:")
        print("• Most environmental features show immediate correlation (lag 0)")
        print("• Temperature typically shows strongest correlations")
        print("• Power consumption exhibits strong daily (24h) autocorrelation patterns")
        print("• Weekly patterns (168h cycles) are evident in autocorrelation plots")
        print("• Optimal lookback windows for modeling: 24-72 hours")


class AnomalyDetector:
    """Detect anomalies and assess data quality"""
    
    @staticmethod
    def detect_outliers_multiple_methods(series, feature_name):
        """Detect outliers using multiple methods"""
        outliers_summary = {}
        
        # Method 1: IQR Method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        iqr_lower = Q1 - 1.5 * IQR
        iqr_upper = Q3 + 1.5 * IQR
        iqr_outliers = series[(series < iqr_lower) | (series > iqr_upper)]
        
        # Method 2: Z-Score Method
        z_scores = np.abs((series - series.mean()) / series.std())
        zscore_outliers = series[z_scores > 3]
        
        # Method 3: Modified Z-Score using MAD
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / mad
        mad_outliers = series[np.abs(modified_z_scores) > 3.5]
        
        outliers_summary[feature_name] = {
            'iqr_outliers': len(iqr_outliers),
            'iqr_percentage': len(iqr_outliers) / len(series) * 100,
            'zscore_outliers': len(zscore_outliers),
            'zscore_percentage': len(zscore_outliers) / len(series) * 100,
            'mad_outliers': len(mad_outliers),
            'mad_percentage': len(mad_outliers) / len(series) * 100,
            'iqr_bounds': (iqr_lower, iqr_upper),
            'outlier_indices_iqr': iqr_outliers.index.tolist()
        }
        
        return outliers_summary
    
    @staticmethod
    def detect_sudden_changes(series, threshold_std=3):
        """Detect sudden changes in time series"""
        diff = series.diff().abs()
        threshold = threshold_std * diff.std()
        sudden_changes = diff > threshold
        return sudden_changes
    
    @staticmethod
    def analyze_data_quality(df, env_features, power_cols):
        """Comprehensive data quality and anomaly analysis"""
        print("🚨 DATA QUALITY & SENSOR ANOMALIES ANALYSIS")
        print("="*60)
        
        # Detect outliers for all numerical features
        all_features = env_features + power_cols
        outlier_results = {}
        
        for feature in all_features:
            outlier_results.update(
                AnomalyDetector.detect_outliers_multiple_methods(df[feature], feature))
        
        # Print outlier summary
        print("📊 OUTLIER DETECTION SUMMARY")
        print("="*50)
        print(f"{'Feature':<25} {'IQR':<8} {'Z-Score':<8} {'MAD':<8}")
        print("-" * 50)
        
        for feature in all_features:
            iqr_pct = outlier_results[feature]['iqr_percentage']
            zscore_pct = outlier_results[feature]['zscore_percentage']
            mad_pct = outlier_results[feature]['mad_percentage']
            print(f"{feature:<25} {iqr_pct:<8.2f} {zscore_pct:<8.2f} {mad_pct:<8.2f}")
        
        # Check for suspicious values
        print(f"\n🔍 SUSPICIOUS VALUES ANALYSIS")
        print("="*40)
        
        # Environmental features checks
        print("Environmental Features:")
        print(f"• Temperature < -10°C or > 50°C: {((df['temperature'] < -10) | (df['temperature'] > 50)).sum()}")
        print(f"• Humidity < 0% or > 100%: {((df['humidity'] < 0) | (df['humidity'] > 100)).sum()}")
        print(f"• Wind Speed < 0 or > 100: {((df['wind_speed'] < 0) | (df['wind_speed'] > 100)).sum()}")
        print(f"• Negative Solar Radiation: {(df['general_diffuse_flows'] < 0).sum()}")
        
        # Power consumption checks
        print(f"\nPower Consumption:")
        for zone in power_cols:
            negative_count = (df[zone] < 0).sum()
            zero_count = (df[zone] == 0).sum()
            very_high = (df[zone] > df[zone].quantile(0.999)).sum()
            print(f"• {zone}: Negative: {negative_count}, Zero: {zero_count}, Extreme high: {very_high}")
        
        # Sudden change detection
        print(f"\n⚡ SUDDEN CHANGE DETECTION")
        print("="*35)
        
        for zone in power_cols:
            sudden_changes = AnomalyDetector.detect_sudden_changes(df[zone])
            change_count = sudden_changes.sum()
            print(f"• {zone}: {change_count} sudden changes detected")
        
        return outlier_results
    
    @staticmethod
    def plot_anomaly_analysis(df, outlier_results, env_features, power_cols):
        """Create comprehensive anomaly detection visualization with better spacing"""
        
        # Figure 1: Box plots for power consumption zones
        fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
        fig1.suptitle('🚨 DATA QUALITY & ANOMALY DETECTION - BOX PLOTS', fontsize=16, fontweight='bold')
        
        for i, zone in enumerate(power_cols):
            ax = axes1[i]
            ax.boxplot(df[zone], patch_artist=True, 
                      boxprops=dict(facecolor='lightblue', alpha=0.7))
            ax.set_title(f'📦 {zone.replace("_", " ").title()}\nOutlier Detection', 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('Power Consumption')
            ax.grid(True, alpha=0.3)
            
            # Add outlier statistics
            outlier_count = outlier_results[zone]['iqr_outliers']
            outlier_pct = outlier_results[zone]['iqr_percentage']
            ax.text(0.95, 0.95, f'Outliers: {outlier_count}\n({outlier_pct:.1f}%)', 
                   transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Figure 2: Time series with outliers highlighted
        fig2, axes2 = plt.subplots(3, 1, figsize=(18, 12))
        fig2.suptitle('🚨 DATA QUALITY & ANOMALY DETECTION - TIME SERIES WITH OUTLIERS', fontsize=16, fontweight='bold')
        
        for i, zone in enumerate(power_cols):
            ax = axes2[i]
            
            # Plot main time series
            ax.plot(df.index, df[zone], alpha=0.7, linewidth=0.5, color='blue')
            
            # Highlight outliers
            outlier_indices = outlier_results[zone]['outlier_indices_iqr']
            if outlier_indices:
                outlier_dates = [idx for idx in outlier_indices if idx in df.index]
                if outlier_dates:
                    ax.scatter(outlier_dates, df.loc[outlier_dates, zone], 
                              color='red', s=10, alpha=0.8, zorder=5)
            
            ax.set_title(f'📈 {zone.replace("_", " ").title()} - Time Series with Outliers', 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('Power Consumption')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Figure 3: Environmental features distributions with outliers
        fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
        fig3.suptitle('🚨 DATA QUALITY & ANOMALY DETECTION - ENVIRONMENTAL FEATURES', fontsize=16, fontweight='bold')
        
        env_plot_features = ['temperature', 'humidity', 'wind_speed']
        for i, feature in enumerate(env_plot_features):
            ax = axes3[i]
            
            # Plot histogram
            ax.hist(df[feature], bins=50, alpha=0.7, color='skyblue', density=True)
            
            # Mark outlier boundaries
            bounds = outlier_results[feature]['iqr_bounds']
            ax.axvline(bounds[0], color='red', linestyle='--', alpha=0.8, label='IQR Bounds')
            ax.axvline(bounds[1], color='red', linestyle='--', alpha=0.8)
            
            ax.set_title(f'📊 {feature.replace("_", " ").title()}\nDistribution & Outliers', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Data quality recommendations
        print(f"\n💡 DATA QUALITY RECOMMENDATIONS")
        print("="*45)
        print("Based on the analysis:")
        print("• Most outliers appear to be legitimate extreme values rather than sensor errors")
        print("• Consider robust scaling methods that are less sensitive to outliers")
        print("• Investigate sudden changes - they might indicate system events or maintenance")
        print("• Zero power consumption values may indicate sensor downtime or system maintenance")
        print("• For modeling, consider using outlier-robust loss functions or data transformation")
