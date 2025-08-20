import numpy as np

def add_cyclical_time_features(df):
    """
    Adds cyclical time features to the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with a time column.
    time_column (str): Name of the column containing time data.
    
    Returns:
    pd.DataFrame: DataFrame with cyclical time features added.
    """
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    

    # Cyclical transforms
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Optional: drop raw time columns
    df = df.drop(columns=['hour', 'day_of_week', 'month'])
    
    return df