from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def scale_continious_features(df, continious_cols, **kwargs):
    """
    Scales the specified continuous features in the DataFrame using StandardScaler.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with features to scale.
    continious_cols (list): List of column names to scale.
    scaler (StandardScaler, optional): Pre-fitted scaler. If None, a new one is created.
    
    Returns:
    pd.DataFrame: DataFrame with scaled features.
    """
    
    scaler = kwargs.get("scaler", None)
    if scaler is None:
        scaler = StandardScaler()
    
    if continious_cols is None:
        raise ValueError("Please provide a list of continuous columns to scale.")
    
    # Ensure the columns are numeric and handle any non-numeric values
    for col in continious_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Fit and transform the scaler on the specified columns
    df[continious_cols] = scaler.fit_transform(df[continious_cols])
    
    return df
