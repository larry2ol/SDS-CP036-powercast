from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def scale_features(train_df, test_df, target_col):
    """
    Fits a scaler on training features (excluding target) 
    and applies it to both train & test.
    """
    feature_cols = [col for col in train_df.columns 
                    if col != target_col and not pd.api.types.is_datetime64_any_dtype(train_df[col])]

    train_df = train_df.copy()
    test_df = test_df.copy()

    for df in [train_df, test_df]:
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    extra_numeric_cols = ['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']
    num_cols = [col for col in train_df[feature_cols].select_dtypes(include='number').columns 
                if 'Power Consumption' in col and ('lag' in col or 'roll' in col)
                or col in ['hour', 'month', 'weekday', 'is_weekend']
                or col in extra_numeric_cols]

    scaler = StandardScaler()

    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])

    for df in [train_df, test_df]:
        df.dropna(subset=feature_cols + [target_col], inplace=True)
        df.reset_index(drop=True, inplace=True)

    return train_df, test_df, scaler, num_cols

