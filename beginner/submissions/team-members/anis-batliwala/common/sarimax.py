import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def sarimax_fast(zone, dataset_train, dataset_test, exog_vars):
    """
    Fast SARIMAX for testing parameter setup.
    """
    # Sample data to hourly (6x reduction) - remove this if you must keep 10-min resolution
    dataset_train = dataset_train.resample('3H').mean()
    dataset_test = dataset_test.resample('3H').mean()
    
    endog_train = dataset_train[zone].dropna()
    exog_train = dataset_train[exog_vars].dropna()
    endog_test = dataset_test[zone].dropna()
    exog_test = dataset_test[exog_vars].dropna()

    # Ensure alignment after dropping NAs
    common_idx = endog_train.index.intersection(exog_train.index)
    endog_train = endog_train.loc[common_idx]
    exog_train = exog_train.loc[common_idx]

    # SARIMAX model with small orders
    model = SARIMAX(
        endog=endog_train,
        exog=exog_train,
        order=(1, 0, 1),            # Small non-seasonal part
        seasonal_order=(0, 1, 1, 24),  # Reduced seasonal part (daily for hourly data)
        enforce_stationarity=False,
        enforce_invertibility=False,
        simple_differencing=True    # Faster estimation
    )

    # Fit with optimized settings
    sarimax_result = model.fit(
        disp=False,
        method='lbfgs',
        maxiter=50,                # Reduced iterations
        low_memory=True,            # Reduce memory usage
        cov_type='none'            # Skip covariance calculation
    )
    # Forecasting
    forecast = sarimax_result.get_forecast(
        steps=len(endog_test),
        exog=exog_test.iloc[:len(endog_test)]  # Ensure same length
    )
    y_pred = forecast.predicted_mean

    # Metrics
    mse = mean_squared_error(endog_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(endog_test, y_pred)
    r2 = r2_score(endog_test, y_pred)

    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R2'],
        'Value': [rmse, mae, r2]
    })

    return zone, metrics_df
