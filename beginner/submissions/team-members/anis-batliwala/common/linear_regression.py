from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

def baseline_model_performance(train_dataset, test_dataset, target_col, feature_cols):
    numeric_feature_cols = [
        col for col in feature_cols 
        if pd.api.types.is_numeric_dtype(train_dataset[col])
    ]

    X_train = train_dataset[numeric_feature_cols].copy()
    y_train = train_dataset[target_col].copy()
    X_test = test_dataset[numeric_feature_cols].copy()
    y_test = test_dataset[target_col].copy()

    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]
    X_test = X_test.dropna()
    y_test = y_test.loc[X_test.index]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R2'],
        'Value': [
            rmse,
            mean_absolute_error(y_test, y_pred),
            model.score(X_test, y_test)
        ]
    })

    coef_df = (
        pd.DataFrame({'Feature': numeric_feature_cols, 'Coefficient': model.coef_})
        .sort_values(by='Coefficient', key=abs, ascending=False)
    )

    return {
        'metrics': metrics_df,
        'coefficients': coef_df,
        'model': model
    }

