# 🔴 PowerCast – Beginner Track

## ✅ Week 2: Feature Engineering & Preprocessing

### 🕒 1. Time-Based Feature Engineering

Q: Which time-based features did you create (e.g., hour, weekday, weekend, month), and why did you select them?  
A: I created hour, day of week, month and is_weekend features as they will help uncover temporal patterns in  power consumption, such as daily cycles, weekly routines and seasonal variations. They allow for detailed analysis of how consumption changes by hour, day, and month, which is crucial for understanding demand drivers and for forecasting.

Q: How did these new features help capture patterns in power consumption?  
A: The new time-based features (hour, day of week, month, is_weekend) revealed clear consumption patterns tied to daily routines, work schedules, and seasonal effects. For example, power usage was higher during business hours and weekdays, while weekends showed a noticeable drop, especially in zone 3. These features made it easier to identify peak demand periods and understand how consumption varies across different timescales.

Q: Did you encounter any challenges when extracting or encoding time features? How did you address them?  
A: No specific challenges since data was clean and had no missing timestamps.

---

### 🔁 2. Lag and Rolling Statistics

Q: How did you determine which lag features and rolling statistics (mean, std, median, etc.) to engineer for each zone?  
A: We used autocorrelation (ACF) and partial autocorrelation (PACF) plots for each zone to identify relevant temporal dependencies and seasonal patterns. The strong daily seasonality and significant autocorrelation at short lags motivated us to engineer immediate lag features (e.g., lag 1), daily seasonal lags (lag 144 for 10-min data), and rolling statistics (mean, std, min, max, median) over windows corresponding to 1 hour, 6 hours, and 1 day. This approach ensures that both short-term and long-term temporal patterns are captured for each zone.

Q: What impact did lag and rolling features have on model performance or interpretability?  
A: Including lag and rolling features significantly improved model performance by allowing models to capture recent trends, seasonality, and local variability in power consumption. These features were among the most influential predictors in linear models, as indicated by their high absolute coefficients. They also enhanced interpretability by providing direct insights into how recent and periodic consumption patterns affect future values.


Q: How did you handle missing values introduced by lag or rolling computations?  
A: Missing values resulting from lag and rolling computations (due to shifting and windowing) were handled by dropping all rows with NaNs after feature engineering. This ensures that the final dataset used for modeling contains only complete cases, preventing data leakage and ensuring robust model training.

---

### ⚖️ 3. Feature Scaling & Normalization

Q: Which normalization or scaling techniques did you apply to your numerical features, and why?  
A: I applied standardization (z-score scaling) to all numerical features using `StandardScaler` from scikit-learn. This technique transforms features to have zero mean and unit variance, which is important for linear models to ensure that all features contribute equally and to improve model convergence. Standardization was applied to both raw features (e.g., temperature, humidity) and engineered features (e.g., lagged values, rolling statistics).

Q: How did you ensure that scaling was performed without introducing data leakage?  
A: To prevent data leakage, the scaler was always fit only on the training data within each fold of the time series split. The fitted scaler was then used to transform the corresponding test set. This approach ensures that information from the test set does not influence the scaling parameters used for the training data, preserving the integrity of the evaluation.

Q: Did you notice any features that required special treatment during normalization?  
A: Most numerical features were standardized directly. However, cyclical features (such as Hour_sin, Hour_cos, DayOfWeek_sin, DayOfWeek_cos) were already bounded between -1 and 1 due to their sine/cosine transformation and did not require further scaling. Additionally, binary features like `is_weekend` were left as-is, since scaling is not necessary for binary indicators.

---

### 🧩 4. Data Splitting & Preparation

Q: How did you split your data into training and test sets to maintain chronological order?  
A: I used TimeSeriesSplit to create chronological train/test splits preserving temporal order without shuffling.

Q: What steps did you take to prevent information leakage between splits?  
A: I fit scalers and feature transformations only on training data and applied them to test data without refitting.

Q: How did you verify that your train/test split was appropriate for time-series forecasting?  
A: By ensuring no future data was used in training and evaluating models only on unseen, chronologically later test sets.

---

### 🧪 5. Data Quality & Preprocessing

Q: What preprocessing steps did you apply to handle missing values or anomalies before modeling?  
A: I dropped missing rows introduced by lag/rolling features.

Q: How did you validate that your feature engineering and preprocessing pipeline produced consistent and reliable results across different data subsets?  
A: I fitted the model on five different train/test splits, and observed consistent model performance across these splits. This consistency indicates that the feature engineering and preprocessing pipeline is robust and produces reliable results on different subsets of the data.