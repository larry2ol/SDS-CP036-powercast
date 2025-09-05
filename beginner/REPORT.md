# 🟢 PowerCast – Beginner Track

## ✅ Week 1: Setup & Exploratory Data Analysis (EDA)

---

### 🧭 1. Time Consistency & Structure

Q: Are there any missing or irregular timestamps in the dataset? How did you verify consistency?
A:

Q: What is the sampling frequency and are all records spaced consistently?
A:

Q: Did you encounter any duplicates or inconsistent `DateTime` entries?
A:

---

### 📊 2. Temporal Trends & Seasonality

Q: What daily or weekly patterns are observable in power consumption across the three zones?
A:

Q: Are there seasonal or time-of-day peaks and dips in energy usage?
A:

Q: Which visualizations helped you uncover these patterns?
A:

---

### 🌦️ 3. Environmental Feature Relationships

Q: Which environmental variables (temperature, humidity, wind speed, solar radiation) correlate most with energy usage?
A:

Q: Are any variables inversely correlated with demand in specific zones?
A:

Q: Did your analysis differ across zones? Why might that be?
A:

---

### 🌀 4. Lag Effects & Time Dependency

Q: Did you observe any lagged effects where past weather conditions predict current power usage?
A:

Q: How did you analyze lag (e.g., shifting features, plotting lag correlation)?
A:

Q: What lag intervals appeared most relevant and why?
A:

---

### ⚠️ 5. Data Quality & Sensor Anomalies

Q: Did you detect any outliers in the weather or consumption readings?
A:

Q: How did you identify and treat these anomalies?
A:

Q: What might be the impact of retaining or removing them in your model?
A:

---

## ✅ Week 2: Feature Engineering & Preprocessing

### 🕒 1. Time-Based Feature Engineering

Q: Which time-based features did you create (e.g., hour, weekday, weekend, month), and why did you select them?  
A:

Q: How did these new features help capture patterns in power consumption?  
A:

Q: Did you encounter any challenges when extracting or encoding time features? How did you address them?  
A:

---

### 🔁 2. Lag and Rolling Statistics

Q: How did you determine which lag features and rolling statistics (mean, std, median, etc.) to engineer for each zone?  
A:

Q: What impact did lag and rolling features have on model performance or interpretability?  
A:

Q: How did you handle missing values introduced by lag or rolling computations?  
A:

---

### ⚖️ 3. Feature Scaling & Normalization

Q: Which normalization or scaling techniques did you apply to your numerical features, and why?  
A:

Q: How did you ensure that scaling was performed without introducing data leakage?  
A:

Q: Did you notice any features that required special treatment during normalization?  
A:

---

### 🧩 4. Data Splitting & Preparation

Q: How did you split your data into training and test sets to maintain chronological order?  
A:

Q: What steps did you take to prevent information leakage between splits?  
A:

Q: How did you verify that your train/test split was appropriate for time-series forecasting?  
A:

---

### 🧪 5. Data Quality & Preprocessing

Q: What preprocessing steps did you apply to handle missing values or anomalies before modeling?  
A:

Q: How did you validate that your feature engineering and preprocessing pipeline produced consistent and reliable results across different data subsets?  
A:

---

## ✅ Week 3: Model Development & Experimentation

---

### 🤖 1. Model Selection & Training

Q: Which machine learning models did you choose for forecasting power consumption, and what motivated your selections?  
A:

Q: How did you structure your models to handle the multi-zone prediction task (separate models vs. multi-output)?  
A:

Q: What challenges did you encounter during model training, and how did you address them?  
A:

---

### 📈 2. Experiment Tracking & Evaluation

Q: Which evaluation metrics did you use to assess model performance, and why are they appropriate for this problem?  
A:

Q: How did you use MLflow (or another tool) to track your experiments and results?  
A:

Q: What insights did you gain from comparing actual vs. predicted curves for each zone?  
A:

---

### 🔍 3. Model Interpretation & Insights

Q: How did you interpret feature importance or model coefficients, and what did they reveal about power consumption drivers?  
A:

Q: Did you observe any systematic errors or biases in your model predictions? How did you investigate and address them?  
A:

Q: What trade-offs did you consider when selecting your final model(s) for each zone?  
A:

---

## ✅ Week 4: Model Tuning & Finalization

### 🛠️ 1. Hyperparameter Optimization

Q: Which hyperparameter tuning methods (e.g., GridSearchCV, Optuna) did you use, and why?  
A:

Q: What were the most important hyperparameters for each model, and how did you select their ranges?  
A:

Q: How did you ensure that your tuning process did not introduce data leakage or overfitting?  
A:

---

### 🔁 2. Retraining & Validation

Q: How did you retrain your best models on the full training data after tuning?  
A:

Q: What steps did you take to validate model performance on the test set?  
A:

Q: Did you observe any changes in performance after retraining? How did you interpret these results?  
A:

---

### ⚖️ 3. Model Selection & Trade-offs

Q: What trade-offs did you consider when selecting your final model(s) for each zone?  
A:

Q: How did you compare and document the performance of different models?  
A:

Q: What factors (e.g., interpretability, accuracy, computational cost) influenced your model selection?  
A:

---

### 📦 4. Model Export & Deployment

Q: How did you export your final model(s) for deployment, and what format did you use?  
A:

Q: What considerations did you take into account for deploying models in a production environment?  
A:

Q: Did you create any custom functions or classes for inference? If so, please describe them.  
A:

---

### 🚀 5. Monitoring & Maintenance Planning

Q: How do you plan to monitor model performance and data quality in production?  
A:

Q: What thresholds or indicators will prompt a model retraining or update?  
A:

Q: How will you handle potential data drift or changes in underlying patterns?  
A:
