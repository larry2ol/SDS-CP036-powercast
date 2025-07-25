# Welcome to the SuperDataScience Community Project!
Welcome to the **PowerCast: Forecasting Regional Power Consumption in Tetouan Using Weather & Environmental Data** repository! 🎉

This project is a collaborative initiative brought to you by SuperDataScience, a thriving community dedicated to advancing the fields of data science, machine learning, and AI. We are excited to have you join us in this journey of learning, experimentation, and growth.

To contribute to this project, please follow the guidelines avilable in our [CONTRIBUTING.md](CONTRIBUTING.md) file.

# Project Scope of Works:

## Project Overview
**PowerCast** is a time-series regression project aimed at predicting the power consumption in three urban zones of Tetouan City, Morocco, using environmental conditions such as temperature, humidity, wind speed, and solar radiation metrics.

The dataset spans over 52,000 time-stamped records, sampled every 10 minutes, making it an excellent candidate for both traditional time-series modeling and modern deep learning approaches such as sequence-to-one forecasting.

Participants will develop models that forecast electricity demand in each zone, helping to simulate how weather fluctuations influence energy usage patterns. These models can be crucial for infrastructure planning, energy efficiency strategies, and grid stability management.

This project is structured into two experience tracks:

- 🟢 **Beginner Track** – Apply feature-based regression models (e.g., Linear Regression, Random Forest, XGBoost) to predict power consumption for each zone individually; deploy results using Streamlit.
- 🔴 **Advanced Track** – Build time-aware deep learning models (e.g., LSTM, GRU, or Temporal Convolutional Networks) to learn sequence patterns and forecast future consumption across zones; deploy using Docker or Hugging Face Spaces.


Link to dataset: https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city


## 🟢 Beginner Track

### Week 1: Setup & Exploratory Data Analysis (EDA)

**1. Data Cleaning & Preparation**
- Convert 'DateTime' column to proper datetime format
- Check for missing timestamps or duplicated entries
- Verify data is sampled uniformly at 10-minute intervals

**2. Time Indexing & Sorting**
- Set 'DateTime' as the index and sort chronologically
- Create separate columns for hour, day, month, and weekday if relevant

**3. Visualization & Trend Detection**
- Visualize power consumption across each zone over time
- Plot daily/weekly trends and seasonality components
- Compare consumption patterns with weather features

**4. Statistical Summary**
- Summary statistics of environmental variables (temperature, humidity, wind speed, solar radiation)
- Detect anomalies or outliers in sensor readings

**5. Correlation & Lag Analysis**
- Correlate weather variables with each zone’s power consumption
- Perform lag correlation to assess delayed effects of temperature, humidity, etc.

### Week 2: Feature Engineering & Data Preprocessing

**1. Time-based Features**
- Extract temporal features: hour, day of week, month, is_weekend, season
- Encode cyclical features using sine/cosine transformations

**2. Lag Features & Rolling Windows**
- Generate lag features (e.g., power consumption lagged by 1, 2, 3 hours)
- Compute rolling averages and standard deviations (e.g., past 1 hour, 6 hours)

**3. Feature Consolidation**
- Remove highly collinear or non-informative features
- Normalize/scale numerical features (e.g., weather metrics, lags)

**4. Train-Test Splitting**
- Perform chronological split (e.g., last 2 weeks as test set)
- Avoid data leakage by not shuffling time-series data

### Week 3: Model Development & Experimentation

**1. Baseline Models**
- Train Linear Regression models for each power consumption zone
- Evaluate with metrics such as MAE, RMSE, and R²

**2. Tree-based Models**
- Implement Random Forest and XGBoost regressors
- Log experiments using MLflow with model versioning and evaluation metrics

**3. Evaluation**
- Plot prediction vs actuals for each zone
- Use residual plots and error distributions to diagnose issues

**4. Cross-validation**
- Perform TimeSeriesSplit for cross-validation
- Compare models across validation folds

### Week 4: Hyperparameter Tuning & Model Selection

**1. Optimization**
- Tune key hyperparameters for tree-based models using RandomizedSearchCV or Optuna

**2. Final Model Packaging**
- Select best-performing model for each zone
- Retrain using entire training data
- Save preprocessing logic and models

**3. Validation & Testing**
- Evaluate final models on holdout test set
- Create summary report of performance for all zones

### Week 5: Model Deployment

**1. Streamlit app**
- Build a simple Streamlit dashboard that forecasts power consumption for each zone
- Include dropdown to select zone and forecast period (next 1–6 hours)
- Host app on Streamlit Community Cloud


## 🔴 Advanced Track

### Week 1:

Same as Beginner Track — perform full EDA as outlined previously.

### Week 2:

Same as Beginner Track, with the following additions:
- **Windowing Strategy**: Define time windows for supervised learning (e.g., past 6 hours to predict next hour)
- **Sequence Construction**: Reshape dataset into 3D format for sequence models (samples × timesteps × features)
- **Padding & Batching**: Build PyTorch/TensorFlow `Dataset` pipelines with shuffling, batching, and windowing

### Week 3: Model Architecture & Training

**1. Model Design**
- Build a base LSTM/GRU or TCN architecture for univariate and multivariate forecasting
- Define input/output shape for sequence-to-one prediction per zone

**2. Training Setup**
- Use early stopping, learning rate schedulers, and appropriate loss function (e.g., MSE)
- Validate on chronological validation set

**3. MLflow Integration**
- Log metrics, parameters, model checkpoints, and training curves to MLflow

### Week 4: Hyperparameter Tuning & Explainability

**1. Architecture Tuning**
- Tune hidden units, layers, dropout, sequence length, and optimizer settings
- Use Optuna or manual sweeps with validation metrics

**2. Model Evaluation**
- Visualize predictions vs actual values across zones
- Calculate MAE, RMSE, and R² for each zone

**3. Model Interpretation**
- Use attention mechanisms, permutation feature importance, or SHAP for interpretability (optional)

### Week 5: Model Deployment

#### 🟢 Easy Track – Streamlit App

- Same as Beginner Track

#### 🟡 Intermediate Track – Docker + Hugging Face

- Dockerize Streamlit interface and push to Hugging Face Spaces
- Ensure compatibility with custom inference script

#### 🔴 Advanced Track – API-based Deployment

- Build a REST API using FastAPI or Flask
- Containerize the API with Docker
- Deploy to Render, Railway, Fly.io, or GCP Cloud Run

## 📅 Project Timeline Overview

| Phase                        | General Activities                                                       | Week   |
| ---------------------------- | ------------------------------------------------------------------------ | ------ |
| Phase 1: Setup + EDA         | Clean, explore, and visualize the data                                   | Week 1 |
| Phase 2: Feature Engineering | Transform features, encode variables, handle imbalance, prepare splits   | Week 2 |
| Phase 3: Model Development   | Train baseline models or neural networks and run initial experiments     | Week 3 |
| Phase 4: Model Optimization  | Tune models, evaluate performance, and apply interpretability techniques | Week 4 |
| Phase 5: Deployment          | Deploy via Streamlit or Docker-based approaches depending on difficulty  | Week 5 |


