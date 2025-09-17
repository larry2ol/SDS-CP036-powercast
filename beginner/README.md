# 🟢 PowerCast Beginner Track

Welcome to the **Beginner Track** of the PowerCast project! This track is designed for participants who want to apply traditional machine learning techniques to a time-series problem and build interpretable models that forecast electricity usage across different urban zones.

You’ll work with 10-minute interval weather and power consumption data for Tetouan City, Morocco, and follow a complete ML pipeline—from time-series EDA and lag feature engineering to model deployment using Streamlit.

---

## 📊 Dataset Overview

* **Source**: [UCI Tetouan Power Consumption Dataset](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city)
* **Goal**: Predict power consumption in 3 urban zones using weather conditions
* **Features**: Temperature, Humidity, Wind Speed, Solar Radiation, Datetime (timestamped every 10 minutes)

---

## 🎓 Weekly Breakdown

### ✅ Week 1: Setup & Exploratory Data Analysis (EDA)

* Parse the `DateTime` column, check frequency and consistency
* Visualize trends in power consumption across zones
* Compare with environmental features using scatter and line plots
* Analyze lag effects and missing timestamps
* Document top 3–5 insights in your report

### ✅ Week 2: Feature Engineering & Preprocessing

* Create time-based features (hour, weekday, weekend, month)
* Engineer lag and rolling statistics per zone (mean, std, median, etc.)
* Normalize numerical features
* Perform chronological train/test split

### ✅ Week 3: Model Development & Experimentation

* Train Linear Regression, Random Forest, and XGBoost models
* Build separate models for each zone or a multi-output model
* Track experiments and metrics (RMSE, MAE, R²) using MLflow
* Visualize actual vs predicted curves for each zone

### ✅ Week 4: Model Tuning & Finalization

* Tune hyperparameters using `GridSearchCV` or `Optuna`
* Retrain best models on full training data
* Validate using test set and compare performance
* Document model trade-offs and final selections

### ✅ Week 5: Deployment

* Build a Streamlit app that allows:

  * User to select zone and forecast horizon
  * Input of recent weather data
  * Visualization of prediction results
* Deploy the app to Streamlit Community Cloud

---

## 🗒️ Project Timeline Overview

| Phase                           | General Activities                                                     |
| ------------------------------- | ---------------------------------------------------------------------- |
| **Week 1: Setup + EDA**         | Clean and inspect time-series data, uncover patterns and relationships |
| **Week 2: Feature Engineering** | Lag features, rolling stats, encode time, normalize data               |
| **Week 3: Model Development**   | Train regression models and compare zone-wise performance              |
| **Week 4: Model Optimization**  | Tune hyperparameters and finalize your best models                     |
| **Week 5: Deployment**          | Build and host your Streamlit power forecasting app                    |

---

## 📃 Report Template

Use the [REPORT.md](./REPORT.md) to document your weekly progress, code reasoning, visualizations, and results.

---

## 🚪 Where to Submit

Please place your work inside the appropriate folder:

* `submissions/team-members/your-name/` if you are part of the official project team
* `submissions/community-contributions/your-name/` if you are an external contributor

Refer to the [CONTRIBUTING.md](../CONTRIBUTING.md) for complete instructions.
