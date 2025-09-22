# 🔴 PowerCast Advanced Track

Welcome to the **Advanced Track** of the PowerCast project! This track is for participants ready to apply deep learning techniques to time-series forecasting. You’ll design time-aware neural networks that can learn patterns in environmental data to predict power consumption across Tetouan City's three urban zones.

You’ll build your models using tools like PyTorch or TensorFlow, experiment with temporal features, and deploy your deep learning solution using advanced options like Docker or API endpoints.

---

## 📊 Dataset Overview

* Source: [UCI Tetouan Power Consumption Dataset](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city)
* Goal: Forecast electricity usage in three urban zones using time-series weather data
* Features: Timestamp, Temperature, Humidity, Wind Speed, Solar Radiation

---

## 🎓 Weekly Breakdown

### Week 1: Exploratory Data Analysis (EDA)

* Perform same EDA steps as in the [beginner track](../beginner/README.md)
* Investigate autocorrelation, lag relationships, and seasonality
* Explore feature-target alignment and potential for lookback windows

### Week 2: Feature Engineering & Deep Learning Prep

* Create lookback windows for sequence-to-one or sequence-to-sequence modeling
* Normalize continuous variables and optionally engineer cyclical time features (e.g., sine/cosine transforms of hour/week)
* Convert data into tensors and format train/val/test splits
* Prepare PyTorch `DataLoader` or TensorFlow `tf.data.Dataset` objects

### Week 3: Neural Network Design & Baseline Training

* Build a baseline sequence model using LSTM, GRU, or TCN
* Include Dropout, Batch Normalization, and ReLU activations
* Train the model using MAE or MSE loss and an optimizer like Adam
* Track training metrics using MLflow
* Evaluate using RMSE, MAE, R², and visualize forecasts vs. actuals

### Week 4: Model Optimization & Interpretability

* Experiment with different architectures (depth, units, bidirectionality, dilation, etc.)
* Apply early stopping and learning rate schedulers
* Optionally integrate SHAP, saliency maps, or attention plots for interpretability
* Analyze residuals and error distributions across zones

### Week 5: Deployment

* 🟢 Easy: **Streamlit Cloud**

  * Build a simple app that takes recent weather input and outputs forecasts
  * Host it on Streamlit Community Cloud

* 🟡 Intermediate: **Docker + Hugging Face Spaces**

  * Containerize the app using Docker
  * Deploy using Docker SDK on Hugging Face Spaces

* 🔴 Advanced: **API-based Deployment (Flask or FastAPI)**

  * Create a RESTful API that receives time-series input and returns predictions
  * Deploy via Docker to services like Railway, Render, or GCP Cloud Run
  * Test using Postman or build a simple client UI

---

## 🗒️ Project Timeline Overview

| Phase                           | General Activities                                                |
| ------------------------------- | ----------------------------------------------------------------- |
| **Week 1: Setup + EDA**         | Analyze structure, lagged effects, and zone-wise trends           |
| **Week 2: Feature Engineering** | Build input sequences, normalize, and prep for deep learning      |
| **Week 3: Model Development**   | Train LSTM/GRU/TCN models and evaluate predictions                |
| **Week 4: Model Optimization**  | Tune architecture, interpret model behavior, and finalize results |
| **Week 5: Deployment**          | Choose from three levels of deployment and publish your model     |

---

## 📃 Report Template

Use the [REPORT.md](./REPORT.md) to document your process, model architecture, evaluation, and deployment steps.

---

## 🚪 Where to Submit

Place your work inside the appropriate folder:

* `submissions/team-members/your-name/` if you are part of the official project team
* `submissions/community-contributions/your-name/` if you are an external contributor

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for full instructions.


### Features
Forecast energy consumption for Zone A, Zone B, and Zone C
Supports both 10-minute, hourly, daily and weekly prediction intervals
Scalable backend powered by FastAPI (or Flask)
Dockerized for easy deployment
RESTful API endpoints for integration


### Model Architecture
Different model architecture was used depending on the task. Daily and weekly forecasts share a typical model architecture. In contrast, hourly forecasts due to required granularity demand a different architecture to achieve high accuracy despite best efforts to use the same architecture

### Model Evaluation 
RMSE, MAE and R2-Score metrics were used for the model evaluation: the different metric gives different perspectives.
RMSE measures the average magnitude of error. Penalises larger errors more heavily. Better fit == Lower RMSE 
MAE measures the average absolute difference between predicted and actual values. More robust to outliers than RMSE. Better fit == Lower MAE
R2-Score: 1.0: Perfect prediction.Higher R² = better model performance

### Deployment
The application was deployed using Render at: Live URL: https://energy-consumption-forecasting-euwq.onrender.com

### API Testing
The /predict endpoint was tested using Postman with a POST request to:

### Sample JSON Payload
json 

{ "input_data": 
[ 
  [6.559, 73.8, 0.083, 0.051, 0.119, 34055.69620, 16128.87538, 20240.96386], 
  [6.414, 74.5, 0.083, 0.070, 0.085, 29814.68354, 19375.07599, 20131.08434], 
  [6.313, 74.5, 0.080, 0.062, 0.100, 29128.10127, 19006.68693, 19668.43373], 
  [6.121, 75.0, 0.083, 0.091, 0.096, 28228.86076, 18361.09422, 18899.27711], 
  [5.921, 75.7, 0.081, 0.048, 0.085, 27335.69620, 17872.34043, 18442.40964], 
  [5.780, 76.5, 0.080, 0.082, 0.094, 26581.89873, 17539.01215, 18048.79518], 
  [5.750, 77.0, 0.083, 0.065, 0.080, 25930.69620, 17209.27051, 17764.09638], 
  [5.690, 77.7, 0.081, 0.086, 0.091, 25290.12658, 16901.39817, 17445.78313], 
  [5.600, 78.4, 0.080, 0.061, 0.077, 24729.49367, 16662.29482, 17182.65060], 
  [5.450, 79.6, 0.083, 0.054, 0.068, 24224.17721, 16457.34340, 16899.27711] 
]
  , "window_size": 10, 
  "n_features": 8 
} 

- This payload represents a 10-step time window with 8 features per step. The model returns predicted energy consumption values for Zone A, Zone B, and Zone C.

### Sample Response
{ "predictions": [ [ 24485.77898039471, 16534.77305121688, 16684.36364400194 ] ] }

- This response represents the predicted energy consumption for Zone A, Zone B, and Zone C, respectively.
