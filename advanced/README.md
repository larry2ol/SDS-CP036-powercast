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

