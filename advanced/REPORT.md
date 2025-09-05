# 🔴 PowerCast – Advanced Track

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

## 🛠️ Week 2: Feature Engineering & Deep Learning Preparation

### 🔄 1. Sequence Construction & Lookback Windows

Q: How did you determine the optimal lookback window size for your sequence models?  
A:

Q: What challenges did you face when converting the time-series data into input/output sequences?  
A:

Q: How did you handle cases where the lookback window extended beyond the available data?  
A:

---

### ⚖️ 2. Feature Scaling & Transformation

Q: Which normalization or standardization techniques did you apply to the features, and why?  
A:

Q: Did you engineer any cyclical time features (e.g., sine/cosine transforms for hour or day)? How did these impact model performance?  
A:

Q: How did you address potential data leakage during scaling or transformation?  
A:

---

### 🧩 3. Data Splitting & Preparation

Q: How did you split your data into training, validation, and test sets to ensure temporal integrity?  
A:

Q: What considerations did you make to prevent information leakage between splits?  
A:

Q: How did you format your data for use with PyTorch DataLoader or TensorFlow tf.data.Dataset?  
A:

---

### 📈 4. Feature-Target Alignment

Q: How did you align your input features and target variables for sequence-to-one or sequence-to-sequence forecasting?  
A:

Q: Did you encounter any issues with misalignment or shifting of targets? How did you resolve them?  
A:

---

### 🧪 5. Data Quality & Preprocessing

Q: What preprocessing steps did you apply to handle missing values or anomalies before modeling?  
A:

Q: How did you verify that your data pipeline produces consistent and reliable outputs for model training?  
A:

---

## ✅ Week 3: Neural Network Design & Baseline Training

---

### 🧠 1. Model Architecture & Design

Q: Which neural network architecture(s) did you choose for baseline forecasting (e.g., LSTM, GRU, TCN), and what motivated your selection?  
A:

Q: How did you structure your input sequences and targets for the chosen model(s)?  
A:

Q: What considerations did you make regarding the depth, number of units, and activation functions in your network?  
A:

---

### 🏋️ 2. Training & Experimentation

Q: Which loss function and optimizer did you use for training, and why are they suitable for this task?  
A:

Q: How did you incorporate regularization techniques such as Dropout or Batch Normalization, and what impact did they have?  
A:

Q: What challenges did you encounter during training (e.g., overfitting, vanishing gradients), and how did you address them?  
A:

---

### 📊 3. Evaluation & Metrics

Q: Which metrics did you use to evaluate your model’s performance, and why are they appropriate for time-series forecasting?  
A:

Q: How did you use MLflow (or another tool) to track your training experiments and results?  
A:

Q: What insights did you gain from visualizing forecasted vs. actual power consumption for each zone?  
A:

---

### 🔍 4. Model Interpretation & Insights

Q: How did you interpret the learned patterns or feature importance in your neural network?  
A:

Q: Did you observe any systematic errors or biases in your model predictions? How did you investigate and address them?  
A:

Q: What trade-offs did you consider when selecting your final baseline model architecture?

---

## ✅ Week 4: Model Optimization & Interpretability

### 🏗️ 1. Architecture Tuning & Experimentation

Q: Which architectural changes (e.g., depth, number of units, bidirectionality, dilation) did you experiment with, and why?  
A:

Q: How did you decide on the final architecture for your deep learning model?  
A:

Q: What impact did these changes have on model performance and training stability?  
A:

---

### ⏸️ 2. Training Strategies & Regularization

Q: How did you apply early stopping or learning rate scheduling during training?  
A:

Q: What regularization techniques (e.g., Dropout, Batch Normalization) did you use, and how did they affect results?  
A:

Q: How did you monitor and address overfitting or underfitting during optimization?  
A:

---

### 🧠 3. Model Interpretability

Q: Which interpretability methods (e.g., SHAP, saliency maps, attention plots) did you use to understand your model’s predictions?  
A:

Q: What insights did you gain about feature importance or temporal dependencies from these methods?  
A:

Q: How did interpretability findings influence your modeling or feature engineering decisions?  
A:

---

### 📊 4. Error Analysis & Residuals

Q: How did you analyze residuals and error distributions across different zones?  
A:

Q: Did you identify any systematic errors or biases in your model predictions? How did you address them?  
A:

Q: What steps did you take to ensure robust evaluation and fair comparison of model performance across different configurations?
