# 🔴 PowerCast – Advanced Track

## ✅ Week 1: Setup & Exploratory Data Analysis (EDA)

---

### 🧭 1. Time Consistency & Structure

Q: Are there any missing or irregular timestamps in the dataset? How did you verify consistency?
A: There are no missing or irregular timestamps in the dataset. This was verified using the deltas.unique() command, which only returned one value, meaning there is only one time difference between timestamps.

Q: What is the sampling frequency and are all records spaced consistently?
A: All records are spaced 10 minutes apart.

Q: Did you encounter any duplicates or inconsistent `DateTime` entries?
A: There were no inconsistent entries, nor duplicates. The absence of duplicates was verified using: df["datetime"].duplicated().sum().

---

### 📊 2. Temporal Trends & Seasonality

Q: What daily or weekly patterns are observable in power consumption across the three zones?
A: (Did not gain enough insight to comment yet).

Q: Are there seasonal or time-of-day peaks and dips in energy usage?
A: Sharp peak of energy consumption in the summer for Zone 3.

Q: Which visualizations helped you uncover these patterns?
A: (Although not all patterns have been uncovered yet) Line plots.

(Note to self: I want to create 'daytime' and 'nighttime' daily average power consumption and identify any trends)

---

### 🌦️ 3. Environmental Feature Relationships

Q: Which environmental variables (temperature, humidity, wind speed, solar radiation) correlate most with energy usage?
A: Temperature and Humidity showed strong positive and negative correlations, respectively, with energy usage across all three zones.

Q: Are any variables inversely correlated with demand in specific zones?
A: Humidity

Q: Did your analysis differ across zones? Why might that be?
A: Correlations were not as strong in zone 3, not sure of the reason for this, may be explored further.

---

### 🌀 4. Lag Effects & Time Dependency
(Not done yet...)

Q: Did you observe any lagged effects where past weather conditions predict current power usage?
A:

Q: How did you analyze lag (e.g., shifting features, plotting lag correlation)?
A:

Q: What lag intervals appeared most relevant and why?
A:

---

### ⚠️ 5. Data Quality & Sensor Anomalies
(Not done yet...)

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
