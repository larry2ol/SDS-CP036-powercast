# 🔴 PowerCast – Advanced Track

## ✅ Week 1: Setup & Exploratory Data Analysis (EDA)

---

### 🧭 1. Time Consistency & Structure

Q: Are there any missing or irregular timestamps in the dataset? How did you verify consistency?
A: No missing or irregular timestamps were found. Evidence: `Inferred frequency: 10min` and `Number of observations: 52416` (printed), and `Duplicate timestamps: 0` (printed). I verified by sorting `datetime`, computing `df["datetime"].diff().dropna()`, and comparing against a reference `pd.date_range(..., freq=modal_delta)` for coverage.

Q: What is the sampling frequency and are all records spaced consistently?
A: 10 minutes. Confirmed by computing `deltas.unique()` which returned `<TimedeltaArray> ['0 days 00:10:00']` only, meaning the 'only difference between any two timestamps' is 10 minutes.

Q: Did you encounter any duplicates or inconsistent `DateTime` entries?
A: No inconsistencies or duplicates. Printed `Duplicate timestamps: 0`.

---

### 📊 2. Temporal Trends & Seasonality

Q: What daily or weekly patterns are observable in power consumption across the three zones?
A: Clear diurnal (daily) cycles across zones: lows overnight (≈03:00–05:00), ramp-up mornings, and evening peaks (≈18:00–21:00). Weekly seasonality appears in the printed day‑of‑week means (see "Daily patterns" printout in the Temporal Patterns section): Zones 1–2 have lower weekends vs. weekdays; Zone 3’s weekend effect is weaker.

Q: Are there seasonal or time-of-day peaks and dips in energy usage?
A: Yes. Zone 3 exhibits a pronounced summer peak (likely cooling loads). Evenings are the daily maxima in all zones; late-night hours are minima. This is visible in the `Daily Mean Power Consumption` figure and time‑series plots.

Q: Which visualizations helped you uncover these patterns?
A: Time‑series line plots including `Daily Mean Power Consumption`, the ACF/PACF panels titled `ACF - PZ1/PZ2/PZ3` and `PACF - PZ1/PZ2/PZ3`, and the printed "Daily patterns" table (hour/day‑of‑week means). These collectively highlighted diurnal and weekly seasonality and the summer peak in Zone 3.


---

### 🌦️ 3. Environmental Feature Relationships

Q: Which environmental variables (temperature, humidity, wind speed, solar radiation) correlate most with energy usage?
A: Based on the printed `CORRELATION SUMMARY TABLE`: strongest positives are solar derivatives and temperature. For example, PZ1: GDF r=0.608 (lag 48 ≈ 8h), temp r=0.476 (lag 12 ≈ 2h); PZ2: GDF r=0.482 (lag 48), temp r=0.408 (lag 12); PZ3: GDF r=0.626 (lag 48), temp r=0.582 (lag 24 ≈ 4h). Wind speed is weaker but positive (e.g., PZ3 r=0.293, lag 24). Humidity effects are modest: positive in PZ1/PZ2 (r≈0.19–0.23 at lag 72), near‑zero to slightly negative in PZ3 (r=−0.020 at lag 72; short lags show more negative values).

Q: Are any variables inversely correlated with demand in specific zones?
A: Humidity shows an inverse relationship in Zone 3 at short lags (e.g., initial lags around −0.24 to −0.25; see printed per‑lag outputs), but the best‑lag correlation printed is near zero (−0.020 at lag 72). In PZ1/PZ2, humidity correlations at best lags are small positive.

Q: Did your analysis differ across zones? Why might that be?
A: Zone 3 shows stronger sensitivity to temperature and GDF (r up to 0.626) than Zones 1–2. Humidity behaves differently: clearer short‑lag negatives in PZ3 versus small positives in PZ1/PZ2. These zone differences likely reflect load composition and cooling sensitivity.

---

### 🌀 4. Lag Effects & Time Dependency
Q: Did you observe any lagged effects where past weather conditions predict current power usage?
A: Yes. The `CORRELATION SUMMARY TABLE` shows temperature leading demand by 12 steps (≈2h) in PZ1/PZ2 and 24 steps (≈4h) in PZ3; GDF leads by 48 steps (≈8h) across zones. Humidity effects are weaker (see per‑lag prints). ACF/PACF plots (`ACF/PACF - PZ1/PZ2/PZ3`) show strong short‑lag autocorrelation.

Q: How did you analyze lag (e.g., shifting features, plotting lag correlation)?
A: Computed shifted Pearson correlations over candidate lags and summarized best lags in the `CORRELATION SUMMARY TABLE`; inspected ACF/PACF (`ACF - PZx`, `PACF - PZx`).

Q: What lag intervals appeared most relevant and why?
A: Best lags (10‑min steps): temp → demand at 12 (PZ1/PZ2) and 24 (PZ3); GDF at 48 across zones; DF at 24; WS at 24. These align with plausible physical/behavioral delays (warming/solar accumulation and demand response).

---

### ⚠️ 5. Data Quality & Sensor Anomalies
Q: Did you detect any outliers in the weather or consumption readings?
A: Yes. Printed summary ("COMPREHENSIVE ANOMALY DETECTION ANALYSIS"): PZ2 IQR outliers 7 (0.01%), PZ3 IQR outliers 1191 (2.27%). Environmental: temp 142 (0.27%), humidity 291 (0.56%), WS 0 (0.00%), GDF 2315 (4.42%), DF 4571 (8.72%). Visuals: `{col} - Time Series with Outliers`, `{col} - Box Plot`, `{col} - Distribution with Outlier Bounds`.

Q: How did you identify and treat these anomalies?
A: Identified via Z‑score and IQR methods (printed counts) with visual checks via the plots above. Since most extremes are physically plausible (e.g., summer cooling peaks), I retain them. For modeling, I’ll consider robust approaches (log/Box‑Cox on demand, RobustScaler/QuantileTransformer), and cap only clear sensor glitches, not genuine peaks.

Q: What might be the impact of retaining or removing them in your model?
A: Removing anomalies risks erasing genuine peaks the model must forecast (e.g., Zone 3 summer surges). Retaining preserves seasonality and peak behavior but increases error variance. The compromise: retain genuine peaks, mitigate sensor glitches, and use robust scaling/metrics so models learn peaks without being dominated by a few extremes.



---

## 🛠️ Week 2: Feature Engineering & Deep Learning Preparation

### 🔄 1. Sequence Construction & Lookback Windows

Q: How did you determine the optimal lookback window size for your sequence models?  
A: I'm treating the lookback window size as a tunable hyperparameter informed by the EDA:

- For a **univariate model** (training on power only), PACF/ACF plots suggested strong early lags, so I used a **short window** of around *3-6 lags*.
- For a **multivariate model** (training on power + weather + time features), I chose a **longer window** (e.g., 24-48 lags) to capture slower weather effects, suggested by these lagged features correlating significantly with power consumption.

Q: What challenges did you face when converting the time-series data into input/output sequences?  
A: The main challenges were understanding the 3D shape the model expects [samples, time_steps, features], keeping features vs targets separate (univariate vs multivariate, single vs multi-target), and handling sequence-to-one vs sequence-to-sequence outputs in pipeline.

Q: How did you handle cases where the lookback window extended beyond the available data?  
A: I loop over the indices in the range: `lookback_window, len(data) - forecast_horizon + 1` to ensure I don't extend beyond the available data.

---

### ⚖️ 2. Feature Scaling & Transformation

Q: Which normalization or standardization techniques did you apply to the features, and why?  
A: I used `StandardSclaer` (z-score standardization) since neural networks train more stably when inputs are centered with unit variance. I fit the scaler only on the training split and transformed val/test with the same parameters to prevent leakage.

Q: Did you engineer any cyclical time features (e.g., sine/cosine transforms for hour or day)? How did these impact model performance?  
A: I added the following cyclical time features:

`hour_sin`, `hour_cos`; `day_sin`, `day_cos`; `month_sin`, `month_cos`; `day_of_year_sin`, `day_of_year_cos`

- This was done to encode the circular nature of time (e.g. for the model to understand that 23:00 and 00:00 are adjacent). I will discover which features are useful during week 3's tasks.
- I have a gut feeling that having both a sine and cosine transformation for each time feature may be redundant, again this will be explored during week 3's tasks.

Q: How did you address potential data leakage during scaling or transformation?  
A: I split the data chronologically first, then fit the `StandardSclaer` on the train set only, and windowing is performed after splitting, so sequences do not cross split boundaries.

---

### 🧩 3. Data Splitting & Preparation

Q: How did you split your data into training, validation, and test sets to ensure temporal integrity?  
A: I used a **sequential split**: first *70% for training*, next *15% for validation*, final *15% for test* (**no shuffling**). This preserves time order and simulates real forecasting conditions.

Q: What considerations did you make to prevent information leakage between splits?  
A: The splits must not be random, they should be **chronological**. Scaling is fit on train **only**, and as previously stated, sliding windows are built *within each split* so no sample uses future information across boundaries.

Q: How did you format your data for use with PyTorch DataLoader or TensorFlow tf.data.Dataset?  
A: After scaling, I built sliding windows that support either seq2one or seq2seq modeling, these are then turned into tensors and DataLoaders are created with batching.

---

### 📈 4. Feature-Target Alignment

Q: How did you align your input features and target variables for sequence-to-one or sequence-to-sequence forecasting?  
A: The windowing function takes feature_cols and target_cols explicitly and aligns them via:

- `in_len` (lookback length)
- `out_horizon` (how far ahead the first target is)
- `out_len` (number of future steps if seq2seq)

For seq2one, `y` is the target at `t + out_horizon`; for seq2seq, `y` spans `[t + out_horizon … t + out_horizon + out_len - 1]`.

Q: Did you encounter any issues with misalignment or shifting of targets? How did you resolve them?  
A: I printed the shapes and sampling of a few example windows to verify the last timestep in `X` algins with the first target in `y`.

---

### 🧪 5. Data Quality & Preprocessing

Q: What preprocessing steps did you apply to handle missing values or anomalies before modeling?  
A: I explicitly checked for NaNs in generated windows and flagged any occurences, none were found across all tested lookback windows.

Q: How did you verify that your data pipeline produces consistent and reliable outputs for model training?  
A:

- Sanity prints of all shapes after each step (splits, windows, loaders)
- NaN checks on `X` and `y`
- A batch sample from the `DataLoader` to confirm shapes `X: [B, in_len, F]`, `y: [B, K]` (or `[B, out_len, K]`)



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
A: 