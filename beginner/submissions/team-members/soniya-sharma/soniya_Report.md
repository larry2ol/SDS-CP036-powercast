# 🔴 PowerCast – Advanced Track

## ✅ Week 1: Setup & Exploratory Data Analysis (EDA)

---

### 🧭 1. Time Consistency & Structure

Q: Are there any missing or irregular timestamps in the dataset? How did you verify consistency?
A:The dataset was checked for missing or irregular timestamps by calculating time differences between consecutive records. All intervals are exactly 10 minutes, with no missing or irregular timestamps detected.

Q: What is the sampling frequency and are all records spaced consistently?
A: The dataset is sampled at 10-minute intervals. After inspection, all records are spaced consistently at 10 minute intervals.

Q: Did you encounter any duplicates or inconsistent `DateTime` entries?
A: No duplicate or inconsistent `DateTime` entries were found; all datetime indexes are unique.

---

### 📊 2. Temporal Trends & Seasonality

Q: What daily or weekly patterns are observable in power consumption across the three zones?
A: All zones show clear daily cycles with peak consumption during business hours and lower usage overnight. Weekly patterns reveal lower consumption on weekends, especially in Zone 3.

Q: Are there seasonal or time-of-day peaks and dips in energy usage?
A: Yes, summer months (June-September) show the highest consumption, with pronounced peaks in Zone 3. Winter months have lower usage. Time-of-day analysis shows consistent peaks during daytime hours.

Q: Which visualizations helped you uncover these patterns?
A: Line plots of raw and resampled data (daily, weekly, monthly), hourly and day-of-week averages, and monthly boxplots were used to visualize and confirm these patterns.

---

### 🌦️ 3. Environmental Feature Relationships

Q: Which environmental variables (temperature, humidity, wind speed, solar radiation) correlate most with energy usage?
A: Temperature shows the strongest positive correlation with power consumption across all zones, especially Zone 3. Solar radiation (General Diffuse Flows) also shows positive but weaker correlations. 

Q: Are any variables inversely correlated with demand in specific zones?
A: Humidity is moderately negatively correlated with power consumption in all zones. Wind speed shows weak positive correlations.

Q: Did your analysis differ across zones? Why might that be?
A: Yes, Zone 3 consistently shows stronger correlations and higher variability, likely due to differences in building types or land use (e.g., industrial vs. residential).
---

### 🌀 4. Lag Effects & Time Dependency

Q: Did you observe any lagged effects where past weather conditions predict current power usage?
A: Yes, lagged cross-correlation analysis shows that temperature and solar radiation have persistent positive effects on power consumption for several hours, especially in Zone 3.

Q: How did you analyze lag (e.g., shifting features, plotting lag correlation)?
A: Lag effects were analyzed using cross-correlation plots at both 10-minute and hourly frequencies, as well as by shifting weather features and plotting their correlations with power consumption at various lags.

Q: What lag intervals appeared most relevant and why?
A: Short lags (up to 6-8 hours) for temperature and solar radiation are most relevant, reflecting daily cycles and the immediate impact of weather on energy demand.

---

### ⚠️ 5. Data Quality & Sensor Anomalies

Q: Did you detect any outliers in the weather or consumption readings?
A: Yes, significant outliers were detected in Zone 3 power consumption (mostly during summer months) and in solar radiation features (Diffuse Flows).

Q: How did you identify and treat these anomalies?
A: Outliers were identified using boxplots and IQR-based thresholds. For weather features with strong skewness, log transformation was applied to stabilize variance and reduce the influence of extreme values.

Q: What might be the impact of retaining or removing them in your model?
A: Outliers in Zone 3 reflect genuine seasonal peaks and should be retained to preserve real-world patterns. Log transformation of skewed weather features improves model robustness without discarding important data.
