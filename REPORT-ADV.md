# 🔴 Advanced Track

## ✅ Week 1: Setup & Exploratory Data Analysis (EDA)

---

### 🔑 Question 1:
**After converting the `DateTime` column and setting it as the index, are there any missing or irregular timestamps in the dataset? How did you check, and what is the impact if these are not addressed?**  
🎯 *Purpose: Tests ability to validate time series consistency and temporal indexing.*

💡 **Hint:**  
Convert to datetime, then use `.diff()` or `pd.date_range()` to check uniformity.  
Ensure every timestamp is spaced 10 minutes apart.  
Discuss whether missing timestamps should be forward-filled, interpolated, or removed.

✏️ *Your answer here...*

---

### 🔑 Question 2:
**What temporal patterns do you observe in the power consumption for each zone (daily, weekly, or seasonal)? How did you visualize and interpret these patterns?**  
🎯 *Purpose: Tests skill in detecting seasonality and time-based consumption behaviors.*

💡 **Hint:**  
Plot rolling averages or resample to daily/weekly intervals.  
Use line plots, boxplots grouped by hour/day.  
Do consumption levels spike during specific hours or days?

✏️ *Your answer here...*

---

### 🔑 Question 3:
**How are temperature, humidity, wind speed, and solar radiation correlated with power consumption in each zone? Which environmental variable appears most influential, and how did you determine that?**  
🎯 *Purpose: Tests reasoning about weather-to-demand relationships.*

💡 **Hint:**  
Use `.corr()` or `sns.heatmap()` to inspect correlations.  
You may compute correlation separately for each zone.  
Discuss whether solar radiation or humidity inversely/positively correlates with energy demand.

✏️ *Your answer here...*

---

### 🔑 Question 4:
**Are there lag effects between environmental conditions and energy usage? Did you test any lagged correlations?**  
🎯 *Purpose: Introduces concept of delayed influence, critical for time-aware feature engineering.*

💡 **Hint:**  
Use `df['temperature'].shift(1)` or cross-correlation functions.  
Plot lagged correlations (1 to 12 hours ahead).  
Identify if temperature or humidity 1–2 hours ago impacts current demand.

✏️ *Your answer here...*

---

### 🔑 Question 5:
**Were there any anomalies or outliers in the weather sensors or consumption readings? How did you detect and interpret them?**  
🎯 *Purpose: Tests ability to clean noisy sensor data and explain its effects.*

💡 **Hint:**  
Use `.describe()`, z-scores, or boxplots.  
Extreme solar radiation at night? Sudden consumption spikes?  
Consider visual inspection and removal or clipping strategies.

✏️ *Your answer here...*

---
