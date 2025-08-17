# 🔴 PowerCast – Advanced Track

## ✅ Week 1: Setup & Exploratory Data Analysis (EDA)

---

### 🧭 1. Time Consistency & Structure

Q: Are there any missing or irregular timestamps in the dataset? How did you verify consistency?
A: No, there are no missing timestamps. Consistency was verified by generating a complete expected time range at a 10-minute frequency and comparing it to the actual timestamps in the DataFrame's index. The count of missing timestamps was 0.

Q: What is the sampling frequency and are all records spaced consistently?
A: The sampling frequency is 10 minutes. All records are spaced consistently at this interval, as verified by calculating the time differences between consecutive timestamps and observing that the only difference present was 10 minutes.

Q: Did you encounter any duplicates or inconsistent `DateTime` entries?
A:No, there were no duplicate or inconsistent DateTime entries. This was verified by checking for duplicate values in the DataFrame's index, which resulted in a count of 0.

---

### 📊 2. Temporal Trends & Seasonality

Q: What daily or weekly patterns are observable in power consumption across the three zones?
A: Clear daily patterns were observed across all zones, with lower consumption in the early morning, increasing during the day, and peaking in the evening (6 PM - 9 PM). Weekly patterns showed higher consumption during weekday work hours compared to weekends. Weekend mornings had a later ramp-up in consumption.

Q: Are there seasonal or time-of-day peaks and dips in energy usage?
A: Yes, distinct time-of-day peaks were observed in the evening (6 PM - 9 PM) and dips in the early morning (3 AM - 6 AM). While not explicitly visualized as seasonal plots, the analysis across a year of data implicitly includes seasonal influences on these patterns.

Q: Which visualizations helped you uncover these patterns?
A: The Average Daily Power Consumption Line Plots clearly showed the 24-hour cycle and time-of-day peaks/dips. The Heatmap of Consumption by Weekday and Hour (and Weekend and Hour) revealed the differences in consumption patterns between weekdays and weekends.

---

### 🌦️ 3. Environmental Feature Relationships

Q: Which environmental variables (temperature, humidity, wind speed, solar radiation) correlate most with energy usage?
A: Temperature showed the strongest positive correlation with power consumption across all zones (0.38 - 0.49). Wind Speed had a weak positive correlation (0.14 - 0.27). General diffuse flows showed weak positive correlations, while diffuse flows had very weak positive or negative correlations depending on the zone.

Q: Are any variables inversely correlated with demand in specific zones?
A: Yes, Humidity was inversely correlated with power consumption in all three zones (-0.23 to -0.29). Diffuse flows also showed a weak inverse correlation with Zone 3 (-0.038).

Q: Did your analysis differ across zones? Why might that be?
A: Yes, the strength of correlations varied across zones (e.g., Temperature correlation was slightly stronger in Zone 3, Wind Speed correlation was notably stronger in Zone 3). This could be due to differences in building types, activities, HVAC systems, or localized environmental conditions within each zone.

---

### 🌀 4. Lag Effects & Time Dependency

Q: Did you observe any lagged effects where past weather conditions predict current power usage?
A: Yes, lagged effects were observed, indicated by correlations between past environmental conditions and current power usage.

Q: How did you analyze lag (e.g., shifting features, plotting lag correlation)?
A: Lag was analyzed by shifting environmental features backward in time (by 10, 60, and 120 minutes), calculating the correlation of these lagged features with current power consumption, and visualizing the results using a heatmap.

Q: What lag intervals appeared most relevant and why?
A: Longer lag intervals (60 and 120 minutes) appeared slightly more relevant for some features (like Temperature and Humidity) as the correlations were generally stronger compared to the 10-minute lag. This suggests a delayed impact of these environmental changes on power consumption.

---

### ⚠️ 5. Data Quality & Sensor Anomalies

Q: Did you detect any outliers in the weather or consumption readings?
A: Yes, outliers were detected in several weather and power consumption readings using both box plots and the IQR statistical method.

Q: How did you identify and treat these anomalies?
A: Outliers were identified visually using box plots and statistically using the IQR method (values outside 1.5 * IQR from Q1 and Q3). In this EDA phase, anomalies were identified but not yet treated.

Q: What might be the impact of retaining or removing them in your model?
A: Retaining outliers can skew statistical measures and negatively impact outlier-sensitive models, though they might represent real events. Removing them can improve model performance for sensitive models but may lead to loss of information or distort the data distribution if not done carefully.
