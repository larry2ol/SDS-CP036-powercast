# 🔴 PowerCast – Advanced Track

## ✅ Week 1: Setup & Exploratory Data Analysis (EDA)

---

### 🧭 1. Time Consistency & Structure

Q: Are there any missing or irregular timestamps in the dataset? How did you verify consistency?
A: There are no missing timestamps in the dataset, I Verified missingness by applying the isnull().sum ()
 functions on the 'DateTime' columns,  This function allowed me to pinpoint the specific entry that have null values. I calculated the time difference between consecutive DateTime entries using .diff(). I then checked the value counts of these time differences. Ideally, for consistent data, there should be only one unique time difference value (in this case, 10 minutes). By calling the nunique(), i verified uniqueness. 


Q: What is the sampling frequency and are all records spaced consistently?
A: The Frequency of the data is 10 minutes and all records are spaced consistently at a 10-minute interval.

Q: Did you encounter any duplicates or inconsistent `DateTime` entries?
A: I did not encounter any duplicate DateTime entries in the dataset. This was confirmed by showing a sum of 0 for duplicated DateTime values.

---

### 📊 2. Temporal Trends & Seasonality 

Q: What daily or weekly patterns are observable in power consumption across the three zones?
A: All the three zones show distinct daily pattern. But commonly, power consumption is consistently lower during the night and early morning hours, increasing during the day and peaking in the evening (around 8pm). The weekly patterns for the three zones reveal fluctuations throughout the year but also the presence of weekly cycles. These cycles, which are more apparent in the daily plots, contribute to the overall trend of higher power consumption during certain periods (e.g., summer months for Zone 1 and Zone 2, and a significant peak in Zone 3 during late summer).

Q: Are there seasonal or time-of-day peaks and dips in energy usage?
A: Yes. The energy usage is consistently lowest during the early morning hours (around 3 AM to 6 AM) for the three zones. There are two predictable peaks in daily energy usage: a smaller peak in the late morning/early afternoon and a more significant peak in the evening (around 7 PM to 9 PM). This reliable pattern is consistent across all zones, with only the magnitude of consumption varying.The monthly usage plot shows distinct seasonal trends. Both Zone 1 and Zone 2 show higher power consumption during the summer months (July and August), and generally lower during the spring and autumn. Zone 3, on the other hand, presents a unique seasonal pattern. With a very pronounced peak in July and August, and significantly lower consumption during the rest of the year, particularly in the winter months, Zone 3's energy usage profile stands out from the other two zones, piquing our curiosity for further investigation.

Q: Which visualizations helped you uncover these patterns?
A: Daily Power Consumption Patterns, Weekly Power Consumption Patterns, Monthly Power Consumption Pattern , Average Power Consumption by Hour of Days, Average Power Consumption by Month and Resampled Zone Power Consumption Over Time. For example, the latter plot resamples and group the data to daily and weekly averages and plotting them over time, to identify the general daily and weekly patterns

---

### 🌦️ 3. Environmental Feature Relationships

Q: Which environmental variables (temperature, humidity, wind speed, solar radiation) correlate most with energy usage?
A: Across zones, Temperature shows the strongest positive correlations, followed by Humidity, which shows a negative correlation.

Q: Are any variables inversely correlated with demand in specific zones?
A: Humidity  and diffuse flows  in zone 3 and Humidity only in zones 1 and 2

Q: Did your analysis differ across zones? Why might that be?
A: Yes. As a result, it's essential to adopt a zone-specific approach to energy analysis, considering the unique characteristics of each zone. These characteristics include different building types, occupancy patterns, or usage behaviours; varying exposure to environmental factors (e.g., sunlight, wind); distinct equipment, insulation, or operational schedules; and localised microclimates or infrastructure differences. These factors cause the relationship between environmental variables (like temperature, humidity, wind speed, and solar radiation) and power consumption to vary for each zone, resulting in different correlation strengths and directions. This variability keeps us engaged and intrigued, as we strive to understand the unique energy dynamics of each zone.

---

### 🌀 4. Lag Effects & Time Dependency

Q: Did you observe any lagged effects where past weather conditions predict current power usage?
A: Yes, based on the analysis, there are lagged effects where past weather conditions, particularly temperature, can predict current power usage.
The correlation analysis with shifted temperature data (cell cc5a7071) shows that the temperature from previous time steps (lags 1, 2, and 3) still has a positive correlation with the current power consumption in all zones, with the correlation slightly increasing with lag up to 3 or 4 steps for Zone 3. The Granger Causality testsv also indicate that past temperature values have a statistically significant impact on predicting current power consumption for all three zones up to a lag of 4. The p-values for the Granger causality tests are very close to zero for all tested lags, suggesting that past temperature values are useful in forecasting current power consumption.This suggests that there is a time lag between changes in temperature and their full impact on power consumption.

Q: How did you analyze lag (e.g., shifting features, plotting lag correlation)?
A: I shifted the environmental features (Temperature, Humidity) by different lag intervals (1, 2, and 3 time steps) and then calculated the correlation between these lagged environmental variables and the current power consumption in each zone. This helps to see if past environmental conditions are correlated with current power usage.
Secondly, engaging in a visual inspection, I plotted the lagged temperature data against time to observe how past temperature trends align with current patterns in the data. This hands-on approach allows for a deeper understanding of the data.
Lastly, I conducted Granger Causality tests. This powerful statistical test helps determine if past values of one time series (e.g., Temperature) can predict future values of another time series (e.g., Zone Power Consumption), beyond what can be predicted by the past values of the second time series alone. I tested this for Temperature and each of the power consumption zones up to a maximum lag of 4, unveiling potential predictive relationships.


Q: What lag intervals appeared most relevant and why?
A: The most relevant lag intervals for temperature in predicting power consumption appear to be lags 1 through 4 for two reasons. 
The correlation values between lagged temperature and power consumption are highest for smaller lags (1 to 3), and for Zone 3, the correlation with temperature continues to increase slightly up to lag 4. This suggests that recent past temperatures have a notable relationship with current power demand. In summary, recent past temperatures, particularly those within lags 1 to 4, are significantly correlated with current power demand.
The Granger Causality tests, a reliable and widely accepted method, showed very low p-values for all tested lags (up to 4) when examining the relationship between temperature and power consumption in each zone. This strong statistical significance indicates that past temperature values up to 4 lags are valuable in predicting current power consumption, reinforcing the robustness of our methodology.


---

### ⚠️ 5. Data Quality & Sensor Anomalies

Q: Did you detect any outliers in the weather or consumption readings?
A: Yes, I detected outliers in both the weather and power consumption readings.

Q: How did you identify and treat these anomalies?
A: Using the Interquartile Range (IQR) method, I identified outliers in several columns, including Temperature, Humidity, general diffuse flows, diffuse flows, Zone 2 Power Consumption, and Zone 3 Power Consumption. The total number of outliers identified by this method was 8659, representing about 17% of the data. Wind Speed and Zone 1 Power Consumption did not show outliers using this method. The boxplots I generated not only visually confirmed the presence of outliers in several of the variables, particularly in the diffuse flows and general diffuse flows columns, as well as some of the power consumption zones, but also provided an intuitive representation of the data. I also used the Isolation Forest algorithm to identify anomalies, which can also be considered outliers. 

I treated the anomalies by filtering the DataFrame to keep only the rows that were identified as normal data points (where the 'anomaly' column was equal to 1). This thorough process effectively removed the rows containing anomalies from the dataset. I then dropped the 'anomaly' column as it was no longer needed. This approach allowed me to identify and remove the anomalies based on the Isolation Forest model's assessment.


Q: What might be the impact of retaining or removing them in your model?
A: Outliers can heavily influence descriptive statistics like the mean and standard deviation, potentially giving a misleading picture of the typical data distribution. Many time series models, especially those based on statistical assumptions like forecasting methods are sensitive to extreme values. Outliers misrepresent the true underlying patterns, leading to a poor fit and inaccurate predictions, inflate the variance of the data, making it harder for the model to identify significant patterns and increasing the uncertainty of predictions. Outliers might be misinterpreted as genuine patterns or shifts in the data, leading to incorrect conclusions about the underlying process
