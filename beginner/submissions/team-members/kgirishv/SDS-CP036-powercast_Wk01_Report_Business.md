# SDS-CP036-powercast – Wk01 – Consolidated Business Report
_2025-08-09 22:30:42_

## Table of Contents
- [Section 1 – Business Report](#section-1-–-business-report)
- [Section 2 – Business Report](#section-2-–-business-report)
- [Section 3 – Business Report](#section-3-–-business-report)
- [Section 4 – Business Report](#section-4-–-business-report)
- [Section 5 – Business Report](#section-5-–-business-report)


## Section 1 – Business Report

_Source: results/Wk01_Section1/reports/SDS-CP036-powercast_Wk01_Section1_Report_Business.md_

## Key Questions Answered
**Q1: Are there any missing or irregular timestamps in the dataset? How did you verify consistency?**  
I converted the date and time into a single timeline and checked the gaps between readings. Most records are evenly spaced; I used the differences between rows to confirm the pattern.

**Q2: What is the sampling frequency and are all records spaced consistently?**  
I measured the time between consecutive rows. The most common spacing is **0 days 00:01:00**, which suggests the intended sampling rate.

**Q3: Did you encounter any duplicates or inconsistent `DateTime` entries?**  
I checked for repeated timestamps. I found **0** duplicates; these could be rechecked or removed if needed.

## Plain-English Notes
- I first made a proper date-time column.
- I looked for missing times and uneven gaps.
- If a business user asked, I’d explain the cadence like this: “On average, I get a reading roughly every 0 days 00:01:00 — so a new data point about once every 1 minute(s).”


---

## Section 2 – Business Report

_Source: results/Wk01_Section2/reports/SDS-CP036-powercast_Wk01_Section2_Report_Business.md_

## Key Questions Answered
**Q1: What daily or weekly patterns are observable in power consumption across the three zones?**  
I reviewed usage by day and week. Kitchen (Zone 1) spikes around meal times. Laundry (Zone 2) is busier on weekends. Water heating/AC (Zone 3) is steadier.

**Q2: Are there seasonal or time-of-day peaks and dips in energy usage?**  
Yes. The line chart shows daily trends; the heatmap highlights time-of-day peaks in the kitchen around lunch and dinner.

**Q3: Which visualizations helped you uncover these patterns?**  
- Line plot (daily averages)  
- Box plot (by day of week)  
- Heatmap (hour vs day for Zone 1)


---

## Section 3 – Business Report

_Source: results/Wk01_Section3/reports/SDS-CP036-powercast_Wk01_Section3_Report_Business.md_

## Key Questions Answered
**Q1: Which environmental variables correlate most with energy usage?**  
I compared temperature, humidity, wind, and sunlight with energy use in each zone. The AC/Water Heating zone showed the strongest link to temperature and wind; the kitchen was less weather-sensitive.

**Q2: Are any variables inversely correlated with demand in specific zones?**  
Yes. In warmer hours, water heating demand can drop (negative link with temperature), while cooling may increase.

**Q3: Did your analysis differ across zones? Why might that be?**  
Yes. Each zone powers different appliances: the kitchen follows meal schedules; laundry is sporadic; HVAC/water heating track outdoor conditions more closely.


---

## Section 4 – Business Report

_Source: results/Wk01_Section4/SDS-CP036-powercast_Wk01_Section4_Report_Business.md_

## Key Questions Answered

**Q1: Did I observe any lagged effects where past weather conditions predict current power usage?**  
Yes. I observed meaningful lagged relationships, especially between temperature/wind and energy usage in the HVAC/water heating zone.

**Q2: How did I analyze lag (e.g., shifting features, plotting lag correlation)?**  
I shifted hourly weather data by 0–12 hours and computed Pearson correlations against each zone’s usage, plotting correlation vs. lag.

**Q3: What lag intervals appeared most relevant and why?**  
- **Kitchen (Zone 1):** Temperature & humidity showed modest effects around 2–4 hours.  
- **Laundry (Zone 2):** Solar radiation showed a minor delayed effect; others were weak.  
- **HVAC/Water Heater (Zone 3):** Temperature and wind peaked around 3–6 hours, aligning with heating/cooling dynamics.

## Visuals

### Zone 1 (Kitchen)

![](plots/SDS-CP036-powercast_Wk01_Section4_lagcorr_Sub_metering_1.png)

### Zone 2 (Laundry)

![](plots/SDS-CP036-powercast_Wk01_Section4_lagcorr_Sub_metering_2.png)

### Zone 3 (Water Heater & AC)

![](plots/SDS-CP036-powercast_Wk01_Section4_lagcorr_Sub_metering_3.png)

### Temperature vs Energy – All Zones

![](plots/SDS-CP036-powercast_Wk01_Section4_lagcorr_temperature_all_zones.png)

## Practical Takeaways

- Short-term forecasts (2–6 hours ahead) can improve scheduling of HVAC and heavy appliances.
- I can automate pre-cooling/heating when temperature/wind trends indicate upcoming load.
- Adding lagged weather features should improve short-term demand predictions.


---

## Section 5 – Business Report

_Source: results/Wk01_Section5/SDS-CP036-powercast_Wk01_Section5_Report_Business.md_

## Key Questions Answered

**Q1: Did I detect any outliers in the weather or consumption readings?**  
Yes. I found outliers across several features using boxplots/IQR rules and histograms. Sub-meter readings occasionally had extreme spikes, and weather features showed sporadic high/low values.

**Q2: How did I identify and treat these anomalies?**  
I used IQR-based clipping (to cap extreme values) and replaced negative sub-meter readings with blanks (then filled small gaps). I also forward-/back-filled short missing stretches.

**Q3: What might be the impact of retaining or removing them in my model?**  
Capping/removing extremes reduces noise and helps models generalize, while retaining them can cause unstable forecasts. For production systems, I would keep this cleaning to improve reliability.

## Missing Values (Before Cleaning)

- Sub_metering_1: 0
- Sub_metering_2: 0
- Sub_metering_3: 0
- relative_humidity_2m: 0
- shortwave_radiation: 0
- temperature_2m: 0
- wind_speed_10m: 0

## Missing Values (After Cleaning)

- Sub_metering_1: 0
- Sub_metering_2: 0
- Sub_metering_3: 0
- relative_humidity_2m: 0
- shortwave_radiation: 0
- temperature_2m: 0
- wind_speed_10m: 0

## Visual Evidence

**Before – Histograms**  
![](plots/SDS-CP036-powercast_Wk01_Section5_Plot_Hist_Before.png)

**Before – Boxplots**  
![](plots/SDS-CP036-powercast_Wk01_Section5_Plot_Box_Before.png)

**After – Histograms**  
![](plots/SDS-CP036-powercast_Wk01_Section5_Plot_Hist_After.png)

**After – Boxplots**  
![](plots/SDS-CP036-powercast_Wk01_Section5_Plot_Box_After.png)

