# SDS-CP036-powercast — Wk01 Consolidated Business Report (Inline Plots v2)

Generated on: 2025-08-15 23:51:41
Project root: `/home/6376f5a9-d12b-4255-9426-c0091ad440a7/Powercast`

Includes Sections: 1, 2, 3, 4, 5


## Table of Contents
- [Section 1 — 💼 Week 1 – Section1: Time Consistency & Structure (Business-Friendly Report)](#section-1)
- [Section 2 — 💼 Week 1 – Section2: Temporal Trends (Business-Friendly Report)](#section-2)
- [Section 3 — 💼 Week 1 – Section3: Drivers & Correlations (Business-Friendly Report)](#section-3)
- [Section 4 — 💼 Week 1 – Section4: Anomalies & Outliers (Business-Friendly Report)](#section-4)
- [Section 5 — 💼 Week 1 – Section5: Baseline Forecast & Capacity Readiness (Business-Friendly Report)](#section-5)

## Section 1 — 💼 Week 1 – Section1: Time Consistency & Structure (Business-Friendly Report)

## Dataset
Using file: **Tetuan City power consumption.csv**

## Key Questions Answered
**Q1: Are there any missing or irregular timestamps in the dataset? How did you verify consistency?**  
I created a canonical `DateTime` column and inspected gaps between consecutive records to detect irregularities.

**Q2: What is the sampling frequency and are all records spaced consistently?**  
I measured the time deltas between consecutive rows. The most common spacing is **0 days 00:10:00**, suggesting the intended sampling cadence.

**Q3: Did you encounter any duplicates or inconsistent `DateTime` entries?**  
I found **0** duplicate timestamps (exact same `DateTime`). These could be reviewed or deduplicated based on your business rules.

## Plain-English Notes
- Built a single `DateTime` column from whatever the dataset provided (flexible parsing with comma/semicolon delimiters and both month-first/day-first formats).
- Looked for missing times and uneven gaps using the distribution of time differences.
- Business takeaway: “On average, there’s a reading roughly every 0 days 00:10:00 — i.e., about 10 minute(s) per record.”

### Visuals

_No plot files found for this section._

## Section 2 — 💼 Week 1 – Section2: Temporal Trends (Business-Friendly Report)

## Dataset
Using file: **Tetuan City power consumption.csv**  
Period: **2017-01-01 00:00:00 → 2017-12-30 23:50:00**  
Rows: **52,416**

## Key Questions Answered
**Q1: What daily or weekly patterns are observable in power consumption across the three zones?**  
- The **line plot** of daily averages (Total & Zones) shows overall movement and relative contribution by zone.  
- By weekday, average total usage peaks on **Thu** based on the dataset's mean profile.

**Q2: Are there seasonal or time-of-day peaks and dips in energy usage?**  
- The **heatmap** for Zone 1 (kitchen proxy) highlights typical time-of-day peaks; the highest average hour is around **20:00**.  
- Broader seasonal effects can be explored by comparing monthly averages (extendable in this section if needed).

**Q3: Which visualizations helped you uncover these patterns?**  
- **Line plot** (daily averages): `results/Wk01_Section2/plots/section2_daily_averages.png`  
- **Box plot** (by day of week): `results/Wk01_Section2/plots/section2_box_by_dow.png`  
- **Heatmap** (hour vs day for Zone 1): `results/Wk01_Section2/plots/section2_heatmap_zone1.png`

## What we computed
- Canonical **DateTime** and zone aliasing (**Zone 1/2/3 → Sub_metering_1/2/3**).  
- **Daily averages** of total and per-zone consumption.  
- **Day-of-week distribution** (box plot) for total consumption.  
- **Hour × day heatmap** for Zone 1.

### Visuals

![Section 2 — section2_daily_averages.png](results/Wk01_Section2/plots/section2_daily_averages.png)
![Section 2 — section2_box_by_dow.png](results/Wk01_Section2/plots/section2_box_by_dow.png)
![Section 2 — section2_heatmap_zone1.png](results/Wk01_Section2/plots/section2_heatmap_zone1.png)

## Section 3 — 💼 Week 1 – Section3: Drivers & Correlations (Business-Friendly Report)

## Dataset
Using file: **Tetuan City power consumption.csv**  
Period: **2017-01-01 00:00:00 → 2017-12-30 23:50:00**  
Rows: **52,416**

## Key Questions Answered
**Q1: Which factors appear most correlated with total and per-zone consumption?**  
- Total consumption driver: **Temperature (|r|=0.73)**  
- Zone 1 driver: **Temperature (|r|=0.73)**  
- Zone 2 driver: **Temperature (|r|=0.42)**  
- Zone 3 driver: **Temperature (|r|=0.61)**

**Q2: Are relationships linear or do they show thresholds?**  
- The **scatter plots** (daily means) help reveal linear vs. curved patterns and potential thresholds. Patterns tend to be smooth with some spread due to operational effects.

**Q3: Which visualizations helped you uncover these patterns?**  
- **Correlation matrix** of daily means: `results/Wk01_Section3/plots/section3_correlation_heatmap.png`  
- **Scatter plots** contrasting key drivers vs Total_kW: plots/section3_scatter_temp_total.png, plots/section3_scatter_humidity_total.png

## What we computed
- Canonical **DateTime** and zone aliasing (**Zone 1/2/3 → Sub_metering 1/2/3**).  
- Daily means to stabilize noise.  
- Pearson correlations among consumption (Total + Zones) and environmental features.

### Visuals

![Section 3 — section3_correlation_heatmap.png](results/Wk01_Section3/plots/section3_correlation_heatmap.png)
![Section 3 — section3_scatter_humidity_total.png](results/Wk01_Section3/plots/section3_scatter_humidity_total.png)
![Section 3 — section3_scatter_temp_total.png](results/Wk01_Section3/plots/section3_scatter_temp_total.png)

## Section 4 — 💼 Week 1 – Section4: Anomalies & Outliers (Business-Friendly Report)

## Dataset
Using file: **Tetuan City power consumption.csv**  
Period: **2017-01-01 00:00:00 → 2017-12-30 23:50:00**  
Rows: **52,416**

## Key Questions Answered
**Q1: Are there anomalous days in total energy consumption? When?**  
- Detected **4** anomalous day(s) using a ±3σ z-score rule on **daily totals**.
- Top examples: 2017-07-25 (z=3.23), 2017-07-24 (z=3.11), 2017-07-27 (z=3.10), 2017-07-26 (z=3.03)

**Q2: What could explain these anomalies?**  
- Check overlaps with weather or events (e.g., heat waves, holidays, maintenance). You can cross-reference with the Tetuan environmental columns or your calendar.

**Q3: Which visualizations helped you uncover these?**  
- Time series with control limits: `results/Wk01_Section4/plots/section4_daily_ts_anoms.png`  
- Boxplot of daily totals: `results/Wk01_Section4/plots/section4_box_daily.png`  
- Histogram for distribution context: `results/Wk01_Section4/plots/section4_hist_daily.png`

## What we computed
- Canonical DateTime & zone aliasing; **Total_kW** across zones.  
- **Daily totals** and a simple **z-score**-based anomaly detector (±3σ).  
- A short list of the most extreme days for review.

### Visuals

![Section 4 — section4_daily_ts_anoms.png](results/Wk01_Section4/plots/section4_daily_ts_anoms.png)
![Section 4 — section4_box_daily.png](results/Wk01_Section4/plots/section4_box_daily.png)
![Section 4 — section4_hist_daily.png](results/Wk01_Section4/plots/section4_hist_daily.png)

## Section 5 — 💼 Week 1 – Section5: Baseline Forecast & Capacity Readiness (Business-Friendly Report)

## Dataset
Using file: **Tetuan City power consumption.csv**  
Period: **2017-01-01 00:00:00 → 2017-12-30 23:50:00**  
Rows: **52,416**

## Key Questions Answered
**Q1: What does the short-term baseline forecast look like?**  
- A simple **day-of-week average** model projects the next 7 days. Peak is **2018-01-04** (~**10444513 kW** daily).

**Q2: How should operations plan around expected peaks?**  
- Staff or schedule energy-intensive tasks away from forecasted peak days; consider shifting flexible loads to lower-demand days.

**Q3: Which visualizations helped?**  
- Recent actuals (last 60 days): `results/Wk01_Section5/plots/section5_recent_daily.png`  
- 7-day baseline forecast: `results/Wk01_Section5/plots/section5_forecast7d.png`

## What we computed
- Canonical DateTime & zone aliasing; **Total_kW** across zones.  
- **Daily totals** and **DoW mean** profile.  
- A lightweight **7-day baseline** forecast using day-of-week averages (extendable to richer models later).

### Visuals

![Section 5 — section5_recent_daily.png](results/Wk01_Section5/plots/section5_recent_daily.png)
![Section 5 — section5_forecast7d.png](results/Wk01_Section5/plots/section5_forecast7d.png)
