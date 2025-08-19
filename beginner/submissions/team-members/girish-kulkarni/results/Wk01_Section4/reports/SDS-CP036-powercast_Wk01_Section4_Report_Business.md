# 💼 Week 1 – Section4: Anomalies & Outliers (Business-Friendly Report)

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
- Time series with control limits: `plots/section4_daily_ts_anoms.png`  
- Boxplot of daily totals: `plots/section4_box_daily.png`  
- Histogram for distribution context: `plots/section4_hist_daily.png`

## What we computed
- Canonical DateTime & zone aliasing; **Total_kW** across zones.  
- **Daily totals** and a simple **z-score**-based anomaly detector (±3σ).  
- A short list of the most extreme days for review.
