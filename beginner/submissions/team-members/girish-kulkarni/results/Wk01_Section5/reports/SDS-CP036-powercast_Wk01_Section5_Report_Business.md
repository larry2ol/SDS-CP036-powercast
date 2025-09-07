# 💼 Week 1 – Section5: Baseline Forecast & Capacity Readiness (Business-Friendly Report)

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
- Recent actuals (last 60 days): `plots/section5_recent_daily.png`  
- 7-day baseline forecast: `plots/section5_forecast7d.png`

## What we computed
- Canonical DateTime & zone aliasing; **Total_kW** across zones.  
- **Daily totals** and **DoW mean** profile.  
- A lightweight **7-day baseline** forecast using day-of-week averages (extendable to richer models later).
